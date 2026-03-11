#!/usr/bin/env python3

from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from ultralytics import YOLO

from svea_vision_msgs.msg import Object, StampedObjectArray


NO_TRACK_ID = 65535


def replace_base(old, new):
    split_last = lambda xs: (xs[:-1], xs[-1])
    is_private = new.startswith("~")
    is_global = new.startswith("/")
    assert not (is_private or is_global)
    ns, _ = split_last(old.split("/"))
    ns += new.split("/")
    return "/".join(ns)


def normalize_class_names(model_names):
    if isinstance(model_names, dict):
        return {int(idx): label for idx, label in model_names.items()}
    if isinstance(model_names, list):
        return {idx: label for idx, label in enumerate(model_names)}
    return {}


@dataclass
class Detection:
    box: tuple[int, int, int, int]
    label: str
    score: float
    track_id: int
    source: str


class ModelRunner:
    def __init__(
        self,
        node: Node,
        source_name: str,
        model_path: str,
        only_objects: list[str],
        skip_objects: list[str],
        image_width: int,
        image_height: int,
        use_cuda: bool,
        id_offset: int = 0,
    ):
        self.node = node
        self.source_name = source_name
        self.model_path = model_path
        self.only_objects = set(only_objects)
        self.skip_objects = set(skip_objects)
        self.image_width = image_width
        self.image_height = image_height
        self.use_cuda = use_cuda
        self.id_offset = id_offset

        self.model = self._load_model()
        self.class_names = normalize_class_names(getattr(self.model, "names", {}))
        self.label_to_class = {label: idx for idx, label in self.class_names.items()}

        self.detect_kwargs = {
            "persist": True,
            "conf": 0.5,
            "verbose": False,
            "imgsz": (self.image_height, self.image_width),
        }
        if self.model_path.endswith(".engine"):
            self.detect_kwargs["device"] = 0

        if self.only_objects:
            class_ids = [
                self.label_to_class[label]
                for label in sorted(self.only_objects)
                if label in self.label_to_class
            ]
            if class_ids:
                self.detect_kwargs["classes"] = class_ids
            else:
                self.node.get_logger().warn(
                    f"{self.source_name}: none of {sorted(self.only_objects)} matched model classes"
                )

    def _load_model(self):
        import torch

        try:
            model = YOLO(self.model_path, task="detect")
            self.node.get_logger().info(
                f"{self.source_name}: loaded YOLO model from {self.model_path}"
            )
        except Exception as first_error:
            self.node.get_logger().warn(
                f"{self.source_name}: standard model load failed: {first_error}"
            )
            original_load = torch.load

            try:
                def safe_load(*args, **kwargs):
                    kwargs.setdefault("weights_only", False)
                    return original_load(*args, **kwargs)

                torch.load = safe_load
                model = YOLO(self.model_path)
                self.node.get_logger().info(
                    f"{self.source_name}: loaded YOLO model with weights_only=False"
                )
            except Exception as second_error:
                self.node.get_logger().error(
                    f"{self.source_name}: failed to load YOLO model: {second_error}"
                )
                raise RuntimeError(
                    f"{self.source_name}: could not load YOLO model from {self.model_path}"
                ) from second_error
            finally:
                torch.load = original_load

        if self.use_cuda:
            try:
                if self.model_path.endswith(".pt"):
                    model.to("cuda")
                    self.node.get_logger().info(
                        f"{self.source_name}: CUDA enabled for PyTorch model"
                    )
                else:
                    self.node.get_logger().info(
                        f"{self.source_name}: TensorRT engine loaded; skipping model.to('cuda')"
                    )
            except Exception as cuda_error:
                self.node.get_logger().warn(
                    f"{self.source_name}: CUDA init failed, falling back to default device: {cuda_error}"
                )

        return model

    def run(self, frame_rgb):
        return self.model.track(frame_rgb, **self.detect_kwargs)[0]

    def detections_from_result(self, result):
        if result.boxes is None or len(result.boxes) == 0:
            return []

        result_cpu = result.cpu().numpy()
        boxes = result_cpu.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = getattr(boxes, "xyxy", [])
        confs = getattr(boxes, "conf", [])
        class_ids = getattr(boxes, "cls", [])
        track_ids = getattr(boxes, "id", None)
        is_track = bool(getattr(boxes, "is_track", False)) and track_ids is not None

        if not is_track:
            track_ids = [None] * len(xyxy)

        detections = []
        for box, score, class_id, track_id in zip(xyxy, confs, class_ids, track_ids):
            label = self.class_names.get(int(class_id), str(int(class_id)))

            if self.skip_objects and label in self.skip_objects:
                continue
            if self.only_objects and label not in self.only_objects:
                continue

            u1, v1, u2, v2 = [int(round(value)) for value in box]
            u1 = max(0, min(u1, self.image_width - 1))
            v1 = max(0, min(v1, self.image_height - 1))
            u2 = max(u1 + 1, min(u2, self.image_width))
            v2 = max(v1 + 1, min(v2, self.image_height))

            if track_id is None:
                object_id = NO_TRACK_ID
            else:
                object_id = int(track_id) + self.id_offset
                object_id = min(object_id, NO_TRACK_ID - 1)

            detections.append(
                Detection(
                    box=(u1, v1, u2, v2),
                    label=label,
                    score=float(score),
                    track_id=object_id,
                    source=self.source_name,
                )
            )

        return detections


class DualObjectDetect(Node):
    def __init__(self):
        super().__init__("dual_object_detect")

        self.declare_parameter("enable_bbox_image", False)
        self.declare_parameter("sub_image", "image")
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 384)
        self.declare_parameter("pub_bbox_image", "bbox_image")
        self.declare_parameter("pub_objects", "objects")
        self.declare_parameter("use_cuda", False)
        self.declare_parameter("max_age", 30)
        self.declare_parameter("primary_model_path", "yolov8n_640x384.engine")
        self.declare_parameter("secondary_model_path", "")
        self.declare_parameter("primary_only_objects", "")
        self.declare_parameter("primary_skip_objects", "")
        self.declare_parameter("secondary_only_objects", "")
        self.declare_parameter("secondary_skip_objects", "")
        self.declare_parameter("secondary_id_offset", 30000)

        self.enable_bbox_image = (
            self.get_parameter("enable_bbox_image").get_parameter_value().bool_value
        )
        self.sub_image = self.get_parameter("sub_image").get_parameter_value().string_value
        self.sub_camera_info = replace_base(self.sub_image, "camera_info")
        self.image_width = (
            self.get_parameter("image_width").get_parameter_value().integer_value
        )
        self.image_height = (
            self.get_parameter("image_height").get_parameter_value().integer_value
        )
        self.pub_bbox_image_topic = (
            self.get_parameter("pub_bbox_image").get_parameter_value().string_value
        )
        self.pub_camera_info_topic = replace_base(self.pub_bbox_image_topic, "camera_info")
        self.pub_objects_topic = (
            self.get_parameter("pub_objects").get_parameter_value().string_value
        )
        self.use_cuda = self.get_parameter("use_cuda").get_parameter_value().bool_value
        self.primary_model_path = (
            self.get_parameter("primary_model_path").get_parameter_value().string_value
        )
        self.secondary_model_path = (
            self.get_parameter("secondary_model_path").get_parameter_value().string_value
        )
        self.primary_only_objects = (
            self.get_parameter("primary_only_objects").get_parameter_value().string_value.split()
        )
        self.primary_skip_objects = (
            self.get_parameter("primary_skip_objects").get_parameter_value().string_value.split()
        )
        self.secondary_only_objects = (
            self.get_parameter("secondary_only_objects").get_parameter_value().string_value.split()
        )
        self.secondary_skip_objects = (
            self.get_parameter("secondary_skip_objects").get_parameter_value().string_value.split()
        )
        self.secondary_id_offset = (
            self.get_parameter("secondary_id_offset").get_parameter_value().integer_value
        )

        if not self.secondary_model_path:
            raise RuntimeError("secondary_model_path must be set for dual_object_detect")

        self.primary_runner = ModelRunner(
            node=self,
            source_name="primary",
            model_path=self.primary_model_path,
            only_objects=self.primary_only_objects,
            skip_objects=self.primary_skip_objects,
            image_width=self.image_width,
            image_height=self.image_height,
            use_cuda=self.use_cuda,
            id_offset=0,
        )
        self.secondary_runner = ModelRunner(
            node=self,
            source_name="secondary",
            model_path=self.secondary_model_path,
            only_objects=self.secondary_only_objects,
            skip_objects=self.secondary_skip_objects,
            image_width=self.image_width,
            image_height=self.image_height,
            use_cuda=self.use_cuda,
            id_offset=self.secondary_id_offset,
        )

        self.pub_objects = self.create_publisher(
            StampedObjectArray, self.pub_objects_topic, 10
        )
        self.get_logger().info(f"Publishing objects to: {self.pub_objects_topic}")

        if self.enable_bbox_image:
            self.pub_bbox_image = self.create_publisher(
                Image, self.pub_bbox_image_topic, 1
            )
            self.pub_camera_info = self.create_publisher(
                CameraInfo, self.pub_camera_info_topic, 1
            )
            self.sub_camera_info = self.create_subscription(
                CameraInfo, self.sub_camera_info, self.pub_camera_info.publish, 1
            )
            self.get_logger().info(f"Publishing bbox image to: {self.pub_bbox_image_topic}")

        self.sub_image_handle = self.create_subscription(
            Image, self.sub_image, self.callback, 1
        )
        self.get_logger().info(f"Subscribing to image topic: {self.sub_image}")

    def run(self):
        rclpy.spin(self)

    def prepare_frame(self, image):
        frame = np.frombuffer(image.data, dtype=np.uint8).reshape(
            image.height, image.width, -1
        )
        if image.width != self.image_width or image.height != self.image_height:
            frame = cv2.resize(frame, (self.image_width, self.image_height))
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        if frame.shape[2] == 3:
            if image.encoding == "rgb8":
                return frame
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[2] == 1:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        raise RuntimeError(f"Unsupported image shape for detection: {frame.shape}")

    def create_object_messages(self, detections):
        objects = []
        for detection in detections:
            u1, v1, u2, v2 = detection.box

            obj = Object()
            obj.id = detection.track_id
            obj.label = detection.label
            obj.detection_conf = detection.score
            obj.tracking_conf = detection.score
            obj.image_width = self.image_width
            obj.image_height = self.image_height
            obj.roi.x_offset = u1
            obj.roi.y_offset = v1
            obj.roi.width = u2 - u1
            obj.roi.height = v2 - v1
            objects.append(obj)

        return objects

    def annotate_frame(self, frame_rgb, detections, color):
        for detection in detections:
            u1, v1, u2, v2 = detection.box
            cv2.rectangle(frame_rgb, (u1, v1), (u2, v2), color, 2)
            caption = f"{detection.label} {detection.score:.2f}"
            cv2.putText(
                frame_rgb,
                caption,
                (u1, max(v1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return frame_rgb

    def publish_bbox_image(self, source_image, frame_rgb):
        new_image = Image()
        new_image.header = source_image.header
        new_image.height = frame_rgb.shape[0]
        new_image.width = frame_rgb.shape[1]
        new_image.encoding = "rgb8"
        new_image.step = frame_rgb.size // new_image.height
        new_image.data = frame_rgb.tobytes()
        self.pub_bbox_image.publish(new_image)

    def callback(self, image):
        frame_rgb = self.prepare_frame(image)

        primary_result = self.primary_runner.run(frame_rgb)
        secondary_result = self.secondary_runner.run(frame_rgb)

        primary_detections = self.primary_runner.detections_from_result(primary_result)
        secondary_detections = self.secondary_runner.detections_from_result(secondary_result)
        detections = primary_detections + secondary_detections

        if self.enable_bbox_image:
            annotated = frame_rgb.copy()
            annotated = self.annotate_frame(annotated, primary_detections, (0, 255, 0))
            annotated = self.annotate_frame(annotated, secondary_detections, (255, 140, 0))
            self.publish_bbox_image(image, annotated)

        objects = self.create_object_messages(detections)
        if objects:
            msg = StampedObjectArray()
            msg.header = image.header
            msg.objects = objects
            self.pub_objects.publish(msg)


def main():
    rclpy.init()
    node = None

    try:
        node = DualObjectDetect()
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
