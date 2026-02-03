#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

import cv2
import numpy as np

from ultralytics import YOLO

from svea_vision_msgs.msg import Object, StampedObjectArray


def replace_base(old, new):
    split_last = lambda xs: (xs[:-1], xs[-1])
    is_private = new.startswith("~")
    is_global = new.startswith("/")
    assert not (is_private or is_global)
    ns, _ = split_last(old.split("/"))
    ns += new.split("/")
    return "/".join(ns)


def iou(box1, box2):
    u11, v11, u21, v21 = box1
    u12, v12, u22, v22 = box2

    u1_max = max(u11, u12)
    v1_max = max(v11, v12)
    u2_min = min(u21, u22)
    v2_min = min(v21, v22)

    # area of intersection
    w = abs(u1_max - u2_min)
    h = abs(v1_max - v2_min)
    area_i = w * h

    # area of box1
    w = abs(u21 - u11)
    h = abs(v21 - v11)
    area_1 = w * h

    # area of box2
    w = abs(u22 - u12)
    h = abs(v22 - v12)
    area_2 = w * h

    # area of union
    area_u = area_1 + area_2 - area_i

    # intersection-over-union
    return area_i / area_u


class ObjectDetect(Node):
    def __init__(self):
        super().__init__("object_detect")

        ## Parameters
        self.declare_parameter("enable_bbox_image", False)
        self.declare_parameter("sub_image", "image")
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("pub_bbox_image", "bbox_image")
        self.declare_parameter("use_cuda", False)
        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("only_objects", "")
        self.declare_parameter("skip_objects", "")
        self.declare_parameter("max_age", 30)
        self.declare_parameter("pub_objects", "objects")

        self.ENABLE_BBOX_IMAGE = self.get_parameter("enable_bbox_image").get_parameter_value().bool_value

        self.SUB_IMAGE = self.get_parameter("sub_image").get_parameter_value().string_value
        self.SUB_CAMERA_INFO = replace_base(self.SUB_IMAGE, "camera_info")

        self.IMAGE_WIDTH = self.get_parameter("image_width").get_parameter_value().integer_value
        self.IMAGE_HEIGHT = self.get_parameter("image_height").get_parameter_value().integer_value

        self.PUB_BBOX_IMAGE = self.get_parameter("pub_bbox_image").get_parameter_value().string_value
        self.PUB_CAMERA_INFO = replace_base(self.PUB_BBOX_IMAGE, "camera_info")

        self.USE_CUDA = self.get_parameter("use_cuda").get_parameter_value().bool_value
        self.MODEL_PATH = self.get_parameter("model_path").get_parameter_value().string_value

        # Space separated list, e.g. 'bed cup dog'
        self.ONLY_OBJECTS = self.get_parameter("only_objects").get_parameter_value().string_value.split()
        self.SKIP_OBJECTS = self.get_parameter("skip_objects").get_parameter_value().string_value.split()

        self.MAX_AGE = self.get_parameter("max_age").get_parameter_value().integer_value

        self.PUB_OBJECTS = self.get_parameter("pub_objects").get_parameter_value().string_value

        ## Neural Network

        # Load YOLO model with proper error handling for PyTorch 2.x compatibility
        import torch
        
        try:
            # First, try standard loading
            self.model = YOLO(self.MODEL_PATH)
            self.get_logger().info(f"Successfully loaded YOLO model from {self.MODEL_PATH}")
        except Exception as first_error:
            self.get_logger().warn(f"Standard loading failed: {first_error}")
            
            try:
                # Try with weights_only=False for legacy models
                original_load = torch.load
                def safe_load(*args, **kwargs):
                    kwargs.setdefault('weights_only', False)
                    return original_load(*args, **kwargs)
                
                torch.load = safe_load
                self.model = YOLO(self.MODEL_PATH)
                torch.load = original_load
                self.get_logger().info(f"Successfully loaded YOLO model with weights_only=False")
                
            except Exception as second_error:
                self.get_logger().error(f"Failed to load YOLO model: {second_error}")
                raise RuntimeError(f"Could not load YOLO model from {self.MODEL_PATH}")

        if self.USE_CUDA:
            try:
                self.model.to("cuda")
                self.get_logger().info("CUDA enabled")
            except Exception as cuda_error:
                self.get_logger().warn(f"CUDA initialization failed: {cuda_error}, falling back to CPU")
                self.USE_CUDA = False
        else:
            self.get_logger().info("CUDA disabled")

        # Get model class names safely
        try:
            model_names = self.model.names
            if isinstance(model_names, dict):
                classes = {lbl: cls for cls, lbl in model_names.items()}
            elif isinstance(model_names, list):
                # Handle case where names is a list
                classes = {name: i for i, name in enumerate(model_names)}
            else:
                # Fallback for unknown structure
                classes = {}
            self.label_to_class = lambda label: classes.get(label, 0)
        except Exception as e:
            self.get_logger().warn(f"Could not load class names: {e}")
            self.label_to_class = lambda label: 0

        self.detect_kwargs = {"persist": True, "conf": 0.5, "verbose": False}

        if self.ONLY_OBJECTS:
            try:
                class_ids = [self.label_to_class(obj) for obj in self.ONLY_OBJECTS]
                self.detect_kwargs["classes"] = class_ids
            except Exception as e:
                self.get_logger().warn(f"Could not set class filter: {e}")

        ## Publishers

        self.pub_objects = self.create_publisher(
            StampedObjectArray, self.PUB_OBJECTS, 10
        )
        self.get_logger().info(f"Publishing to: {self.PUB_OBJECTS}")

        if self.ENABLE_BBOX_IMAGE:
            self.pub_bbox_image = self.create_publisher(
                Image, self.PUB_BBOX_IMAGE, 1
            )
            self.get_logger().info(f"Publishing to: {self.PUB_BBOX_IMAGE}")

        ## Subscribers

        self.sub_image = self.create_subscription(Image, self.SUB_IMAGE, self.callback, 1)
        self.get_logger().info(f"Subscribing to: {self.SUB_IMAGE}")

        ## Relay (sub->pub) camera info

        if self.ENABLE_BBOX_IMAGE:
            self.pub_camera_info = self.create_publisher(CameraInfo, self.PUB_CAMERA_INFO, 1)
            self.sub_camera_info = self.create_subscription(CameraInfo, self.SUB_CAMERA_INFO, self.pub_camera_info.publish, 1)

    def run(self):
        rclpy.spin(self)

    def callback(self, image):
        ## Detect objects

        frame = np.frombuffer(image.data, dtype=np.uint8).reshape(
            image.height, image.width, -1
        )
        frame = cv2.resize(frame, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        result = self.model.track(frame, **self.detect_kwargs)[0]

        # if enabled, modify frame (add bounding boxes)
        if self.ENABLE_BBOX_IMAGE:
            frame = result.plot()

        if self.ENABLE_BBOX_IMAGE:
            new_image = Image()
            new_image.header = image.header
            new_image.height = frame.shape[0]
            new_image.width = frame.shape[1]
            new_image.encoding = "rgb8"
            new_image.step = frame.size // new_image.height
            new_image.data = frame.tobytes()

            self.pub_bbox_image.publish(new_image)

        if len(result.boxes) == 0 or not result.boxes.is_track:
            return

        result = result.cpu().numpy()
        it = zip(result.boxes.xyxy,
                result.boxes.conf,
                result.boxes.cls,
                result.boxes.id)

        ## Create object messages

        objects = []
        for box, _pred, _cls, _id in it:
            u1, v1, u2, v2 = box

            # get the label name
            label = result.names[_cls]

            if self.SKIP_OBJECTS and label in self.SKIP_OBJECTS:
                continue

            if self.ONLY_OBJECTS and label not in self.ONLY_OBJECTS:
                continue

            # get real pixel coords
            u1, u2 = [round(u) for u in (u1, u2)]
            v1, v2 = [round(v) for v in (v1, v2)]

            obj = Object()
            # obj.id = int(_id)
            obj.id = int(_id) if _id is not None else -1
            obj.label = label
            obj.detection_conf = float(_pred)
            obj.tracking_conf = float(_pred)
            obj.image_width = self.IMAGE_WIDTH
            obj.image_height = self.IMAGE_HEIGHT
            obj.roi.x_offset = u1
            obj.roi.y_offset = v1
            obj.roi.width = u2 - u1
            obj.roi.height = v2 - v1
            objects.append(obj)

        # Publish objects
        if objects:
            object_array = StampedObjectArray()
            object_array.header = image.header
            object_array.objects = objects
            self.pub_objects.publish(object_array)


def main():
    ##  Start node  ##

    rclpy.init()
    node = ObjectDetect()
    
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
