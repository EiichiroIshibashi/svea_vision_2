#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_cuda", default_value="true"),
            DeclareLaunchArgument("enable_bbox_image", default_value="false"),
            DeclareLaunchArgument("image", default_value="image"),
            DeclareLaunchArgument("objects", default_value="objects"),
            DeclareLaunchArgument("bbox_image", default_value="bbox_image"),
            DeclareLaunchArgument("image_width", default_value="640"),
            DeclareLaunchArgument("image_height", default_value="384"),
            DeclareLaunchArgument("max_age", default_value="30"),
            DeclareLaunchArgument(
                "primary_model_path",
                default_value="/home/user/zed_x_one_demo/yolov8n_640x384.engine",
            ),
            DeclareLaunchArgument(
                "secondary_model_path",
                default_value="/home/user/zed_x_one_demo/best.pt",
            ),
            DeclareLaunchArgument("primary_only_objects", default_value=""),
            DeclareLaunchArgument("primary_skip_objects", default_value=""),
            DeclareLaunchArgument("secondary_only_objects", default_value=""),
            DeclareLaunchArgument("secondary_skip_objects", default_value=""),
            DeclareLaunchArgument("secondary_id_offset", default_value="30000"),
            Node(
                package="svea_vision_experimental",
                executable="dual_object_detect",
                name="dual_object_detect",
                output="screen",
                parameters=[
                    {
                        "use_cuda": LaunchConfiguration("use_cuda"),
                        "enable_bbox_image": LaunchConfiguration("enable_bbox_image"),
                        "sub_image": LaunchConfiguration("image"),
                        "pub_objects": LaunchConfiguration("objects"),
                        "pub_bbox_image": LaunchConfiguration("bbox_image"),
                        "image_width": LaunchConfiguration("image_width"),
                        "image_height": LaunchConfiguration("image_height"),
                        "max_age": LaunchConfiguration("max_age"),
                        "primary_model_path": LaunchConfiguration("primary_model_path"),
                        "secondary_model_path": LaunchConfiguration("secondary_model_path"),
                        "primary_only_objects": LaunchConfiguration("primary_only_objects"),
                        "primary_skip_objects": LaunchConfiguration("primary_skip_objects"),
                        "secondary_only_objects": LaunchConfiguration("secondary_only_objects"),
                        "secondary_skip_objects": LaunchConfiguration("secondary_skip_objects"),
                        "secondary_id_offset": LaunchConfiguration("secondary_id_offset"),
                    }
                ],
            ),
        ]
    )
