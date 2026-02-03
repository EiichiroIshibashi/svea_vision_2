#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments - Options
        DeclareLaunchArgument('camera_name', default_value='zed', description='Camera name'),
        DeclareLaunchArgument('camera_model', default_value='zed', description='Camera model'),
        DeclareLaunchArgument('use_cuda', default_value='true', description='Use CUDA'),
        DeclareLaunchArgument('enable_bbox_image', default_value='false', description='Enable bbox image'),
        DeclareLaunchArgument('enable_aruco', default_value='false', description='Enable ArUco detection'),
        DeclareLaunchArgument('enable_state_estimation', default_value='true', description='Enable state estimation'),
        DeclareLaunchArgument('frame_id', default_value='map', description='Frame ID'),
        DeclareLaunchArgument('enable_sidewalk_segmentation', default_value='false', description='Enable sidewalk segmentation'),
        DeclareLaunchArgument('sidewalk_frame_id', default_value='', description='Sidewalk frame ID'),
        DeclareLaunchArgument('sidewalk_prompt_type', default_value='bbox', description='Sidewalk prompt type'),
        DeclareLaunchArgument('sidewalk_prompt_text', default_value='a sidewalk or footpath or walkway or paved path', description='Sidewalk prompt text'),
        DeclareLaunchArgument('sidewalk_owl_roi', default_value='[0.25, 0.50, 0.75, 0.95]', description='Sidewalk OWL ROI'),
        DeclareLaunchArgument('verbose', default_value='false', description='Verbose output'),

        # Camera pose parameters
        DeclareLaunchArgument('zed_base_frame', default_value='base_link', description='ZED base frame'),
        DeclareLaunchArgument('zed_cam_pos_x', default_value='0.0', description='ZED camera X position'),
        DeclareLaunchArgument('zed_cam_pos_y', default_value='0.0', description='ZED camera Y position'),
        DeclareLaunchArgument('zed_cam_pos_z', default_value='0.0', description='ZED camera Z position'),
        DeclareLaunchArgument('zed_cam_roll', default_value='0.0', description='ZED camera roll'),
        DeclareLaunchArgument('zed_cam_pitch', default_value='0.0', description='ZED camera pitch'),
        DeclareLaunchArgument('zed_cam_yaw', default_value='0.0', description='ZED camera yaw'),

        # Auxiliary parameters
        DeclareLaunchArgument('only_objects', default_value='', description='Only detect specified objects'),
        DeclareLaunchArgument('skip_objects', default_value='', description='Skip specified objects'),
        DeclareLaunchArgument('max_age', default_value='30', description='Maximum tracking age'),
        DeclareLaunchArgument('aruco_size', default_value='0.1', description='ArUco marker size'),

        # Pedestrian state estimation constants
        DeclareLaunchArgument('max_time_missing', default_value='1.0', description='Max time missing for pedestrian tracking'),
        DeclareLaunchArgument('vel_filter_window', default_value='15', description='Velocity filter window'),
        DeclareLaunchArgument('discard_id_threshold', default_value='0.5', description='Discard ID threshold'),

        # Constants (topics)
        DeclareLaunchArgument('depth_image', default_value='/zed/zed_node/depth/depth_registered', description='Depth image topic'),
        DeclareLaunchArgument('image', default_value='/zed/zed_node/rgb/color/rect/image', description='RGB image topic'),
        DeclareLaunchArgument('point_cloud', default_value='/zed/zed_node/point_cloud/cloud_registered', description='Point cloud topic'),
        DeclareLaunchArgument('map_path', default_value='', description='Map file path'),

        # Conditional map server (commented out - needs map_server package)
        # Node(
        #     condition=IfCondition(LaunchConfiguration('map_path')),
        #     package='nav2_map_server',
        #     executable='map_server',
        #     name='map_server',
        #     parameters=[{'yaml_filename': LaunchConfiguration('map_path')}]
        # ),

        # ZED Camera wrapper
        GroupAction([
            IncludeLaunchDescription(
                PathJoinSubstitution([
                    FindPackageShare('zed_wrapper'),
                    'launch',
                    'zed_camera.launch.py'
                ]),
                launch_arguments={
                    'camera_name': LaunchConfiguration('camera_name'),
                    'camera_model': LaunchConfiguration('camera_model'),
                    # Note: zed_camera.launch.py doesn't support base_frame and cam_pos parameters
                    # These would need to be handled differently if required
                }.items()
            )
        ], scoped=True),

        # Include object_pose launch
        IncludeLaunchDescription(
            PathJoinSubstitution([
                FindPackageShare('svea_vision'),
                'launch',
                'object_pose.launch.py'
            ]),
            launch_arguments={
                'use_cuda': LaunchConfiguration('use_cuda'),
                'enable_bbox_image': LaunchConfiguration('enable_bbox_image'),
                'frame_id': LaunchConfiguration('frame_id'),
                'image': LaunchConfiguration('image'),
                'depth_image': LaunchConfiguration('depth_image'),
                'only_objects': LaunchConfiguration('only_objects'),
                'skip_objects': LaunchConfiguration('skip_objects'),
                'max_age': LaunchConfiguration('max_age'),
            }.items()
        ),

        # Include aruco_detect launch (conditional)
        IncludeLaunchDescription(
            PathJoinSubstitution([
                FindPackageShare('svea_vision'),
                'launch',
                'aruco_detect.launch.py'
            ]),
            launch_arguments={
                'image': LaunchConfiguration('image'),
                'aruco_size': LaunchConfiguration('aruco_size'),
            }.items(),
            condition=IfCondition(LaunchConfiguration('enable_aruco'))
        ),

        # State estimation nodes (conditional group)
        GroupAction([
            Node(
                package='svea_vision',
                executable='detection_splitter',
                name='detection_splitter',
                output='screen'
            ),
            Node(
                package='svea_vision',
                executable='person_state_estimation',
                name='person_state_estimation',
                output='screen'
            ),
            Node(
                package='svea_vision',
                executable='pedestrian_flow_estimate',
                name='pedestrian_flow_estimate',
                output='screen',
                parameters=[{
                    'persons_topic': '/detection_splitter/persons',
                    'max_time_missing': LaunchConfiguration('max_time_missing'),
                    'vel_filter_window': LaunchConfiguration('vel_filter_window'),
                    'discard_id_threshold': LaunchConfiguration('discard_id_threshold'),
                }]
            )
        ], condition=IfCondition(LaunchConfiguration('enable_state_estimation'))),

        # Include sidewalk_segmentation launch (conditional)
        IncludeLaunchDescription(
            PathJoinSubstitution([
                FindPackageShare('svea_vision'),
                'launch',
                'sidewalk_segmentation.launch.py'
            ]),
            launch_arguments={
                'use_cuda': LaunchConfiguration('use_cuda'),
                'frame_id': LaunchConfiguration('sidewalk_frame_id'),
                'prompt_type': LaunchConfiguration('sidewalk_prompt_type'),
                'prompt_text': LaunchConfiguration('sidewalk_prompt_text'),
                'owl_roi': LaunchConfiguration('sidewalk_owl_roi'),
                'verbose': LaunchConfiguration('verbose'),
                'rgb_topic': LaunchConfiguration('image'),
                'pointcloud_topic': LaunchConfiguration('point_cloud'),
                'sidewalk_mask_topic': 'sidewalk_mask',
                'sidewalk_image_topic': 'sidewalk_image',
                'sidewalk_pointcloud_topic': 'sidewalk_pointcloud',
            }.items(),
            condition=IfCondition(LaunchConfiguration('enable_sidewalk_segmentation'))
        )
    ])
