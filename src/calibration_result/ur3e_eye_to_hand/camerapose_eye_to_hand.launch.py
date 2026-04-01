""" Static transform publisher acquired via MoveIt 2 hand-eye calibration """
""" EYE-TO-HAND: base_link -> camera_color_optical_frame """
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    nodes = [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            output="log",
            arguments=[
                "--frame-id",
                "base_link",
                "--child-frame-id",
                "camera_color_optical_frame",
                "--x",
                "-0.337647",
                "--y",
                "-0.240584",
                "--z",
                "0.56153",
                "--qx",
                "-0.705976",
                "--qy",
                "-0.0665758",
                "--qz",
                "0.0387974",
                "--qw",
                "0.704031",
                # "--roll",
                # "1.56506",
                # "--pitch",
                # "-2.99252",
                # "--yaw",
                # "3.10177",
            ],
        ),
    ]
    return LaunchDescription(nodes)
