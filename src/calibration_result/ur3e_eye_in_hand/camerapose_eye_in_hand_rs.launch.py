""" Static transform publisher acquired via MoveIt 2 hand-eye calibration """
""" EYE-IN-HAND: tool0 -> camera_inhand_color_optical_frame """
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
                "tool0",
                "--child-frame-id",
                "camera_inhand_color_optical_frame",
                "--x",
                "-0.0331484",
                "--y",
                "-0.0991703",
                "--z",
                "0.00759719",
                "--qx",
                "0.00427845",
                "--qy",
                "-0.000158831",
                "--qz",
                "0.0200313",
                "--qw",
                "0.99979",
                # "--roll",
                # "0.00856158",
                # "--pitch",
                # "-0.000146189",
                # "--yaw",
                # "0.0400662",
            ],
        ),
    ]
    return LaunchDescription(nodes)
