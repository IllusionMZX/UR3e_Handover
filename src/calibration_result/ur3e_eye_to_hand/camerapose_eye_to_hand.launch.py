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
                "-0.207681",
                "--y",
                "0.611696",
                "--z",
                "0.686143",
                "--qx",
                "-0.381286",
                "--qy",
                "0.829368",
                "--qz",
                "-0.396254",
                "--qw",
                "0.0987551",
                # "--roll",
                # "2.42377",
                # "--pitch",
                # "0.484742",
                # "--yaw",
                # "2.46473",
            ],
        ),
    ]
    return LaunchDescription(nodes)
