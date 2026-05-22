""" Static transform publisher acquired via MoveIt 2 hand-eye calibration """
""" EYE-IN-HAND: tool0 -> zed_left_camera_frame_optical """
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
                "zed_inhand_camera_frame_optical",
                "--x",
                "-0.0662321",
                "--y",
                "-0.043534",
                "--z",
                "0.0530993",
                "--qx",
                "0.0242133",
                "--qy",
                "-0.00928276",
                "--qz",
                "-0.021766",
                "--qw",
                "0.999427",
                # "--roll",
                # "0.0480224",
                # "--pitch",
                # "-0.0196102",
                # "--yaw",
                # "-0.0430792",
            ],
        ),
    ]
    return LaunchDescription(nodes)
