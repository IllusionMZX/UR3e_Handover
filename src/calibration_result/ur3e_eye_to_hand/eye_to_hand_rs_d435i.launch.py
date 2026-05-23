""" Static transform publisher acquired via MoveIt 2 hand-eye calibration """
""" EYE-TO-HAND: base_link -> rs_tohand_color_optical_frame """
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
                "rs_tohand_color_optical_frame",
                "--x",
                "0.802489",
                "--y",
                "0.517322",
                "--z",
                "1.06262",
                "--qx",
                "0.599875",
                "--qy",
                "0.749088",
                "--qz",
                "-0.225732",
                "--qw",
                "-0.167516",
                # "--roll",
                # "2.98005",
                # "--pitch",
                # "-0.548948",
                # "--yaw",
                # "-1.83671",
            ],
        ),
    ]
    return LaunchDescription(nodes)
