import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import open3d as o3d
import os
import sys
import argparse
import asyncio
import datetime
import scipy.io as scio
import time
import logging
import base64
import re
import json
import threading
import tempfile
import subprocess
import shutil
from pathlib import Path

# ====== Volcengine SDK ======
from volcenginesdkarkruntime import Ark, AsyncArk

# ====== ZED Camera ======
import pyzed.sl as sl

# ====== ROS2 & RTDE Imports ======
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import tf2_ros
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation as R

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from robotiq_2f_urcap_adapter.action import GripperCommand
# =================================
import mediapipe as mp

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ===== GPU config (SAM2 requires bfloat16) =====
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# === 路径设置 ===
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.dirname(ROOT_DIR)
SRC_DIR = os.path.dirname(BASELINE_DIR)
sys.path.append(os.path.join(BASELINE_DIR, 'models'))
sys.path.append(os.path.join(BASELINE_DIR, 'dataset'))
sys.path.append(os.path.join(BASELINE_DIR, 'utils'))

# SAM2 path
SAM2_DIR = os.path.join(SRC_DIR, "SAM2_streaming")
sys.path.insert(0, SAM2_DIR)
from sam2.build_sam import build_sam2_camera_predictor

# FFS path
FFS_DIR = os.path.join(SRC_DIR, "Fast-FoundationStereo")
sys.path.append(FFS_DIR)
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, vis_disparity

# FFS Constants
MODEL_DIR = os.path.join(FFS_DIR, "weights/23-36-37/model_best_bp2_serialize.pth")
VALID_ITERS = 8
MAX_DISP = 192
ZFAR = 5.0
ZNEAR = 0.2

from graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup
from data_utils import CameraInfo, create_point_cloud_from_depth_image

# ===== Volcengine Config (大模型) =====
ARK_API_KEY = os.getenv('ARK_API_KEY')
client = Ark(
    base_url='https://ark.cn-beijing.volces.com/api/v3',
    api_key=ARK_API_KEY,
)
async_client = AsyncArk(
    base_url='https://ark.cn-beijing.volces.com/api/v3',
    api_key=ARK_API_KEY,
)
MODEL_NAME = "doubao-seed-2-0-mini-260428"
REASONING_EFFORT = "minimal"
AUDIO_FORMAT_MAP = {
    ".wav": "wav",
    ".mp3": "mp3",
    ".m4a": "m4a",
    ".flac": "flac",
    ".ogg": "ogg",
}
VOICE_MAX_RECORD_SECONDS = 10.0

# ===== MediaPipe Config ======
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
PHYSICAL_RIGHT_HAND_MP_LABEL = "Left"
HANDOVER_DEFAULT_BASE_YAW = math.radians(100)
HANDOVER_JOINT_SHAPE = [
    HANDOVER_DEFAULT_BASE_YAW,
    math.radians(45),
    math.radians(-95),
    math.radians(-130),
    math.radians(-90),
    math.radians(0),
]
HANDOVER_RIGHT_HAND_TIMEOUT = 5.0
HANDOVER_RIGHT_HAND_MAX_AGE = 3.0
FOLLOW_RIGHT_HAND_DISTANCE = 0.15
FOLLOW_RIGHT_HAND_Z_OFFSET = 0.00
FOLLOW_MOVE_SPEED = 0.05
FOLLOW_MOVE_ACCEL = 0.05
FOLLOW_MAX_XY_STEP = 0.03
FOLLOW_MAX_Z_STEP = 0.02
WORKSPACE_OBSERVE_BASE_OFFSET = 0.10
WORKSPACE_OBSERVE_Z_OFFSET_HIGH = 0.30
WORKSPACE_OBSERVE_RETRY_STEP = 0.10
WORKSPACE_OBSERVE_SLOW_MOVE_SPEED = 0.05
WORKSPACE_OBSERVE_SLOW_MOVE_ACCEL = 0.05
SAM2_SETTLE_SECONDS = 1.0
DEBUG_MODE = False

# ===== ZED Downscale 参数 =====
SCALE_FACTOR = 0.5
IMG_WIDTH = int(1280 * SCALE_FACTOR) # 640
IMG_HEIGHT = int(720 * SCALE_FACTOR) # 360
MASK_ALPHA = 0.5
MASK_COLORS_BGR = {1: [75, 70, 203], 2: [203, 192, 75]}
DENOISE_VOXEL_SIZE = 0.001
DENOISE_NB_POINTS = 30
DENOISE_RADIUS = 0.03
CAMERA_GRASP_MAX_TILT_DEG = 30.0
CAMERA_GRASP_MAX_CLOSING_FORWARD_COS = 0.75
HIGH_SCORE_POOL_TOP_K = 120
HIGH_SCORE_MIN_RATIO = 0.55
BACKOFF_DIST = 0.08
Z_OFFSET_DOWN = 0.005
BASE_TO_CONTROLLER_ROT = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)

# === 全局变量 ===
vlm_running = False
doubao_trigger = False
workspace_bbox = None
pending_targets = []
request_selection_prompt = False
global ROS_NODE
ROS_NODE = None
global STATE
STATE = "IDLE"
OPEN3D_VIEWER = None


class Open3DLiveViewer:
    def __init__(self, window_name="Open3D Grasp Viewer"):
        self.window_name = window_name
        self.vis = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.pending_update = None
        self.cloud = o3d.geometry.PointCloud()
        self.cloud_added = False
        self.gripper_geometries = []
        self.view_initialized = False
        self.render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self.render_thread.start()

    def _render_loop(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=self.window_name, width=1280, height=720, visible=True)
        self.ready_event.set()

        while not self.stop_event.is_set():
            with self.lock:
                update_payload = self.pending_update
                self.pending_update = None

            if update_payload is not None:
                points, colors, gg_array, top_k = update_payload
                self.cloud.points = o3d.utility.Vector3dVector(points)
                self.cloud.colors = o3d.utility.Vector3dVector(colors)

                if not self.cloud_added:
                    self.vis.add_geometry(self.cloud)
                    self.cloud_added = True
                else:
                    self.vis.update_geometry(self.cloud)

                for geom in self.gripper_geometries:
                    self.vis.remove_geometry(geom, reset_bounding_box=False)
                self.gripper_geometries = []

                gg = GraspGroup(gg_array).nms().sort_by_score()
                if len(gg) > 0:
                    show_n = min(len(gg), max(int(top_k), 1))
                    self.gripper_geometries = gg[:show_n].to_open3d_geometry_list()
                    for geom in self.gripper_geometries:
                        self.vis.add_geometry(geom, reset_bounding_box=False)

                if not self.view_initialized:
                    ctr = self.vis.get_view_control()
                    ctr.set_front([0, 0, -1])
                    ctr.set_lookat([0, 0, 0.5])
                    ctr.set_up([0, -1, 0])
                    ctr.set_zoom(0.8)
                    self.view_initialized = True

            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.02)

        self.vis.destroy_window()
        self.vis = None

    def update(self, points, colors, gg_array, top_k=1):
        self.ready_event.wait()
        with self.lock:
            self.pending_update = (
                np.asarray(points, dtype=np.float64),
                np.asarray(colors, dtype=np.float64),
                np.asarray(gg_array),
                top_k,
            )

    def close(self):
        self.stop_event.set()
        if self.render_thread.is_alive():
            self.render_thread.join(timeout=1.0)
        with self.lock:
            self.pending_update = None
            self.cloud_added = False
            self.gripper_geometries = []
            self.view_initialized = False


def get_open3d_viewer():
    global OPEN3D_VIEWER
    if OPEN3D_VIEWER is None:
        OPEN3D_VIEWER = Open3DLiveViewer()
    return OPEN3D_VIEWER


def close_open3d_viewer():
    global OPEN3D_VIEWER
    if OPEN3D_VIEWER is not None:
        OPEN3D_VIEWER.close()
        OPEN3D_VIEWER = None

# --- ROS2 机器人控制节点 ---
class RobotControllerNode(Node):
    def __init__(self):
        super().__init__('robot_controller_node')
        self.callback_group = ReentrantCallbackGroup()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._action_client = ActionClient(self, GripperCommand, '/robotiq_2f_urcap_adapter/gripper_command', callback_group=self.callback_group)
        self.ur_ip = "192.168.1.10"
        
        # 使用 Socket 提前检测机器人的连通性，防止 ur_rtde 底层因无法连接而引发内部崩溃或退出
        import socket
        robot_reachable = False
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            if sock.connect_ex((self.ur_ip, 30004)) == 0:
                robot_reachable = True
            sock.close()
        except:
            pass

        self.rtde_c = None
        self.rtde_r = None
        self.dynamic_start_pose = None
        self.right_hand_base_pos = None
        self.right_hand_seen_time = 0.0
        self.right_hand_lock = threading.Lock()
        self.handover_follow_pose = None
        self.handover_follow_z_offset = None
        if robot_reachable:
            try:
                self.rtde_c = RTDEControlInterface(self.ur_ip)
                self.rtde_r = RTDEReceiveInterface(self.ur_ip)
                self.get_logger().info("✓ Connected to UR3e robot arm.")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize RTDE: {e}")
                self.rtde_c = None
        else:
            self.get_logger().warning(f"UR3e robot arm at {self.ur_ip} is unreachable! Running in VISION-ONLY mode.")

    def send_gripper_command(self, position, speed=0.15, force=140.0):
        if not self._action_client.wait_for_server(timeout_sec=5.0): return False
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = force
        goal_msg.command.max_speed = speed
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        while not send_goal_future.done(): time.sleep(0.1)
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted: return False
        get_result_future = goal_handle.get_result_async()
        
        timeout_start = time.time()
        while not get_result_future.done(): 
            if time.time() - timeout_start > 5.0:
                self.get_logger().warning("Gripper action timeout.")
                break
            time.sleep(0.1)
            
        if get_result_future.done():
            return get_result_future.result().result.reached_goal
        return True

    def update_right_hand_base_pos(self, hand_base_pos):
        with self.right_hand_lock:
            self.right_hand_base_pos = np.array(hand_base_pos, dtype=float)
            self.right_hand_seen_time = time.time()

    def get_recent_right_hand_base_pos(self, max_age=HANDOVER_RIGHT_HAND_MAX_AGE):
        with self.right_hand_lock:
            if self.right_hand_base_pos is None:
                return None
            if time.time() - self.right_hand_seen_time > max_age:
                return None
            return self.right_hand_base_pos.copy()

    def wait_for_right_hand_base_pos(self, timeout=HANDOVER_RIGHT_HAND_TIMEOUT):
        start_time = time.time()
        last_log_time = 0.0
        while True:
            hand_base_pos = self.get_recent_right_hand_base_pos(max_age=HANDOVER_RIGHT_HAND_MAX_AGE)
            if hand_base_pos is not None:
                return hand_base_pos
            if timeout is not None and timeout > 0 and (time.time() - start_time) >= timeout:
                return None
            now = time.time()
            if now - last_log_time > 1.5:
                if timeout is None or timeout <= 0:
                    self.get_logger().info("Waiting for physical right hand to appear...")
                last_log_time = now
            time.sleep(0.05)

    def get_handover_default_tcp_xy(self):
        try:
            default_tcp_pose = self.rtde_c.getForwardKinematics(HANDOVER_JOINT_SHAPE)
            return np.array(default_tcp_pose[:2], dtype=float)
        except Exception as e:
            self.get_logger().warning(f"Failed to compute default handover TCP pose: {e}")
            return None

    def build_handover_joint_q(self, hand_base_pos):
        handover_joint_q = HANDOVER_JOINT_SHAPE.copy()
        if hand_base_pos is None:
            self.get_logger().warning("Right hand direction unavailable; using default handover base yaw.")
            return handover_joint_q

        hand_controller_xy = np.array([-hand_base_pos[0], -hand_base_pos[1]], dtype=float)
        hand_xy_norm = np.linalg.norm(hand_controller_xy)
        if hand_xy_norm < 1e-4:
            self.get_logger().warning("Right hand XY direction is too small; using default handover base yaw.")
            return handover_joint_q

        default_tcp_xy = self.get_handover_default_tcp_xy()
        if default_tcp_xy is None or np.linalg.norm(default_tcp_xy) < 1e-4:
            default_xy_angle = 0.0
        else:
            default_xy_angle = math.atan2(default_tcp_xy[1], default_tcp_xy[0])

        hand_xy_angle = math.atan2(hand_controller_xy[1], hand_controller_xy[0])
        yaw_delta = math.atan2(math.sin(hand_xy_angle - default_xy_angle), math.cos(hand_xy_angle - default_xy_angle))
        base_yaw = HANDOVER_DEFAULT_BASE_YAW + yaw_delta
        base_yaw = math.atan2(math.sin(base_yaw), math.cos(base_yaw))
        handover_joint_q[0] = base_yaw
        self.get_logger().info(
            f"Physical right hand direction set handover base yaw to {math.degrees(base_yaw):.1f} deg "
            f"(delta {math.degrees(yaw_delta):.1f} deg)."
        )
        return handover_joint_q

    def get_handover_orientation_for_hand(self, hand_base_pos):
        handover_joint_q = self.build_handover_joint_q(hand_base_pos)
        try:
            facing_tcp_pose = self.rtde_c.getForwardKinematics(handover_joint_q)
            return facing_tcp_pose[3], facing_tcp_pose[4], facing_tcp_pose[5]
        except Exception as e:
            self.get_logger().warning(f"Failed to compute hand-facing orientation: {e}")
            current_tcp_pose = self.rtde_r.getActualTCPPose()
            return current_tcp_pose[3], current_tcp_pose[4], current_tcp_pose[5]

    def build_follow_right_hand_pose(self, hand_base_pos):
        if self.handover_follow_pose is None:
            self.handover_follow_pose = list(self.rtde_r.getActualTCPPose())
        if self.handover_follow_z_offset is None:
            self.handover_follow_z_offset = FOLLOW_RIGHT_HAND_Z_OFFSET

        current_tcp_pose = self.rtde_r.getActualTCPPose()
        current_controller_xy = np.array(current_tcp_pose[:2], dtype=float)
        hand_base_xy = np.array(hand_base_pos[:2], dtype=float)
        hand_norm = np.linalg.norm(hand_base_xy)
        if hand_norm < 1e-4:
            hand_dir = np.array([1.0, 0.0], dtype=float)
        else:
            hand_dir = hand_base_xy / hand_norm

        follow_base_xy = hand_base_xy - hand_dir * FOLLOW_RIGHT_HAND_DISTANCE
        target_controller_xy = np.array([-follow_base_xy[0], -follow_base_xy[1]], dtype=float)
        delta_xy = target_controller_xy - current_controller_xy
        delta_norm = np.linalg.norm(delta_xy)
        if delta_norm > FOLLOW_MAX_XY_STEP:
            target_controller_xy = current_controller_xy + delta_xy / delta_norm * FOLLOW_MAX_XY_STEP

        current_z = float(current_tcp_pose[2])
        target_z = float(hand_base_pos[2]) + float(self.handover_follow_z_offset)
        delta_z = target_z - current_z
        if abs(delta_z) > FOLLOW_MAX_Z_STEP:
            target_z = current_z + math.copysign(FOLLOW_MAX_Z_STEP, delta_z)

        rx, ry, rz = self.handover_follow_pose[3], self.handover_follow_pose[4], self.handover_follow_pose[5]
        return [
            target_controller_xy[0],
            target_controller_xy[1],
            target_z,
            rx,
            ry,
            rz,
        ]

    def follow_right_hand(self, hand_base_pos):
        if not self.rtde_c:
            self.get_logger().error("RTDE not connected.")
            return

        follow_pose = self.build_follow_right_hand_pose(hand_base_pos)
        self.get_logger().info(f"Following physical right hand at fixed handover orientation: {follow_pose}")
        self.rtde_c.moveL(follow_pose, FOLLOW_MOVE_SPEED, FOLLOW_MOVE_ACCEL)

    def execute_grasp(self, gg, viz_process=None):
        """Use ZED in-hand 6D grasp pose execution, then enter handover-follow state."""
        if not self.rtde_c: 
            self.get_logger().error("RTDE not connected.")
            return False

        def close_viz_process():
            if viz_process is None:
                return
            try:
                if viz_process.is_alive():
                    viz_process.terminate()
                    viz_process.join(timeout=1.0)
            except Exception as e:
                self.get_logger().warning(f"Failed to close Open3D visualization process: {e}")

        try:
            best_grasp = gg[0]
            trans_cam = best_grasp.translation
            rot_mat_cam = best_grasp.rotation_matrix

            # Transform grasp pose from ZED in-hand camera frame to base_link
            tf_msg = self.tf_buffer.lookup_transform('base_link', 'zed_inhand_camera_frame_optical', rclpy.time.Time())
            tf_trans = np.array([
                tf_msg.transform.translation.x,
                tf_msg.transform.translation.y,
                tf_msg.transform.translation.z
            ])
            tf_rot = R.from_quat([tf_msg.transform.rotation.x, tf_msg.transform.rotation.y, tf_msg.transform.rotation.z, tf_msg.transform.rotation.w])

            base_pos = tf_rot.apply(trans_cam) + tf_trans
            base_rot_mat = tf_rot.as_matrix() @ rot_mat_cam
            rot_ctrl = BASE_TO_CONTROLLER_ROT @ base_rot_mat

            grasp_approach_ctrl = rot_ctrl[:, 0]
            grasp_closing_ctrl = rot_ctrl[:, 1]

            tool_z = grasp_approach_ctrl / (np.linalg.norm(grasp_approach_ctrl) + 1e-8)
            tool_x = grasp_closing_ctrl - np.dot(grasp_closing_ctrl, tool_z) * tool_z
            tool_x = tool_x / (np.linalg.norm(tool_x) + 1e-8)
            tool_y = np.cross(tool_z, tool_x)
            tool_y = tool_y / (np.linalg.norm(tool_y) + 1e-8)

            r_target_primary = np.column_stack([tool_x, tool_y, tool_z])
            r_target_flip = np.column_stack([-tool_x, -tool_y, tool_z])

            current_tcp_pose = self.rtde_r.getActualTCPPose()
            r_curr, _ = cv2.Rodrigues(np.array(current_tcp_pose[3:]))

            def rotation_distance(r_candidate):
                r_delta = r_curr.T @ r_candidate
                trace_val = np.clip((np.trace(r_delta) - 1.0) * 0.5, -1.0, 1.0)
                return math.acos(trace_val)

            if rotation_distance(r_target_flip) < rotation_distance(r_target_primary):
                r_target = r_target_flip
            else:
                r_target = r_target_primary

            rvec, _ = cv2.Rodrigues(r_target)
            rx, ry, rz = rvec.flatten()

            controller_x = -base_pos[0]
            controller_y = -base_pos[1]
            controller_z = base_pos[2] - Z_OFFSET_DOWN
            target_pose = [controller_x, controller_y, controller_z, rx, ry, rz]
            self.get_logger().info(f"Target pose (Controller): {target_pose}")

            approach_pose = list(target_pose)
            tool_z_ctrl = r_target[:, 2]
            approach_pose[0] -= BACKOFF_DIST * tool_z_ctrl[0]
            approach_pose[1] -= BACKOFF_DIST * tool_z_ctrl[1]
            approach_pose[2] -= BACKOFF_DIST * tool_z_ctrl[2]

            start_joint_q = [0.0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0.0]
            if self.dynamic_start_pose:
                self.get_logger().info("Moving to Dynamic Start Position...")
                self.rtde_c.moveL(
                    self.dynamic_start_pose,
                    WORKSPACE_OBSERVE_SLOW_MOVE_SPEED,
                    WORKSPACE_OBSERVE_SLOW_MOVE_ACCEL,
                )
            else:
                self.get_logger().info("Moving to Start Joint Position...")
                self.rtde_c.moveJ(start_joint_q, 0.5, 0.5)
            self.send_gripper_command(0.085)

            while True:
                ans = input("\nReady to grasp? Press 'y' to continue, 'n' to cancel: ").strip().lower()
                print()
                if ans == "y":
                    close_viz_process()
                    break
                if ans == "n":
                    close_viz_process()
                    self.get_logger().info("Grasp cancelled.")
                    return False
                self.get_logger().info("Please enter 'y' to continue or 'n' to cancel.")

            self.get_logger().info("Moving to approach pose...")
            self.rtde_c.moveL(approach_pose, 0.05, 0.05)
            self.get_logger().info("Moving to target pose...")
            self.rtde_c.moveL(target_pose, 0.05, 0.05)
            self.get_logger().info("Closing gripper...")
            self.send_gripper_command(0.0)
            
            self.get_logger().info("Lifting...")
            self.rtde_c.moveL(approach_pose, 0.05, 0.05)
            
            # === Phase 2: Move to Handover Observation Position ===
            global STATE
            STATE = "WAITING_HANDOVER_START_POSE"
            self.get_logger().info("Waiting for MediaPipe Right hand before moving to handover position...")
            right_hand_base_pos = self.wait_for_right_hand_base_pos(timeout=None)
            if right_hand_base_pos is None:
                self.get_logger().warning("Right hand not available. Skip moving to handover position.")
                return False
            handover_joint_q = self.build_handover_joint_q(right_hand_base_pos)
            self.get_logger().info("Moving to Handover Observation Position...")
            self.rtde_c.moveJ(handover_joint_q, 0.5, 0.5)
            self.handover_follow_pose = list(self.rtde_r.getActualTCPPose())
            self.handover_follow_z_offset = None
            
            self.get_logger().info("Object is held. Starting to follow the right hand.")
            STATE = "FOLLOWING_HAND"
            return True
            
        except Exception as e:
            self.get_logger().error(f"Grasp execution failed: {e}")
            return False

    def deliver_to_hand(self, hand_base_pos):
        # Reference the handover delivery rhythm from robot_to_human_delivery.py:
        # move above the hand center in base_link z, descend to the hand center,
        # release, then retreat back upward in base_link z.
        try:
            current_tcp_pose = self.rtde_r.getActualTCPPose()
            rx, ry, rz = current_tcp_pose[3], current_tcp_pose[4], current_tcp_pose[5]

            c_x = -hand_base_pos[0]
            c_y = -hand_base_pos[1]
            c_z = hand_base_pos[2]

            deliver_target_pose = [c_x, c_y, c_z, rx, ry, rz]
            deliver_approach_pose = list(deliver_target_pose)
            deliver_approach_pose[2] += 0.05

            self.get_logger().info(f"Approaching above right-hand center in base_link z: {deliver_approach_pose}")
            self.rtde_c.moveL(deliver_approach_pose, 0.15, 0.15)
            time.sleep(0.3)
            self.get_logger().info(f"Descending to right-hand center for release: {deliver_target_pose}")
            self.rtde_c.moveL(deliver_target_pose, 0.1, 0.1)
            
            self.send_gripper_command(0.085)
            self.get_logger().info("Object Released to Human.")
            time.sleep(0.5)

            self.rtde_c.moveL(deliver_approach_pose, 0.15, 0.15)
            
            try:
                current_q = self.rtde_r.getActualQ()
                initial_q = [current_q[0], -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0.0]
                self.get_logger().info(f"Returning to initial joint shape before workspace start: {initial_q}")
                self.rtde_c.moveJ(initial_q, 0.30, 0.30)
            except Exception as e:
                self.get_logger().error(f"Failed to move to initial joint shape after delivery: {e}")

            if self.dynamic_start_pose:
                self.get_logger().info("Moving from initial joint shape to workspace start pose...")
                self.rtde_c.moveL(self.dynamic_start_pose, 0.15, 0.15)
            else:
                start_joint_q = [0.0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0.0]
                self.rtde_c.moveJ(start_joint_q, 0.15, 0.15)
            self.get_logger().info("Returned to workspace/start position.")
            
            global STATE
            STATE = "IDLE"
            global request_selection_prompt
            request_selection_prompt = True
            self.handover_follow_pose = None
            self.handover_follow_z_offset = None
            
        except Exception as e:
            self.get_logger().error(f"Delivery failed: {e}")

# === MLLM 工作台识别 API 调用 ===
def call_doubao_workspace(image_bgr):
    if not ARK_API_KEY:
        logging.error("ARK_API_KEY not set.")
        return None

    _, buffer = cv2.imencode('.jpg', image_bgr)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{img_base64}"

    grounding_text = (
        "Instructions: \n"
        "1. Detect the main workspace, desk, table, or flat surface where objects are located for manipulation.\n"
        "2. Provide its location in JSON format as follows: \n"
        "   {\"workspace\": [ymin, xmin, ymax, xmax]}\n"
        "   Coordinates should be normalized 0-1000. \n"
        "3. If no clear workspace is detected, return exactly: 'No workspace detected'."
    )
    
    try:
        kwargs = {
            "model": MODEL_NAME,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": "You are a professional vision assistant for robot perception."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": data_url},
                        {"type": "input_text", "text": grounding_text},
                    ],
                }
            ]
        }
        if REASONING_EFFORT != "minimal":
            kwargs["reasoning_effort"] = REASONING_EFFORT

        response = client.responses.create(**kwargs)
        
        content = ""
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
        elif hasattr(response, 'output') and len(response.output) > 1:
             content = response.output[1].content[0].text
        else:
            content = str(response)

        logging.info(f"Doubao Workspace Response: {content}")

        if "No workspace detected" in content:
            logging.warning("MLLM: Workspace not found in frame.")
            return None

        try:
            json_str = re.search(r'\{.*\}', content, re.DOTALL)
            if json_str:
                data = json.loads(json_str.group())
                h, w = image_bgr.shape[:2]
                
                if "workspace" in data:
                    ymin, xmin, ymax, xmax = data["workspace"]
                    bbox = (int(xmin*w/1000), int(ymin*h/1000), int(xmax*w/1000), int(ymax*h/1000))
                    return bbox
        except Exception as je:
            logging.error(f"JSON Parse failed: {je}")
            
        return None
    except Exception as e:
        logging.error(f"API call failed: {e}")
        return None

async def upload_audio_prompt_file(audio_path):
    """Upload a local audio file to Ark and return an input_audio content item."""
    if not audio_path:
        return None

    path = Path(audio_path)
    if not path.is_file():
        logging.error(f"Audio prompt file not found: {audio_path}")
        return None

    audio_format = AUDIO_FORMAT_MAP.get(path.suffix.lower())
    if audio_format is None:
        logging.error(f"Unsupported audio prompt format: {path.suffix}. Supported: {sorted(AUDIO_FORMAT_MAP.values())}")
        return None

    try:
        with path.open("rb") as f:
            uploaded = await async_client.files.create(
                file=f,
                purpose="user_data",
            )
        file_id = getattr(uploaded, "id", None)
        if not file_id:
            logging.error("Ark file upload succeeded but no file id was returned.")
            return None

        await async_client.files.wait_for_processing(file_id)

        logging.info(f"Ark audio file uploaded: {file_id}")
    except Exception as exc:
        logging.error(f"Failed to upload audio prompt file {audio_path}: {exc}")
        return None

    return {
        "type": "input_audio",
        "file_id": file_id,
    }


def record_audio_prompt(output_path, duration_sec=4.0, sample_rate=16000):
    """Record audio and save it as MP3, stopping when the user presses Enter again."""
    arecord_path = shutil.which("arecord")
    if arecord_path is None:
        logging.error("`arecord` not found, cannot capture voice prompt from microphone.")
        return False
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        logging.error("`ffmpeg` not found, cannot convert voice prompt to mp3.")
        return False

    max_duration_sec = max(1.0, float(duration_sec))
    wav_output_path = f"{output_path}.wav"
    cmd = [
        arecord_path,
        "-q",
        "-f", "S16_LE",
        "-r", str(sample_rate),
        "-c", "1",
        "-t", "wav",
        wav_output_path,
    ]

    proc = None

    try:
        input("[Object Selection] Press Enter to start voice recording...")
        logging.info(f"Recording voice prompt. Press Enter to stop; max {max_duration_sec:.0f}s.")
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def _stop_recording():
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1.0)

        stopper = threading.Timer(max_duration_sec, _stop_recording)
        stopper.start()
        try:
            input("[Object Selection] Recording... Press Enter to stop.")
        finally:
            stopper.cancel()
            _stop_recording()

        if not os.path.exists(wav_output_path) or os.path.getsize(wav_output_path) <= 44:
            logging.warning("Recorded voice prompt is empty.")
            return False

        convert_cmd = [
            ffmpeg_path,
            "-y",
            "-i", wav_output_path,
            "-codec:a", "libmp3lame",
            "-q:a", "2",
            output_path,
        ]
        subprocess.run(convert_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            logging.warning("Converted mp3 voice prompt is empty.")
            return False

        logging.info(f"Voice prompt saved to: {output_path}")
        return True
    except Exception as exc:
        logging.error(f"Audio recording failed: {exc}")
        return False
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=1.0)
        if os.path.exists(wav_output_path) and not DEBUG_MODE:
            try:
                os.remove(wav_output_path)
            except OSError:
                pass


async def _call_doubao_object_bbox_async(image_bgr, user_prompt=None, audio_prompt_path=None):
    """Use Doubao MLLM to locate the user-described object from text/audio prompt."""
    if not ARK_API_KEY:
        logging.error("ARK_API_KEY not set.")
        return None

    if image_bgr is None:
        logging.error("Object selection image is empty.")
        return None

    user_prompt = (user_prompt or "").strip()
    if audio_prompt_path:
        try:
            save_audio_prompt_artifact(audio_prompt_path)
        except Exception as exc:
            logging.warning(f"Failed to save audio prompt artifact: {exc}")
    audio_content = await upload_audio_prompt_file(audio_prompt_path) if audio_prompt_path else None
    if not user_prompt and audio_content is None:
        logging.warning("Neither text prompt nor audio prompt was provided.")
        return None

    _, buffer = cv2.imencode('.jpg', image_bgr)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{img_base64}"

    grounding_text = (
        "Instructions:\n"
        "1. Find the single best-matching object in the image according to the user's request.\n"
        "2. The request may be provided as text or audio.\n"
        "3. Return exactly one JSON object in this format:\n"
        "   {\"object\": [xmin, ymin, xmax, ymax], \"label\": \"english name\"}\n"
        "   Coordinates must be normalized in [0, 1000].\n"
        "   label must be a short English object name.\n"
        "4. If the target object is not found, return exactly: 'No object detected'."
    )

    user_content = [
        {"type": "input_image", "image_url": data_url},
        {"type": "input_text", "text": grounding_text},
    ]
    if user_prompt:
        user_content.append({"type": "input_text", "text": f"Text prompt from user: {user_prompt}"})
    if audio_content is not None:
        user_content.append(audio_content)

    try:
        kwargs = {
            "model": MODEL_NAME,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": "You are a professional vision grounding assistant for robot grasping."}]
                },
                {
                    "role": "user",
                    "content": user_content,
                }
            ]
        }
        if REASONING_EFFORT != "minimal":
            kwargs["reasoning_effort"] = REASONING_EFFORT

        response = await async_client.responses.create(**kwargs)

        content = ""
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
        elif hasattr(response, 'output') and len(response.output) > 1:
            content = response.output[1].content[0].text
        else:
            content = str(response)

        logging.info(f"Doubao Object Response: {content}")

        if "No object detected" in content:
            logging.warning("MLLM: target object not found in frame.")
            return None

        json_str = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_str:
            logging.error("MLLM response has no JSON object for target bbox.")
            return None

        data = json.loads(json_str.group())
        if "object" not in data:
            logging.error("MLLM JSON missing 'object' key.")
            return None

        values = data["object"]
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            logging.error("MLLM object bbox format invalid.")
            return None

        h, w = image_bgr.shape[:2]
        xmin, ymin, xmax, ymax = [float(v) for v in values]
        xmin = np.clip(xmin, 0.0, 1000.0)
        ymin = np.clip(ymin, 0.0, 1000.0)
        xmax = np.clip(xmax, 0.0, 1000.0)
        ymax = np.clip(ymax, 0.0, 1000.0)

        x1 = int(min(xmin, xmax) * w / 1000.0)
        y1 = int(min(ymin, ymax) * h / 1000.0)
        x2 = int(max(xmin, xmax) * w / 1000.0)
        y2 = int(max(ymin, ymax) * h / 1000.0)

        x1 = int(np.clip(x1, 0, w - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        y2 = int(np.clip(y2, 0, h - 1))
        if (x2 - x1) < 8 or (y2 - y1) < 8:
            logging.warning("MLLM target bbox is too small. Please try a clearer prompt.")
            return None

        detected_label = str(data.get("label") or "object").strip() or "object"
        try:
            save_object_bbox_artifacts(
                image_bgr=image_bgr,
                bbox_xyxy=(x1, y1, x2, y2),
                label=detected_label,
                backend="doubao",
                prefix="robot2human_bbox",
            )
        except Exception as e:
            logging.warning(f"Failed to annotate/save Doubao bbox image: {e}")

        return {"bbox": (x1, y1, x2, y2), "label": detected_label}
    except Exception as e:
        logging.error(f"Object API call failed: {e}")
        return None


def call_doubao_object_bbox(image_bgr, user_prompt=None, audio_prompt_path=None):
    """Sync wrapper for object selection; audio path uses AsyncArk file upload flow."""
    return asyncio.run(
        _call_doubao_object_bbox_async(
            image_bgr=image_bgr,
            user_prompt=user_prompt,
            audio_prompt_path=audio_prompt_path,
        )
    )

def save_audio_prompt_artifact(audio_path, prefix="robot2human_prompt_audio"):
    """Save the audio prompt under ./llm_output/<timestamp>/ for debugging."""
    if not DEBUG_MODE or not audio_path:
        return None

    src_path = Path(audio_path)
    if not src_path.is_file():
        return None

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = Path.cwd() / "llm_output" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = src_path.suffix or ".mp3"
    dst_path = out_dir / f"{prefix}{suffix}"
    shutil.copy2(src_path, dst_path)
    logging.info(f"Saved audio prompt file: {dst_path}")
    return str(dst_path)

def save_object_bbox_artifacts(image_bgr, bbox_xyxy, label, backend, prefix="object_bbox"):
    """Save bbox JSON and visualization image under ./llm_output/<timestamp>/."""
    if not DEBUG_MODE:
        return None, None
    if image_bgr is None or bbox_xyxy is None:
        return None, None

    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = Path.cwd() / "llm_output" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_backend = (backend or "vlm").strip().lower()
    safe_prefix = (prefix or "object_bbox").strip()
    json_path = out_dir / f"{safe_prefix}_{safe_backend}.json"
    img_path = out_dir / f"{safe_prefix}_{safe_backend}.jpg"

    payload = {
        "timestamp": ts,
        "backend": safe_backend,
        "label": str(label or "object"),
        "bbox_xyxy": [x1, y1, x2, y2],
        "image_size": {"width": int(image_bgr.shape[1]), "height": int(image_bgr.shape[0])},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    vis = image_bgr.copy()
    draw_label = str(label or "object")
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis, draw_label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(img_path), vis)

    logging.info(f"Saved MLLM bbox JSON: {json_path}")
    logging.info(f"Saved MLLM bbox image: {img_path}")
    return str(json_path), str(img_path)

def save_mediapipe_intent_frame(image_bgr, prefix="mediapipe_intent", detected_text="Detected intent"):
    """Save the current MediaPipe-annotated frame under ./mediapipe_output/<timestamp>/."""
    if not DEBUG_MODE:
        return None
    if image_bgr is None:
        return None

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = Path.cwd() / "mediapipe_output" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"{prefix}.jpg"
    vis = image_bgr.copy()
    cv2.putText(vis, detected_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    if cv2.imwrite(str(img_path), vis):
        logging.info(f"Saved MediaPipe intent frame: {img_path}")
        return str(img_path)

    logging.warning("Failed to save MediaPipe intent frame.")
    return None

def vlm_thread_worker(image_bgr):
    global workspace_bbox, vlm_running
    try:
        bbox = call_doubao_workspace(image_bgr)
        if bbox:
            workspace_bbox = bbox
            logging.info(f"MLLM Worker: Found workspace at {bbox}")
    finally:
        vlm_running = False

# === 工作台平面法向量评估 ===
def estimate_workspace_normal(bbox, depth_img, camera_info, margin=15):
    """基于工作区域边界框，提取局部点云并用 RANSAC 拟合工作平面提取法向量"""
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 + margin)
    y1 = max(0, y1 + margin)
    x2 = min(depth_img.shape[1], x2 - margin)
    y2 = min(depth_img.shape[0], y2 - margin)
    
    crop_mask = np.zeros(depth_img.shape, dtype=bool)
    crop_mask[y1:y2, x1:x2] = True
    
    # 获取非空深度的掩码点
    mask = (crop_mask) & (depth_img > 0)
    
    # 构建完整点云，提取框内的有效点
    cloud = create_point_cloud_from_depth_image(depth_img, camera_info, organized=False)
    depth_flat = depth_img.flatten()
    mask_flat = mask.flatten()
    points = cloud[mask_flat]
    
    # 滤除原点异常值
    points = points[np.linalg.norm(points, axis=1) > 0.1]
    
    default_normal = np.array([0., 0., -1.]) # 默认朝向相机

    if len(points) < 500:
        logging.warning("Workspace points too few, using default Z-axis logic.")
        return default_normal
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    
    if len(pcd.points) < 30:
        return default_normal
        
    try:
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
        normal = np.array(plane_model[:3])
        
        # 相机视野为光学坐标系(Z向前)，平面法向应该是指向相机的，使得其 Z 轴分量一般为负
        if normal[2] > 0:
            normal = -normal
            
        logging.info(f"Estimated workspace normal (in camera frame): {normal}")
        return normal
    except Exception as e:
        logging.warning(f"Plane estimation failed: {e}. Fallback to default.")
        return default_normal

def get_net(checkpoint_path, num_view=300):
    net = GraspNet(input_feature_dim=0, num_view=num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net, device

def start_zed():
    logging.info("Initializing ZED Camera...")
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.coordinate_units = sl.UNIT.METER

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED Camera: {err}")

    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters
    left_cam = calib.left_cam
    baseline = calib.get_camera_baseline()

    fx_scaled = left_cam.fx * SCALE_FACTOR
    fy_scaled = left_cam.fy * SCALE_FACTOR
    cx_scaled = left_cam.cx * SCALE_FACTOR
    cy_scaled = left_cam.cy * SCALE_FACTOR

    K = np.array([
        [fx_scaled, 0, cx_scaled],
        [0, fy_scaled, cy_scaled],
        [0, 0, 1]
    ], dtype=np.float32)

    camera_info_graspnet = CameraInfo(float(IMG_WIDTH), float(IMG_HEIGHT), fx_scaled, fy_scaled, cx_scaled, cy_scaled, 1000.0)
    
    return zed, K, baseline, camera_info_graspnet

def start_realsense():
    logging.info("Initializing RealSense for object segmentation/grasp...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    factor_depth = 1.0 / depth_scale

    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intrinsics = color_profile.get_intrinsics()
    camera_info_graspnet = CameraInfo(
        640.0,
        480.0,
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.ppx,
        intrinsics.ppy,
        factor_depth,
    )
    return pipeline, align, camera_info_graspnet

# --- SAM2 Global State & Mouse Callback ---
drawing = False
ix, iy, fx_mouse, fy_mouse = -1, -1, -1, -1
pending_bbox = None
pending_point = None

def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, fx_mouse, fy_mouse, pending_bbox, pending_point

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        fx_mouse, fy_mouse = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx_mouse, fy_mouse = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx_mouse, fy_mouse = x, y
        dx = abs(fx_mouse - ix)
        dy = abs(fy_mouse - iy)
        if dx > 8 and dy > 8:
            x1, y1 = min(ix, fx_mouse), min(iy, fy_mouse)
            x2, y2 = max(ix, fx_mouse), max(iy, fy_mouse)
            pending_bbox = (x1, y1, x2, y2)
        else:
            pending_point = (x, y)

def process_frame(color_img, depth_img, camera, device, sam_mask=None, num_point=20000):
    color = color_img.astype(np.float32) / 255.0
    
    if sam_mask is not None and np.any(sam_mask):
        workspace_mask = (sam_mask > 0) & (depth_img > 0) & (depth_img < 2000)
    else:
        z_min_mm, z_max_mm = 100, 1000 
        workspace_mask = (depth_img > z_min_mm) & (depth_img < z_max_mm)
    
    cloud = create_point_cloud_from_depth_image(depth_img, camera, organized=True)
    mask = (workspace_mask & (depth_img > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]
    
    if len(cloud_masked) == 0:
        return None, None, None

    cloud_masked_denoised, color_masked_denoised = denoise_points(cloud_masked, color_masked)
    if len(cloud_masked_denoised) > 0:
        cloud_masked = cloud_masked_denoised
        color_masked = color_masked_denoised

    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
        
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    valid_depth_mask = depth_img > 0
    cloud_full = cloud[valid_depth_mask]
    color_full = color[valid_depth_mask]
    cloud_full_denoised, color_full_denoised = denoise_points(cloud_full, color_full)
    if len(cloud_full_denoised) > 0:
        cloud_full = cloud_full_denoised
        color_full = color_full_denoised
    
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_full.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_full.astype(np.float32))
    
    end_points = dict()
    cloud_sampled_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device, dtype=torch.float32)
    end_points['point_clouds'] = cloud_sampled_tensor
    end_points['cloud_colors'] = color_sampled
    
    return end_points, cloud_o3d, workspace_mask


def denoise_points(points, colors):
    if len(points) == 0:
        return points, colors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    pcd = pcd.voxel_down_sample(voxel_size=DENOISE_VOXEL_SIZE)
    if len(pcd.points) == 0:
        return points, colors

    _, ind = pcd.remove_radius_outlier(
        nb_points=DENOISE_NB_POINTS,
        radius=DENOISE_RADIUS
    )
    if len(ind) == 0:
        return points, colors

    pcd = pcd.select_by_index(ind)
    return np.asarray(pcd.points, dtype=np.float32), np.asarray(pcd.colors, dtype=np.float32)


def select_grasp_target_mask(current_masks, prompts):
    if not current_masks:
        return None

    for prompt in reversed(prompts):
        target_id = prompt.get("id")
        mask = current_masks.get(target_id)
        if mask is not None and np.any(mask):
            return mask > 0

    for mask in current_masks.values():
        if mask is not None and np.any(mask):
            return mask > 0

    return None

def estimate_workspace_normal_from_points(scene_points):
    default_normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    if len(scene_points) < 100:
        return default_normal

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene_points.astype(np.float64))

    try:
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    except Exception as e:
        logging.warning(f"Plane estimation failed: {e}")
        return default_normal

    if len(inliers) < 50:
        return default_normal

    normal = np.asarray(plane_model[:3], dtype=np.float32)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-8:
        return default_normal

    normal = normal / normal_norm
    if normal[2] > 0:
        normal = -normal
    return normal

def filter_grasps_by_camera_direction(gg, camera_rot_mat):
    if len(gg) == 0:
        return gg

    camera_forward_base = camera_rot_mat[:, 2]
    camera_forward_base = camera_forward_base / (np.linalg.norm(camera_forward_base) + 1e-8)
    max_tilt_cos = math.cos(math.radians(CAMERA_GRASP_MAX_TILT_DEG))

    keep_mask = []
    for grasp in gg:
        rot_mat = grasp.rotation_matrix
        approach_camera = rot_mat[:, 0]

        approach_camera = approach_camera / (np.linalg.norm(approach_camera) + 1e-8)
        approach_base = camera_rot_mat @ approach_camera

        approach_base = approach_base / (np.linalg.norm(approach_base) + 1e-8)

        cos_tilt = float(np.clip(np.dot(approach_base, camera_forward_base), -1.0, 1.0))
        keep_mask.append(cos_tilt >= max_tilt_cos)

    keep_mask = np.asarray(keep_mask, dtype=bool)
    return gg[keep_mask]


def select_physical_hand(results, mp_label):
    if not results.multi_hand_landmarks or not results.multi_handedness:
        return None, None

    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        if not handedness.classification:
            continue
        hand_cls = handedness.classification[0]
        if hand_cls.label == mp_label:
            return hand_landmarks, hand_cls.score

    return None, None

def select_right_hand(results):
    return select_physical_hand(results, PHYSICAL_RIGHT_HAND_MP_LABEL)

def landmark_dist2d(lm1, lm2):
    return math.hypot(lm1.x - lm2.x, lm1.y - lm2.y)

def is_fist(hand_landmarks):
    wrist_lm = hand_landmarks.landmark[0]
    tips = [8, 12, 16, 20]
    mcps = [5, 9, 13, 17]
    curled_count = sum(
        1
        for tip_idx, mcp_idx in zip(tips, mcps)
        if landmark_dist2d(hand_landmarks.landmark[tip_idx], wrist_lm)
        < landmark_dist2d(hand_landmarks.landmark[mcp_idx], wrist_lm) * 1.2
    )
    return curled_count == 4

def calc_hand_base_pos_from_ffs(l_rgb, r_rgb, K_s, bl, pt_x, pt_y, max_depth=1.5, log_prefix="hand"):
    logging.info(f"Calculating depth for {log_prefix} via FFS...")
    with torch.autograd.set_grad_enabled(False):
        ffs_model = torch.load(MODEL_DIR, map_location='cpu', weights_only=False)
        ffs_model.args.valid_iters = VALID_ITERS
        ffs_model.args.max_disp = MAX_DISP
        ffs_model.cuda().eval()
        
        img0 = torch.as_tensor(l_rgb).cuda().float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(r_rgb).cuda().float()[None].permute(0, 3, 1, 2)
        
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0_p, img1_p = padder.pad(img0, img1)

        with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
            disp = ffs_model.forward(img0_p, img1_p, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
        
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(IMG_HEIGHT, IMG_WIDTH).clip(0, None)
    
    del ffs_model
    del img0, img1, img0_p, img1_p
    torch.cuda.empty_cache()

    xx = np.arange(IMG_WIDTH)[None, :].repeat(IMG_HEIGHT, axis=0)
    invalid = (xx - disp) < 0
    disp[invalid] = np.inf

    with np.errstate(divide='ignore', invalid='ignore'):
        depth_ffs = K_s[0, 0] * bl / disp

    pt_x = int(np.clip(pt_x, 0, IMG_WIDTH - 1))
    pt_y = int(np.clip(pt_y, 0, IMG_HEIGHT - 1))
    depth_val = depth_ffs[pt_y, pt_x]
    
    if not (0.1 < depth_val < max_depth):
        logging.warning(f"Calculated depth for {log_prefix} is invalid/out-of-bounds.")
        return None

    fx = K_s[0, 0]
    fy = K_s[1, 1]
    px = K_s[0, 2]
    py = K_s[1, 2]
    x_cam = (pt_x - px) * depth_val / fx
    y_cam = (pt_y - py) * depth_val / fy
    z_cam = depth_val
    point_3d_cam = np.array([x_cam, y_cam, z_cam])
    
    tf_msg = ROS_NODE.tf_buffer.lookup_transform('base_link', 'rs_tohand_color_optical_frame', rclpy.time.Time())
    tf_rot = R.from_quat([tf_msg.transform.rotation.x, tf_msg.transform.rotation.y, tf_msg.transform.rotation.z, tf_msg.transform.rotation.w])
    tf_trans = np.array([tf_msg.transform.translation.x, tf_msg.transform.translation.y, tf_msg.transform.translation.z])
    
    return tf_rot.apply(point_3d_cam) + tf_trans


def compute_ffs_depth_map(l_rgb, r_rgb, K_s, bl):
    """Use ZED stereo pair + FFS to estimate depth map (unit: mm)."""
    depth_mm, _ = compute_ffs_depth_and_disp(l_rgb, r_rgb, K_s, bl)
    return depth_mm


def compute_ffs_depth_and_disp(l_rgb, r_rgb, K_s, bl):
    """Use ZED stereo pair + FFS to estimate depth map (mm) and disparity."""
    with torch.autograd.set_grad_enabled(False):
        ffs_model = torch.load(MODEL_DIR, map_location='cpu', weights_only=False)
        ffs_model.args.valid_iters = VALID_ITERS
        ffs_model.args.max_disp = MAX_DISP
        ffs_model.cuda().eval()

        img0 = torch.as_tensor(l_rgb).cuda().float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(r_rgb).cuda().float()[None].permute(0, 3, 1, 2)

        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0_p, img1_p = padder.pad(img0, img1)

        with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
            disp = ffs_model.forward(img0_p, img1_p, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')

        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(IMG_HEIGHT, IMG_WIDTH).clip(0, None)

    del ffs_model
    del img0, img1, img0_p, img1_p
    torch.cuda.empty_cache()

    xx = np.arange(IMG_WIDTH)[None, :].repeat(IMG_HEIGHT, axis=0)
    invalid = (xx - disp) < 0
    disp[invalid] = np.inf

    with np.errstate(divide='ignore', invalid='ignore'):
        depth_m = K_s[0, 0] * bl / disp
    depth_m[~np.isfinite(depth_m)] = 0.0
    depth_m[(depth_m < 0.1) | (depth_m > 2.0)] = 0.0
    depth_mm = (depth_m * 1000.0).astype(np.uint16)
    return depth_mm, disp


def add_colorbar_and_text(vis_bgr, vmin, vmax, title, invalid_mask=None, invert_labels=False):
    h, w = vis_bgr.shape[:2]
    bar_w = 30
    gap = 10
    top_pad = 32
    bottom_pad = 26
    canvas_h = h + top_pad + bottom_pad
    canvas_w = w + gap + bar_w + 90
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    canvas[top_pad: top_pad + h, :w] = vis_bgr

    bar_x0 = w + gap
    bar_x1 = bar_x0 + bar_w
    grad = np.linspace(255, 0, h, dtype=np.uint8).reshape(h, 1)
    grad = np.repeat(grad, bar_w, axis=1)
    bar = cv2.applyColorMap(grad, cv2.COLORMAP_TURBO)
    canvas[top_pad: top_pad + h, bar_x0:bar_x1] = bar

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, title, (8, 24), font, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
    top_label = f"min: {vmin:.3f}" if invert_labels else f"max: {vmax:.3f}"
    bottom_label = f"max: {vmax:.3f}" if invert_labels else f"min: {vmin:.3f}"
    cv2.putText(canvas, top_label, (bar_x1 + 8, top_pad + 14), font, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, bottom_label, (bar_x1 + 8, top_pad + h - 6), font, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    if invalid_mask is not None:
        invalid_ratio = float(invalid_mask.mean()) * 100.0
        cv2.putText(canvas, f"invalid: {invalid_ratio:.1f}%", (8, canvas_h - 8), font, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def save_depth_vis(depth_m, out_path):
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(valid):
        vis = np.full((depth_m.shape[0], depth_m.shape[1], 3), 255, dtype=np.uint8)
        dmin, dmax = ZNEAR, ZFAR
    else:
        d = depth_m.copy()
        d[~valid] = 0
        dmin = max(ZNEAR, float(d[valid].min()))
        dmax = min(ZFAR, float(d[valid].max()))
        denom = max(dmax - dmin, 1e-6)
        inv = (1.0 - (d - dmin) / denom).clip(0, 1)
        vis = cv2.applyColorMap((inv * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        vis[~valid] = 255

    vis = add_colorbar_and_text(vis, dmin, dmax, "Depth (m)", invalid_mask=~valid, invert_labels=True)
    cv2.imwrite(out_path, vis)


def save_disparity_vis(disp, out_path):
    disp_stats = {}
    disp_vis_rgb = vis_disparity(disp, invalid_thres=np.inf, other_output=disp_stats)
    disp_valid = np.isfinite(disp) & (disp > 0)
    disp_vis_bgr = cv2.cvtColor(disp_vis_rgb, cv2.COLOR_RGB2BGR)
    disp_vis_bgr[~disp_valid] = 255

    if disp_stats.get("min_val") is None or disp_stats.get("max_val") is None:
        disp_min, disp_max = 0.0, float(MAX_DISP)
    else:
        disp_min = float(disp_stats["min_val"])
        disp_max = float(disp_stats["max_val"])

    disp_vis_bgr = add_colorbar_and_text(disp_vis_bgr, disp_min, disp_max, "Disparity (px)", invalid_mask=~disp_valid)
    cv2.imwrite(out_path, disp_vis_bgr)


def save_pointcloud_xyzrgb(points, colors_rgb, out_path):
    if len(points) == 0:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb.astype(np.float64))
    o3d.io.write_point_cloud(out_path, pcd)


def save_zed_grasp_capture_bundle(left_rgb, tracking_bgr, depth_mm, disp, camera, workspace_mask, gg_before, gg_after, note="robot2human"):
    if not DEBUG_MODE:
        return None
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path.cwd() / "captured_data" / ts
    save_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(save_dir / "sam2_input.png"), tracking_bgr)
    cv2.imwrite(str(save_dir / "left_rgb.png"), cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(save_dir / "depth_raw_mm.png"), depth_mm)
    save_disparity_vis(disp, str(save_dir / "disparity.png"))
    save_depth_vis(depth_mm.astype(np.float32) / 1000.0, str(save_dir / "depth.png"))
    cv2.imwrite(str(save_dir / "workspace_mask_used.png"), workspace_mask.astype(np.uint8) * 255)

    cloud = create_point_cloud_from_depth_image(depth_mm, camera, organized=True)
    color = left_rgb.astype(np.float32) / 255.0
    valid_depth_mask = depth_mm > 0
    full_points = cloud[valid_depth_mask]
    full_colors = color[valid_depth_mask]
    save_pointcloud_xyzrgb(full_points, full_colors, str(save_dir / "pointcloud_full.ply"))

    used_valid = workspace_mask & (depth_mm > 0)
    used_points = cloud[used_valid]
    used_colors = color[used_valid]
    save_pointcloud_xyzrgb(used_points, used_colors, str(save_dir / "pointcloud_workspace_used.ply"))

    gg_before_array = gg_before.grasp_group_array if len(gg_before) > 0 else np.empty((0, 17), dtype=np.float32)
    gg_after_array = gg_after.grasp_group_array if len(gg_after) > 0 else np.empty((0, 17), dtype=np.float32)
    np.save(str(save_dir / "grasp_before_filter.npy"), gg_before_array)
    np.save(str(save_dir / "grasp_after_filter.npy"), gg_after_array)
    save_open3d_compare_images(full_points, full_colors, gg_before_array, gg_after_array, str(save_dir))

    filter_info = {
        "timestamp": ts,
        "note": note,
        "raw_count": int(len(gg_before)),
        "selected_count": int(len(gg_after)),
        "image_size": {"width": int(left_rgb.shape[1]), "height": int(left_rgb.shape[0])},
    }
    with open(save_dir / "grasp_filter_info.json", "w", encoding="utf-8") as f:
        json.dump(filter_info, f, ensure_ascii=False, indent=2)

    meta = {
        'intrinsic_matrix': np.array([
            [camera.fx, 0, camera.cx],
            [0, camera.fy, camera.cy],
            [0, 0, 1]
        ]),
        'factor_depth': np.array([[camera.scale]])
    }
    scio.savemat(str(save_dir / "meta.mat"), meta)
    logging.info(f"Saved ZED grasp capture to {save_dir}")
    return save_dir


def calc_hand_base_pos_from_rs_depth(depth_img, camera, pt_x, pt_y, max_depth=1.5, log_prefix="hand"):
    h, w = depth_img.shape[:2]
    pt_x = int(np.clip(pt_x, 0, w - 1))
    pt_y = int(np.clip(pt_y, 0, h - 1))

    x1, x2 = max(0, pt_x - 2), min(w, pt_x + 3)
    y1, y2 = max(0, pt_y - 2), min(h, pt_y + 3)
    patch = depth_img[y1:y2, x1:x2].astype(np.float32)
    valid = patch[patch > 0]
    if valid.size == 0:
        logging.warning(f"Depth around {log_prefix} is invalid.")
        return None

    depth_raw = float(np.median(valid))
    # CameraInfo in this repo uses `scale`; keep compatibility with older field names.
    depth_scale_factor = getattr(camera, "scale", None)
    if depth_scale_factor is None:
        depth_scale_factor = getattr(camera, "factor_depth", None)
    if depth_scale_factor is None or depth_scale_factor <= 0:
        raise AttributeError("CameraInfo is missing a valid depth scale factor (`scale`).")
    depth_m = depth_raw / depth_scale_factor
    if not (0.1 < depth_m < max_depth):
        logging.warning(f"Depth for {log_prefix} is out-of-bounds: {depth_m:.3f} m")
        return None

    x_cam = (pt_x - camera.cx) * depth_m / camera.fx
    y_cam = (pt_y - camera.cy) * depth_m / camera.fy
    point_3d_cam = np.array([x_cam, y_cam, depth_m], dtype=np.float64)

    tf_msg = ROS_NODE.tf_buffer.lookup_transform('base_link', 'rs_tohand_color_optical_frame', rclpy.time.Time())
    tf_rot = R.from_quat([tf_msg.transform.rotation.x, tf_msg.transform.rotation.y, tf_msg.transform.rotation.z, tf_msg.transform.rotation.w])
    tf_trans = np.array([tf_msg.transform.translation.x, tf_msg.transform.translation.y, tf_msg.transform.translation.z], dtype=np.float64)
    return tf_rot.apply(point_3d_cam) + tf_trans

def show_open3d_process(points, colors, gg_array):
    get_open3d_viewer().update(points, colors, gg_array, top_k=1)


def _render_open3d_snapshot(points, colors, gg_array, out_path, window_name, top_k=1):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    gg = GraspGroup(gg_array).nms().sort_by_score()
    geoms = [cloud]
    if len(gg) > 0:
        show_n = min(len(gg), max(int(top_k), 1))
        geoms.extend(gg[:show_n].to_open3d_geometry_list())

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720, visible=False)
    for g in geoms:
        vis.add_geometry(g)

    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0.5])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.8)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(out_path)
    vis.destroy_window()


def save_open3d_compare_images(points, colors, gg_before_array, gg_after_array, save_dir):
    before_path = os.path.join(save_dir, "open3d_before_filter.png")
    after_path = os.path.join(save_dir, "open3d_after_filter.png")

    try:
        _render_open3d_snapshot(points, colors, gg_before_array, before_path, "Open3D Before Filter", top_k=20)
        _render_open3d_snapshot(points, colors, gg_after_array, after_path, "Open3D After Filter", top_k=1)

        img_before = cv2.imread(before_path)
        img_after = cv2.imread(after_path)
        if img_before is not None and img_after is not None and img_before.shape == img_after.shape:
            cv2.imwrite(os.path.join(save_dir, "open3d_filter_compare.png"), cv2.hconcat([img_before, img_after]))
    except Exception as e:
        logging.warning(f"Open3D screenshot save failed: {e}")


def _render_open3d_snapshot(points, colors, gg_array, out_path, window_name, top_k=1):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    gg = GraspGroup(gg_array).nms().sort_by_score()
    geoms = [cloud]
    if len(gg) > 0:
        show_n = min(len(gg), max(int(top_k), 1))
        geoms.extend(gg[:show_n].to_open3d_geometry_list())

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720, visible=False)
    for g in geoms:
        vis.add_geometry(g)

    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0.5])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.8)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(out_path)
    vis.destroy_window()


def save_open3d_compare_images(points, colors, gg_before_array, gg_after_array, save_dir):
    before_path = os.path.join(save_dir, "open3d_before_filter.png")
    after_path = os.path.join(save_dir, "open3d_after_filter.png")

    try:
        _render_open3d_snapshot(points, colors, gg_before_array, before_path, "Open3D Before Filter", top_k=20)
        _render_open3d_snapshot(points, colors, gg_after_array, after_path, "Open3D After Filter", top_k=1)

        img_before = cv2.imread(before_path)
        img_after = cv2.imread(after_path)
        if img_before is not None and img_after is not None and img_before.shape == img_after.shape:
            cv2.imwrite(os.path.join(save_dir, "open3d_filter_compare.png"), cv2.hconcat([img_before, img_after]))
    except Exception as e:
        logging.warning(f"Open3D screenshot save failed: {e}")


def main(debug=False, voice_prompt_seconds=4.0):
    global DEBUG_MODE
    DEBUG_MODE = debug
    global need_reset, num_targets, pending_bbox, pending_point, current_masks, all_prompts, drawing, ix, iy, fx_mouse, fy_mouse
    global vlm_running, doubao_trigger, workspace_bbox, ROS_NODE, STATE, request_selection_prompt
    
    rclpy.init()
    ROS_NODE = RobotControllerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(ROS_NODE)
    t_ros = threading.Thread(target=executor.spin, daemon=True)
    t_ros.start()

    if ROS_NODE.rtde_c:
        ROS_NODE.get_logger().info("【初始化】打开夹爪，并移动到初始观察姿态...")
        success = ROS_NODE.send_gripper_command(0.085)
        if not success:
            ROS_NODE.get_logger().warning("夹爪打开指令发送失败或超时，请检查 Action Server")
        
        try:
            current_q = ROS_NODE.rtde_r.getActualQ()
            # 保持当前 base 关节(yaw)不变，其它关节移动到指定形状 90度90度夹爪朝下
            initial_q = [current_q[0], -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0.0]
            ROS_NODE.get_logger().info(f"【初始化】移动到初始关节形状: {initial_q}")
            ROS_NODE.rtde_c.moveJ(initial_q, 0.5, 0.5)
        except Exception as e:
            ROS_NODE.get_logger().error(f"移动到初始关节位置失败: {e}")
            
        ROS_NODE.get_logger().info("【初始化完成】启动相机流...")
    
    checkpoint_path = os.path.join(BASELINE_DIR, 'logs', 'log_rs', 'checkpoint-rs.tar')
    fallback_checkpoint_path = os.path.join(BASELINE_DIR, 'logs', 'log_rs', 'checkpoint.tar')
    if not os.path.exists(checkpoint_path) and os.path.exists(fallback_checkpoint_path):
        checkpoint_path = fallback_checkpoint_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "GraspNet checkpoint not found. Tried:\n"
            f"  - {os.path.join(BASELINE_DIR, 'logs', 'log_rs', 'checkpoint-rs.tar')}\n"
            f"  - {fallback_checkpoint_path}"
        )
    net, device = get_net(checkpoint_path)
    
    SAM2_CHECKPOINT = os.path.join(SAM2_DIR, "checkpoints/sam2.1/sam2.1_hiera_small.pt")
    SAM2_CFG = "sam2.1/sam2.1_hiera_s.yaml"
    logging.info("Loading SAM2 model...")
    sam2_predictor = build_sam2_camera_predictor(SAM2_CFG, SAM2_CHECKPOINT)
    sam2_predictor.fill_hole_area = 0
    logging.info("SAM2 model loaded")
    
    rs_pipeline, rs_align, rs_camera_info = start_realsense()
    zed, K_scaled, baseline, camera_info_zed = start_zed()
    image_left = sl.Mat()
    image_right = sl.Mat()
    zed_sam2_predictor = build_sam2_camera_predictor(SAM2_CFG, SAM2_CHECKPOINT)
    zed_sam2_predictor.fill_hole_area = 0

    cv2.namedWindow("Realsense Viewer", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Realsense Viewer", mouse_callback)

    zed_ui = {
        "drawing": False,
        "ix": -1,
        "iy": -1,
        "fx": -1,
        "fy": -1,
        "pending_bbox": None,
        "pending_label": None,
        "pending_point": None,
        "selection_source": None,
        "need_reset": False,
        "num_targets": 0,
        "current_masks": {},
        "all_prompts": [],
        "tracking_img": None,
        "left_rgb": None,
        "right_rgb": None,
        "grasp_request_pending": False,
        "segmentation_ready_time": 0.0,
        "grasp_generation_running": False,
        "observe_lowered_once": False,
    }
    selection_prompt_running = False

    def maybe_start_zed_grasp_generation():
        if STATE != "IDLE":
            return
        if zed_ui["left_rgb"] is None or zed_ui["right_rgb"] is None:
            logging.warning("ZED grasp viewer is not ready.")
            return
        if zed_ui["grasp_generation_running"]:
            return

        object_mask = select_grasp_target_mask(zed_ui["current_masks"], zed_ui["all_prompts"])
        if object_mask is None or not np.any(object_mask):
            logging.info("No valid ZED target mask found, using default workspace heuristic.")
            object_mask = None

        zed_ui["grasp_generation_running"] = True
        logging.info("Processing ZED frame for GraspNet...")

        def process_grasp_zed(l_rgb, r_rgb, cam_info, dev, c_mask):
            try:
                def queue_low_observation_retry(reason):
                    if ROS_NODE is None or ROS_NODE.rtde_c is None or ROS_NODE.dynamic_start_pose is None:
                        return False
                    if zed_ui["observe_lowered_once"]:
                        return False
                    try:
                        retry_pose = list(ROS_NODE.dynamic_start_pose)
                        retry_pose[2] -= WORKSPACE_OBSERVE_RETRY_STEP
                        logging.warning(
                            f"{reason} Lowering observation pose by {WORKSPACE_OBSERVE_RETRY_STEP * 100:.0f}cm "
                            f"(target z={retry_pose[2]:.3f}) and retrying GraspNet once."
                        )
                        ROS_NODE.rtde_c.moveL(
                            retry_pose,
                            WORKSPACE_OBSERVE_SLOW_MOVE_SPEED,
                            WORKSPACE_OBSERVE_SLOW_MOVE_ACCEL,
                        )
                        zed_ui["observe_lowered_once"] = True
                        zed_ui["grasp_request_pending"] = True
                        zed_ui["segmentation_ready_time"] = time.time()
                        return True
                    except Exception as e:
                        logging.warning(f"Failed to move to lower retry pose: {e}")
                        return False

                depth_mm, disp = compute_ffs_depth_and_disp(l_rgb, r_rgb, K_scaled, baseline)
                end_points, cloud_o3d, workspace_mask = process_frame(l_rgb, depth_mm, cam_info, dev, sam_mask=c_mask)
                if end_points is None:
                    if not queue_low_observation_retry("No points found in ZED workspace mask."):
                        logging.warning("No points found in ZED workspace mask.")
                    return

                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.float32):
                        end_points = net(end_points)
                        grasp_preds = pred_decode(end_points)

                gg_array = grasp_preds[0].detach().cpu().numpy()
                gg = GraspGroup(gg_array)
                raw_count = len(gg)
                gg_nms = gg.nms().sort_by_score()
                nms_count = len(gg_nms)

                try:
                    tf_msg = ROS_NODE.tf_buffer.lookup_transform('base_link', 'zed_inhand_camera_frame_optical', rclpy.time.Time())
                    tf_rot = R.from_quat([
                        tf_msg.transform.rotation.x,
                        tf_msg.transform.rotation.y,
                        tf_msg.transform.rotation.z,
                        tf_msg.transform.rotation.w
                    ])
                    camera_rot_mat = tf_rot.as_matrix()
                except Exception as e:
                    logging.warning(f"Failed to get camera orientation TF, fallback to identity for grasp filtering: {e}")
                    camera_rot_mat = np.eye(3, dtype=np.float32)

                gg_filtered = filter_grasps_by_camera_direction(gg_nms, camera_rot_mat).sort_by_score()
                orient_count = len(gg_filtered)
                selected_count = len(gg_filtered)
                logging.info(
                    "Grasp filtering (demo-style + camera-axis): "
                    f"raw={raw_count}, nms={nms_count}, "
                    f"orient={orient_count}, selected={selected_count}"
                )

                save_zed_grasp_capture_bundle(
                    left_rgb=l_rgb,
                    tracking_bgr=l_rgb[:, :, ::-1].copy(),
                    depth_mm=depth_mm,
                    disp=disp,
                    camera=cam_info,
                    workspace_mask=workspace_mask,
                    gg_before=gg_nms,
                    gg_after=gg_filtered,
                    note="robot2human",
                )

                gg_vis_array = gg_filtered.grasp_group_array if len(gg_filtered) > 0 else np.empty((0, 17), dtype=np.float32)
                show_open3d_process(
                    np.asarray(cloud_o3d.points),
                    np.asarray(cloud_o3d.colors),
                    gg_vis_array
                )
                p = None

                if len(gg_filtered) > 0 and ROS_NODE:
                    logging.info("Sending ZED-selected grasps to execution node.")
                    grasp_ok = ROS_NODE.execute_grasp(gg_filtered, viz_process=p)
                    if grasp_ok:
                        zed_ui["observe_lowered_once"] = False
                        zed_ui["need_reset"] = True
                        zed_ui["grasp_request_pending"] = False
                        logging.info("Grasp completed. Resetting ZED SAM2 targets.")
                elif len(gg_filtered) == 0:
                    if not queue_low_observation_retry("No valid grasp remains after NMS + camera-axis filtering."):
                        logging.warning("No valid grasp remains after NMS + camera-axis filtering.")
            finally:
                zed_ui["grasp_generation_running"] = False

        t = threading.Thread(target=process_grasp_zed, args=(
            zed_ui["left_rgb"].copy(),
            zed_ui["right_rgb"].copy(),
            camera_info_zed,
            device,
            object_mask.copy() if object_mask is not None else None,
        ), daemon=True)
        t.start()

    def prompt_selection_mode_after_workspace(default_mode=None):
        nonlocal selection_prompt_running
        try:
            print("\n[Object Selection] Workspace pose reached.")
            mode = (default_mode or "").strip().lower()
            if mode not in ("1", "2", "vlm", "v", "model", "llm", "manual", "m"):
                print("[Object Selection] Choose mode: 1) MLLM selection  2) Manual selection")
                mode = input("[Object Selection] Enter 1 or 2: ").strip().lower()

            if mode in ("1", "vlm", "v", "model", "llm"):
                print("[Object Selection] Prompt type: 1) Text  2) Voice")
                prompt_mode = input("[Object Selection] Enter 1 or 2: ").strip().lower()

                object_prompt = ""
                audio_prompt_path = None
                temp_audio_file = None

                try:
                    if prompt_mode in ("1", "text", "t", ""):
                        object_prompt = input("[Object Selection] Enter object prompt: ").strip()
                    elif prompt_mode in ("2", "voice", "audio", "v"):
                        temp_audio_file = tempfile.NamedTemporaryFile(prefix="robot2human_prompt_", suffix=".mp3", delete=False)
                        temp_audio_file.close()
                        if record_audio_prompt(temp_audio_file.name, duration_sec=voice_prompt_seconds):
                            audio_prompt_path = temp_audio_file.name
                    else:
                        logging.warning("Unknown prompt type, fallback to manual selection.")
                        return

                    if not object_prompt and audio_prompt_path is None:
                        logging.warning("No valid text/audio prompt collected, fallback to manual selection.")
                        return

                    tracking_img = zed_ui.get("tracking_img", None)
                    if tracking_img is None:
                        logging.warning("ZED image not ready for MLLM selection, fallback to manual selection.")
                        return

                    selection = call_doubao_object_bbox(
                        tracking_img.copy(),
                        user_prompt=object_prompt,
                        audio_prompt_path=audio_prompt_path,
                    )
                    if selection is None:
                        logging.warning("MLLM failed to locate object. Please use manual selection.")
                        return

                    zed_ui["num_targets"] = 0
                    zed_ui["current_masks"] = {}
                    zed_ui["all_prompts"] = []
                    zed_ui["pending_point"] = None
                    zed_ui["pending_bbox"] = selection["bbox"]
                    zed_ui["pending_label"] = selection["label"]
                    zed_ui["selection_source"] = "vlm"
                    zed_ui["need_reset"] = False
                    logging.info(f"MLLM selected object bbox: {selection['bbox']} ({selection['label']}). SAM2 will segment it automatically.")
                finally:
                    if temp_audio_file is not None and os.path.exists(temp_audio_file.name) and not DEBUG_MODE:
                        try:
                            os.remove(temp_audio_file.name)
                        except OSError:
                            pass
            else:
                zed_ui["selection_source"] = "manual"
                logging.info("Manual selection mode enabled. Use mouse in ZED Viewer as before.")
        finally:
            selection_prompt_running = False

    def zed_mouse_callback(event, x, y, flags, param):
        if zed_ui.get("selection_source") != "manual":
            if event == cv2.EVENT_LBUTTONDOWN:
                logging.info("Mouse selection is disabled. Please choose 2) Manual selection first.")
            zed_ui["drawing"] = False
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            zed_ui["drawing"] = True
            zed_ui["ix"], zed_ui["iy"] = x, y
            zed_ui["fx"], zed_ui["fy"] = x, y
        elif event == cv2.EVENT_MOUSEMOVE and zed_ui["drawing"]:
            zed_ui["fx"], zed_ui["fy"] = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            zed_ui["drawing"] = False
            zed_ui["fx"], zed_ui["fy"] = x, y
            dx = abs(zed_ui["fx"] - zed_ui["ix"])
            dy = abs(zed_ui["fy"] - zed_ui["iy"])
            if dx > 8 and dy > 8:
                zed_ui["pending_bbox"] = (
                    min(zed_ui["ix"], zed_ui["fx"]),
                    min(zed_ui["iy"], zed_ui["fy"]),
                    max(zed_ui["ix"], zed_ui["fx"]),
                    max(zed_ui["iy"], zed_ui["fy"]),
                )
                zed_ui["selection_source"] = "manual"
            else:
                zed_ui["pending_point"] = (x, y)
                zed_ui["selection_source"] = "manual"
    cv2.namedWindow("ZED Viewer", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("ZED Viewer", zed_mouse_callback)

    rs_latest = {"color_img": None, "depth_img": None}

    logging.info("Camera Started. UI Controls (focus on OpenCV window):")
    logging.info("  - w: Trigger SAM2 to detect Workpace")
    logging.info("  - Left-click drag / Left-click in ZED Viewer: only available in Manual selection mode")
    logging.info("  - Space: Process Depth (FFS, ZED in-hand), Generate Grasp and Execute")
    logging.info("  - r: Reset Selection")
    logging.info("  - q: Quit")
    logging.info("  - After moving to workspace, terminal asks: MLLM/manual")
    
    num_targets = 0 
    need_reset = False
    current_masks = {} 
    all_prompts = []
    
    initialized_dynamic_start = False
    handover_start_calc_running = False
    follow_calc_running = False
    
    # 刚启动时，自动触发工作台检测
    doubao_trigger = True
    
    try:
        while True:
            t0 = time.time()
            frames_rs = rs_pipeline.poll_for_frames()
            if frames_rs:
                aligned_frames_rs = rs_align.process(frames_rs)
                color_frame_rs = aligned_frames_rs.get_color_frame()
                depth_frame_rs = aligned_frames_rs.get_depth_frame()
                if color_frame_rs and depth_frame_rs:
                    rs_color_img = np.asanyarray(color_frame_rs.get_data())
                    rs_depth_img = np.asanyarray(depth_frame_rs.get_data())
                    color_img_bgr = cv2.cvtColor(rs_color_img, cv2.COLOR_RGB2BGR)
                    tracking_img = color_img_bgr.copy()
                    color_img = rs_color_img.copy()
                    display = tracking_img.copy()
                    rs_latest["color_img"] = rs_color_img.copy()
                    rs_latest["depth_img"] = rs_depth_img.copy()
                else:
                    color_img_bgr = None
            else:
                color_img_bgr = None

            if color_img_bgr is not None:
                if request_selection_prompt and (not selection_prompt_running) and STATE == "IDLE":
                    selection_prompt_running = True
                    request_selection_prompt = False
                    t_select = threading.Thread(target=prompt_selection_mode_after_workspace, daemon=True)
                    t_select.start()

                # --- MediaPipe Right-Hand Handover (RealSense eye-to-hand) ---
                if STATE in ("WAITING_HANDOVER_START_POSE", "WAITING_DELIVERY", "FOLLOWING_HAND"):
                    results = hands.process(cv2.cvtColor(color_img_bgr, cv2.COLOR_BGR2RGB))
                    right_hand_landmarks, right_hand_score = select_right_hand(results)

                    if right_hand_landmarks:
                        mp_drawing.draw_landmarks(display, right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        h, w, _ = display.shape
                        palm_center_lm = right_hand_landmarks.landmark[9]
                        cx, cy = int(palm_center_lm.x * w), int(palm_center_lm.y * h)
                        cv2.circle(display, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                        cv2.putText(display, f"Physical Right hand {right_hand_score:.2f}", (cx-70, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                        def calc_rs_and_update_handover_start(d_img, cam_info, pt_x, pt_y):
                            nonlocal handover_start_calc_running
                            try:
                                hand_base_pos = calc_hand_base_pos_from_rs_depth(d_img, cam_info, pt_x, pt_y, max_depth=1.5, log_prefix="right hand handover start")
                                if hand_base_pos is not None:
                                    ROS_NODE.update_right_hand_base_pos(hand_base_pos)
                            finally:
                                handover_start_calc_running = False

                        def calc_rs_and_follow(d_img, cam_info, pt_x, pt_y):
                            nonlocal follow_calc_running
                            try:
                                hand_base_pos = calc_hand_base_pos_from_rs_depth(d_img, cam_info, pt_x, pt_y, max_depth=1.5, log_prefix="right hand follow")
                                if hand_base_pos is not None and STATE == "FOLLOWING_HAND":
                                    ROS_NODE.update_right_hand_base_pos(hand_base_pos)
                                    ROS_NODE.follow_right_hand(hand_base_pos)
                            finally:
                                follow_calc_running = False

                        if STATE == "WAITING_HANDOVER_START_POSE":
                            cv2.putText(display, "Aligning handover start to Right hand...", (cx-90, cy-35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            if not handover_start_calc_running:
                                handover_start_calc_running = True
                                t = threading.Thread(target=calc_rs_and_update_handover_start, args=(rs_depth_img.copy(), rs_camera_info, cx, cy), daemon=True)
                                t.start()
                        elif STATE == "WAITING_DELIVERY":
                            cv2.putText(display, "Starting to follow Right hand...", (cx-90, cy-35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            STATE = "FOLLOWING_HAND"
                        elif STATE == "FOLLOWING_HAND":
                            cv2.putText(display, "Following Right hand at 15cm. Fist to release.", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            if is_fist(right_hand_landmarks):
                                cv2.putText(display, "Right fist Detected! Delivering...", (cx-80, cy-35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                save_mediapipe_intent_frame(
                                    display.copy(),
                                    prefix="robot2human_right_fist_intent",
                                    detected_text="Detected right fist intent",
                                )
                                STATE = "DELIVERING_CALCULATING"

                                def calc_rs_and_deliver(d_img, cam_info, pt_x, pt_y):
                                    global STATE
                                    hand_base_pos = calc_hand_base_pos_from_rs_depth(d_img, cam_info, pt_x, pt_y, max_depth=1.5, log_prefix="right hand delivery")
                                    if hand_base_pos is not None:
                                        ROS_NODE.update_right_hand_base_pos(hand_base_pos)
                                        STATE = "DELIVERING"
                                        ROS_NODE.deliver_to_hand(hand_base_pos)
                                    else:
                                        STATE = "FOLLOWING_HAND"

                                t = threading.Thread(target=calc_rs_and_deliver, args=(rs_depth_img.copy(), rs_camera_info, cx, cy), daemon=True)
                                t.start()
                            elif not follow_calc_running:
                                follow_calc_running = True
                                t = threading.Thread(target=calc_rs_and_follow, args=(rs_depth_img.copy(), rs_camera_info, cx, cy), daemon=True)
                                t.start()
                    else:
                        cv2.putText(display, "Waiting for MediaPipe Right hand...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                # --- SAM2 for RealSense (workspace/eye-to-hand stream) ---
                if need_reset:
                    if hasattr(sam2_predictor, 'condition_state') and 'point_inputs_per_obj' in sam2_predictor.condition_state:
                        sam2_predictor.reset_state()
                    num_targets = 0
                    need_reset = False
                    pending_bbox = None
                    pending_point = None
                    current_masks = {}
                    all_prompts = []
                    logging.info("Reset, select new targets.")

                if (pending_bbox is not None or pending_point is not None) and num_targets < 2:
                    target_id = num_targets + 1
                    if pending_bbox is not None:
                        all_prompts.append({'id': target_id, 'bbox': pending_bbox})
                        pending_bbox = None
                    elif pending_point is not None:
                        all_prompts.append({'id': target_id, 'point': pending_point})
                        pending_point = None
                    num_targets += 1

                    if hasattr(sam2_predictor, 'condition_state') and 'point_inputs_per_obj' in sam2_predictor.condition_state:
                        sam2_predictor.reset_state()

                    sam2_predictor.load_first_frame(tracking_img)
                    for p in all_prompts:
                        tid = p['id']
                        if 'bbox' in p:
                            x1, y1, x2, y2 = p['bbox']
                            bbox_arr = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                            sam2_predictor.add_new_prompt(frame_idx=0, obj_id=tid, bbox=bbox_arr)
                        elif 'point' in p:
                            px, py = p['point']
                            pts_arr = np.array([[px, py]], dtype=np.float32)
                            lbl_arr = np.array([1], dtype=np.int32)
                            sam2_predictor.add_new_prompt(frame_idx=0, obj_id=tid, points=pts_arr, labels=lbl_arr)

                if num_targets > 0:
                    out_obj_ids, out_mask_logits = sam2_predictor.track(tracking_img)
                    current_masks = {}
                    for i in range(len(out_obj_ids)):
                        obj_id = out_obj_ids[i]
                        current_masks[obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()

                for obj_id, mask in current_masks.items():
                    if mask is not None:
                        color = MASK_COLORS_BGR.get(obj_id, [0, 255, 0])
                        overlay = display.copy()
                        overlay[mask > 0] = color
                        display = cv2.addWeighted(display, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

                if workspace_bbox is not None:
                    cv2.polylines(display, [workspace_bbox], isClosed=True, color=(0, 165, 255), thickness=2, lineType=cv2.LINE_AA)
                    cv2.putText(display, "Workspace", (workspace_bbox[0][0], max(20, workspace_bbox[0][1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                if drawing and ix >= 0:
                    cv2.rectangle(display, (ix, iy), (fx_mouse, fy_mouse), (255, 200, 0), 2)

                t1 = time.time()
                fps = 1.0 / max(1e-6, (t1 - t0))
                cv2.putText(display, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                status = f"Targets: {num_targets} | STATE: {STATE} | w=Workspace | SPACE=ZED Grasp | r=reset | q=quit"
                cv2.putText(display, status, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Realsense Viewer", display)

            # --- ZED in-hand stream (for grasp selection) ---
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image_left, sl.VIEW.LEFT)
                zed.retrieve_image(image_right, sl.VIEW.RIGHT)

                raw_left_bgr = image_left.get_data()[:, :, :3]
                raw_right_bgr = image_right.get_data()[:, :, :3]
                left_bgr = cv2.resize(raw_left_bgr, (IMG_WIDTH, IMG_HEIGHT))
                right_bgr = cv2.resize(raw_right_bgr, (IMG_WIDTH, IMG_HEIGHT))
                left_rgb = left_bgr[:, :, ::-1].copy()
                right_rgb = right_bgr[:, :, ::-1].copy()

                zed_tracking = left_bgr.copy()
                zed_display = zed_tracking.copy()
                zed_ui["tracking_img"] = zed_tracking.copy()
                zed_ui["left_rgb"] = left_rgb.copy()
                zed_ui["right_rgb"] = right_rgb.copy()

                if zed_ui["need_reset"]:
                    if hasattr(zed_sam2_predictor, 'condition_state') and 'point_inputs_per_obj' in zed_sam2_predictor.condition_state:
                        zed_sam2_predictor.reset_state()
                    zed_ui["num_targets"] = 0
                    zed_ui["need_reset"] = False
                    zed_ui["pending_bbox"] = None
                    zed_ui["pending_label"] = None
                    zed_ui["pending_point"] = None
                    zed_ui["current_masks"] = {}
                    zed_ui["all_prompts"] = []
                    zed_ui["grasp_request_pending"] = False
                    zed_ui["segmentation_ready_time"] = 0.0
                    zed_ui["observe_lowered_once"] = False

                if (zed_ui["pending_bbox"] is not None or zed_ui["pending_point"] is not None) and zed_ui["num_targets"] < 2:
                    target_id = zed_ui["num_targets"] + 1
                    if zed_ui["pending_bbox"] is not None:
                        zed_ui["all_prompts"].append({
                            'id': target_id,
                            'bbox': zed_ui["pending_bbox"],
                            'label': zed_ui["pending_label"] or "object",
                        })
                        zed_ui["pending_bbox"] = None
                        zed_ui["pending_label"] = None
                    elif zed_ui["pending_point"] is not None:
                        zed_ui["all_prompts"].append({'id': target_id, 'point': zed_ui["pending_point"]})
                        zed_ui["pending_point"] = None
                    zed_ui["num_targets"] += 1
                    zed_ui["grasp_request_pending"] = True
                    zed_ui["segmentation_ready_time"] = time.time()
                    zed_ui["observe_lowered_once"] = False

                    if hasattr(zed_sam2_predictor, 'condition_state') and 'point_inputs_per_obj' in zed_sam2_predictor.condition_state:
                        zed_sam2_predictor.reset_state()
                    zed_sam2_predictor.load_first_frame(zed_tracking)
                    for prompt in zed_ui["all_prompts"]:
                        pid = prompt["id"]
                        if "bbox" in prompt:
                            x1, y1, x2, y2 = prompt["bbox"]
                            bbox_arr = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                            zed_sam2_predictor.add_new_prompt(frame_idx=0, obj_id=pid, bbox=bbox_arr)
                        elif "point" in prompt:
                            px, py = prompt["point"]
                            pts_arr = np.array([[px, py]], dtype=np.float32)
                            lbl_arr = np.array([1], dtype=np.int32)
                            zed_sam2_predictor.add_new_prompt(frame_idx=0, obj_id=pid, points=pts_arr, labels=lbl_arr)

                if zed_ui["num_targets"] > 0:
                    out_obj_ids, out_mask_logits = zed_sam2_predictor.track(zed_tracking)
                    zed_ui["current_masks"] = {}
                    for i in range(len(out_obj_ids)):
                        obj_id = out_obj_ids[i]
                        zed_ui["current_masks"][obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()

                for obj_id, mask in zed_ui["current_masks"].items():
                    if mask is not None:
                        color = MASK_COLORS_BGR.get(obj_id, [0, 255, 0])
                        overlay = zed_display.copy()
                        overlay[mask > 0] = color
                        zed_display = cv2.addWeighted(zed_display, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(zed_display, contours, -1, (0, 255, 0), 2)
                        label = next((p.get("label") for p in zed_ui["all_prompts"] if p.get("id") == obj_id), str(obj_id))
                        if contours:
                            x, y, _, _ = cv2.boundingRect(contours[0])
                            cv2.putText(zed_display, str(label), (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if zed_ui["drawing"] and zed_ui["ix"] >= 0:
                    cv2.rectangle(zed_display, (zed_ui["ix"], zed_ui["iy"]), (zed_ui["fx"], zed_ui["fy"]), (255, 200, 0), 2)

                zed_status = "ZED Viewer | SPACE=Grasp | r=reset | q=quit"
                if zed_ui["grasp_request_pending"]:
                    wait_left = max(0.0, SAM2_SETTLE_SECONDS - (time.time() - zed_ui["segmentation_ready_time"]))
                    zed_status = f"ZED Viewer | waiting {wait_left:.1f}s -> Grasp | r=reset | q=quit"
                if STATE != "IDLE":
                    zed_status = f"ZED Viewer | STATE={STATE} | waiting for handover to finish"
                cv2.putText(zed_display, zed_status, (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("ZED Viewer", zed_display)

                if zed_ui["grasp_request_pending"] and zed_ui["num_targets"] > 0 and not zed_ui["grasp_generation_running"]:
                    if time.time() - zed_ui["segmentation_ready_time"] >= SAM2_SETTLE_SECONDS:
                        zed_ui["grasp_request_pending"] = False
                        maybe_start_zed_grasp_generation()
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32 and STATE == "IDLE": 
                if zed_ui["num_targets"] <= 0:
                    logging.warning("No ZED SAM2 targets available yet.")
                else:
                    zed_ui["grasp_request_pending"] = True
                    if zed_ui["segmentation_ready_time"] <= 0:
                        zed_ui["segmentation_ready_time"] = time.time()
                    logging.info("Queued grasp generation. Waiting 1.0s for SAM2 to stabilize.")
                
            elif key == ord('w'):
                if num_targets > 0:
                    logging.info("Setting current SAM2 targets as workspace...")
                    if rs_latest["depth_img"] is None:
                        logging.warning("RealSense workspace stream not ready.")
                        continue
                    rs_h, rs_w = rs_latest["depth_img"].shape
                    combined_mask = np.zeros((rs_h, rs_w), dtype=bool)
                    for mask in current_masks.values():
                        if mask is not None:
                            combined_mask |= (mask > 0)
                    if np.any(combined_mask):
                        contour, _ = cv2.findContours(combined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if len(contour) > 0:
                            c = max(contour, key=cv2.contourArea)
                            rect = cv2.minAreaRect(c)
                            box = cv2.boxPoints(rect)
                            box = np.int32(box)  # 4 corners of the rotated bounding box
                            workspace_bbox = box
                            
                            cx, cy = int(rect[0][0]), int(rect[0][1])
                            
                            # 获取最长边作为X轴平行的参考
                            edge1_len = np.linalg.norm(box[0] - box[1])
                            edge2_len = np.linalg.norm(box[1] - box[2])
                            if edge1_len > edge2_len:
                                edge_pt1, edge_pt2 = box[0], box[1]
                            else:
                                edge_pt1, edge_pt2 = box[1], box[2]
                            
                            def calc_rs_and_move_ws(depth_img, camera, pt_x, pt_y, e_pt1, e_pt2):
                                nonlocal selection_prompt_running
                                center_base = calc_hand_base_pos_from_rs_depth(depth_img, camera, pt_x, pt_y, max_depth=2.0, log_prefix="workspace center")
                                pt1_base = calc_hand_base_pos_from_rs_depth(depth_img, camera, int(e_pt1[0]), int(e_pt1[1]), max_depth=2.0, log_prefix="workspace edge1")
                                pt2_base = calc_hand_base_pos_from_rs_depth(depth_img, camera, int(e_pt2[0]), int(e_pt2[1]), max_depth=2.0, log_prefix="workspace edge2")
                                if center_base is None or pt1_base is None or pt2_base is None:
                                    logging.warning("Workspace selection failed: depth invalid.")
                                    return

                                V = pt2_base - pt1_base
                                V[2] = 0
                                v_norm = np.linalg.norm(V)
                                if v_norm < 1e-5:
                                    V = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                                    v_norm = 1.0

                                X_axis = V / v_norm
                                Z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
                                current_tcp = ROS_NODE.rtde_r.getActualTCPPose()
                                curr_rvec = np.array([current_tcp[3], current_tcp[4], current_tcp[5]])
                                curr_R, _ = cv2.Rodrigues(curr_rvec)
                                curr_X_axis = curr_R[:, 0]
                                if np.dot(X_axis, curr_X_axis) < 0:
                                    X_axis = -X_axis

                                Y_axis = np.cross(Z_axis, X_axis)
                                R_target = np.column_stack((X_axis, Y_axis, Z_axis))
                                rvec, _ = cv2.Rodrigues(R_target)
                                rx, ry, rz = rvec.flatten()

                                center_xy = np.array(center_base[:2], dtype=np.float64)
                                center_xy_norm = np.linalg.norm(center_xy)
                                observe_xy = center_xy.copy()
                                if center_xy_norm >= 1e-5:
                                    observe_xy = center_xy - (center_xy / center_xy_norm) * WORKSPACE_OBSERVE_BASE_OFFSET
                                else:
                                    logging.warning("Workspace center too close to base origin; skip 10cm base offset for observation pose.")

                                dyn_start_pose = [
                                    -observe_xy[0], -observe_xy[1], center_base[2] + WORKSPACE_OBSERVE_Z_OFFSET_HIGH, rx, ry, rz
                                ]
                                ROS_NODE.dynamic_start_pose = dyn_start_pose
                                zed_ui["observe_lowered_once"] = False
                                ROS_NODE.get_logger().info(
                                    f"Moving to dynamic start pose (30cm above workspace, 10cm toward base, tool0 X parallel to bbox edge): {dyn_start_pose}"
                                )
                                ROS_NODE.rtde_c.moveL(dyn_start_pose, 0.1, 0.1)

                                if not selection_prompt_running:
                                    selection_prompt_running = True
                                    t_select = threading.Thread(target=prompt_selection_mode_after_workspace, daemon=True)
                                    t_select.start()

                            t_start = threading.Thread(target=calc_rs_and_move_ws, args=(rs_latest["depth_img"].copy(), rs_camera_info, cx, cy, edge_pt1, edge_pt2), daemon=True)
                            t_start.start()
                        
                            # 清理现有 targets 留空进行抓取选择
                            need_reset = True
                else:
                    logging.warning("No SAM2 tracking target to set as workspace. Please specify target first.")
            elif key == ord('r'):
                need_reset = True
                zed_ui["need_reset"] = True
                zed_ui["grasp_request_pending"] = False
                zed_ui["selection_source"] = None
                zed_ui["drawing"] = False
                if STATE == "IDLE" and not selection_prompt_running:
                    selection_prompt_running = True
                    t_select = threading.Thread(target=prompt_selection_mode_after_workspace, daemon=True)
                    t_select.start()
            elif key == ord('q'):
                break
                
    finally:
        rs_pipeline.stop()
        zed.close()
        if ROS_NODE and ROS_NODE.rtde_c: ROS_NODE.rtde_c.stopScript()
        rclpy.shutdown()
        close_open3d_viewer()
        cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="Robot-to-human handover pipeline")
    parser.add_argument("--debug", action="store_true", help="Save debug artifacts like captured_data, llm_output, mediapipe_output")
    parser.add_argument("--voice-prompt-seconds", type=float, default=VOICE_MAX_RECORD_SECONDS, help="Maximum microphone recording duration for voice prompt mode")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(debug=args.debug, voice_prompt_seconds=args.voice_prompt_seconds)
