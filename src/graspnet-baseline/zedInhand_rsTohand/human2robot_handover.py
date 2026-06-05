import numpy as np
import cv2
import torch
import open3d as o3d
import os
import sys
import argparse
import datetime
import scipy.io as scio
import time
import logging
import base64
import re
import json
import threading
from pathlib import Path

# ====== Volcengine SDK ======
try:
    from volcenginesdkarkruntime import Ark
except ImportError:
    Ark = None

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
import pyrealsense2 as rs

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ===== GPU config (SAM2 requires bfloat16) =====
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# === 路径设置 ===
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = ROOT_DIR if os.path.isdir(os.path.join(ROOT_DIR, "models")) else os.path.dirname(ROOT_DIR)
SRC_DIR = os.path.dirname(BASELINE_DIR)

sys.path.append(BASELINE_DIR)
sys.path.append(os.path.join(BASELINE_DIR, 'models'))
sys.path.append(os.path.join(BASELINE_DIR, 'dataset'))
sys.path.append(os.path.join(BASELINE_DIR, 'utils'))

from utils.collision_detector import ModelFreeCollisionDetector
from models.graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup

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

from data_utils import CameraInfo, create_point_cloud_from_depth_image

# ===== MLLM Config (大模型) =====
ARK_API_KEY = os.getenv("ARK_API_KEY")
ARK_BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
DOUBAO_MODEL_NAME = os.getenv("DOUBAO_MODEL_NAME", "doubao-seed-2-0-mini-260215")
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "minimal")

if Ark is None:
    raise ImportError("volcenginesdkarkruntime is required for Doubao MLLM")
client = Ark(
    base_url=ARK_BASE_URL,
    api_key=ARK_API_KEY,
)

# ===== MediaPipe Config ======
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
PHYSICAL_RIGHT_HAND_MP_LABEL = "Left"
PHYSICAL_LEFT_HAND_MP_LABEL = "Right"
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
COLLISION_VOXEL_SIZE = 0.005
COLLISION_APPROACH_DIST = 0.02
COLLISION_THRESH = 0.03
ENABLE_COLLISION_FALLBACK = True
BASE_TO_CONTROLLER_ROT = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
GRASP_BACKOFF_DIST = 0.10
GRASP_Z_OFFSET_DOWN = 0.0

# === 全局变量 ===
vlm_running = False
doubao_trigger = False
workspace_bbox = None
pending_targets = []
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
        self.dynamic_workspace_center_pose = None
        self.right_hand_base_pos = None
        self.right_hand_seen_time = 0.0
        self.right_hand_lock = threading.Lock()
        self.handover_follow_pose = None
        self.handover_follow_z_offset = None
        self.zed_selection_mode_callback = None
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
        while time.time() - start_time < timeout:
            hand_base_pos = self.get_recent_right_hand_base_pos(max_age=HANDOVER_RIGHT_HAND_MAX_AGE)
            if hand_base_pos is not None:
                return hand_base_pos
            time.sleep(0.05)
        return None

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
            self.get_logger().warning("Right hand direction unavailable; skip moving to default handover pose.")
            return None

        hand_controller_xy = np.array([-hand_base_pos[0], -hand_base_pos[1]], dtype=float)
        hand_xy_norm = np.linalg.norm(hand_controller_xy)
        if hand_xy_norm < 1e-4:
            self.get_logger().warning("Right hand XY direction is too small; waiting for a stable right hand direction.")
            return None

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

    def wait_for_handover_joint_q(self, timeout=HANDOVER_RIGHT_HAND_TIMEOUT, retry_sleep=0.1):
        while True:
            hand_base_pos = self.wait_for_right_hand_base_pos(timeout=timeout)
            if hand_base_pos is None:
                self.get_logger().info("Right hand not detected yet, keep waiting before moving to handover observation pose...")
                continue

            handover_joint_q = self.build_handover_joint_q(hand_base_pos)
            if handover_joint_q is not None:
                return handover_joint_q
            time.sleep(retry_sleep)

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

    def execute_grasp(self, gg, viz_process=None, require_confirmation=True):
        if not self.rtde_c: 
            self.get_logger().error("RTDE not connected.")
            return

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

            tf_msg = self.tf_buffer.lookup_transform('base_link', 'zed_inhand_camera_frame_optical', rclpy.time.Time())
            tf_trans = np.array([tf_msg.transform.translation.x, tf_msg.transform.translation.y, tf_msg.transform.translation.z])
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
            R_curr, _ = cv2.Rodrigues(np.array(current_tcp_pose[3:]))

            def rotation_distance(r_candidate):
                r_delta = R_curr.T @ r_candidate
                trace_val = np.clip((np.trace(r_delta) - 1.0) * 0.5, -1.0, 1.0)
                return math.acos(trace_val)

            if rotation_distance(r_target_flip) < rotation_distance(r_target_primary):
                R_target = r_target_flip
            else:
                R_target = r_target_primary

            rvec_target, _ = cv2.Rodrigues(R_target)
            rx, ry, rz = rvec_target.flatten()

            controller_x = -base_pos[0]
            controller_y = -base_pos[1]
            controller_z = base_pos[2] - GRASP_Z_OFFSET_DOWN

            target_pose = [controller_x, controller_y, controller_z, rx, ry, rz]
            approach_pose = list(target_pose)
            Z_tool_ctrl = R_target[:, 2] 
            approach_pose[0] -= GRASP_BACKOFF_DIST * Z_tool_ctrl[0]
            approach_pose[1] -= GRASP_BACKOFF_DIST * Z_tool_ctrl[1]
            approach_pose[2] -= GRASP_BACKOFF_DIST * Z_tool_ctrl[2]
            
            self.get_logger().info(f"Target pose (Controller): {target_pose}")

            self.send_gripper_command(0.085)
            
            if require_confirmation:
                ans = input("\nReady to grasp? Press 'y' to continue, 'n' to cancel: ")
                if ans.lower() != 'y':
                    close_viz_process()
                    self.get_logger().info("Grasp cancelled.")
                    return
                close_viz_process()

            self.get_logger().info("Moving to approach pose...")
            self.rtde_c.moveL(approach_pose, 0.05, 0.05)
            self.get_logger().info("Moving to target pose...")
            self.rtde_c.moveL(target_pose, 0.05, 0.05)
            self.get_logger().info("Closing gripper...")
            self.send_gripper_command(0.0)
            
            self.get_logger().info("Moving back to approach pose...")
            self.rtde_c.moveL(approach_pose, 0.05, 0.05)
            
            try:
                current_q = self.rtde_r.getActualQ()
                downward_q = [current_q[0], -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0.0]
                self.get_logger().info(f"Returning to downward gripper joint shape: {downward_q}")
                self.rtde_c.moveJ(downward_q, 0.15, 0.15)
            except Exception as e:
                self.get_logger().error(f"Failed to rotate back to downward gripper pose: {e}")

            workspace_place_pose = self.dynamic_start_pose if self.dynamic_start_pose else self.dynamic_workspace_center_pose
            if workspace_place_pose:
                workspace_pose_30cm = list(workspace_place_pose)
                workspace_pose_10cm = list(workspace_place_pose)
                workspace_pose_10cm[2] -= 0.20

                self.get_logger().info("Moving to workspace pose 30cm above the table...")
                self.rtde_c.moveL(workspace_pose_30cm, 0.15, 0.15)
                self.get_logger().info("Lowering to workspace pose 10cm above the table...")
                self.rtde_c.moveL(workspace_pose_10cm, 0.10, 0.10)
                self.get_logger().info("Releasing object on workspace...")
                self.send_gripper_command(0.085)
                time.sleep(0.3)
                self.get_logger().info("Returning to workspace pose 30cm above the table...")
                self.rtde_c.moveL(workspace_pose_30cm, 0.15, 0.15)
            else:
                self.get_logger().warning("Workspace center pose unavailable, keeping current safe pose.")
                self.send_gripper_command(0.085)

            global STATE
            next_mode = None
            if callable(self.zed_selection_mode_callback):
                next_mode = self.zed_selection_mode_callback()
            if next_mode in ("mllm", "manual"):
                self.get_logger().info(f"ZED hand/object selection mode for the next round: {next_mode}")
            else:
                self.get_logger().warning("Next-round ZED selection mode was not updated; keeping previous mode.")
            STATE = "WAITING_HANDOVER_START_POSE"
            self.get_logger().info("Re-entering hand recognition for the next handover cycle...")
            handover_joint_q = self.wait_for_handover_joint_q()
            self.get_logger().info("Moving to Handover Observation Position...")
            self.rtde_c.moveJ(handover_joint_q, 0.5, 0.5)
            self.handover_follow_pose = list(self.rtde_r.getActualTCPPose())
            self.handover_follow_z_offset = None
            self.get_logger().info("Standing by at initial handover position.")
            STATE = "FOLLOWING_HAND"
                
        except Exception as e:
            self.get_logger().error(f"Grasp execution failed: {e}")

    def deliver_to_hand(self, hand_base_pos):
        # Called when MediaPipe finds the hand and user triggers delivery
        try:
            current_tcp_pose = self.rtde_r.getActualTCPPose()
            rx, ry, rz = current_tcp_pose[3], current_tcp_pose[4], current_tcp_pose[5]
            
            DELIVER_X_OFFSET = 0.02
            
            c_x = -hand_base_pos[0] + DELIVER_X_OFFSET
            c_y = -hand_base_pos[1]
            c_z = hand_base_pos[2]

            deliver_target_pose = [c_x, c_y, c_z, rx, ry, rz]

            self.get_logger().info(f"Delivering to Hand at Controller Pose: {deliver_target_pose}")
            self.rtde_c.moveL(deliver_target_pose, 0.05, 0.05)
            
            self.send_gripper_command(0.085)
            self.get_logger().info("Object Released to Human.")
            time.sleep(0.5)
            
            try:
                current_q = self.rtde_r.getActualQ()
                initial_q = [current_q[0], -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0.0]
                self.get_logger().info(f"Returning to initial joint shape before workspace start: {initial_q}")
                self.rtde_c.moveJ(initial_q, 0.15, 0.15)
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


# --- SAM2 Global State & Mouse Callback ---
drawing = False
ix, iy, fx_mouse, fy_mouse = -1, -1, -1, -1
pending_bbox = None
pending_point = None
zed_drawing = False
zed_ix, zed_iy, zed_fx_mouse, zed_fy_mouse = -1, -1, -1, -1
zed_pending_bbox = None
zed_pending_point = None

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


def zed_mouse_callback(event, x, y, flags, param):
    global zed_drawing, zed_ix, zed_iy, zed_fx_mouse, zed_fy_mouse, zed_pending_bbox, zed_pending_point

    if event == cv2.EVENT_LBUTTONDOWN:
        zed_drawing = True
        zed_ix, zed_iy = x, y
        zed_fx_mouse, zed_fy_mouse = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if zed_drawing:
            zed_fx_mouse, zed_fy_mouse = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        zed_drawing = False
        zed_fx_mouse, zed_fy_mouse = x, y
        dx = abs(zed_fx_mouse - zed_ix)
        dy = abs(zed_fy_mouse - zed_iy)
        if dx > 8 and dy > 8:
            x1, y1 = min(zed_ix, zed_fx_mouse), min(zed_iy, zed_fy_mouse)
            x2, y2 = max(zed_ix, zed_fx_mouse), max(zed_iy, zed_fy_mouse)
            zed_pending_bbox = (x1, y1, x2, y2)
        else:
            zed_pending_point = (x, y)


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

def select_left_hand(results):
    return select_physical_hand(results, PHYSICAL_LEFT_HAND_MP_LABEL)

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

def is_thumbs_up(hand_landmarks):
    lm = hand_landmarks.landmark
    wrist_lm = lm[0]
    thumb_tip = lm[4]
    thumb_ip = lm[3]
    thumb_mcp = lm[2]

    fingers_curled = all(
        landmark_dist2d(lm[tip_idx], wrist_lm) < landmark_dist2d(lm[mcp_idx], wrist_lm) * 1.25
        for tip_idx, mcp_idx in zip([8, 12, 16, 20], [5, 9, 13, 17])
    )
    thumb_extended = landmark_dist2d(thumb_tip, wrist_lm) > landmark_dist2d(thumb_mcp, wrist_lm) * 1.15
    thumb_points_up = thumb_tip.y < thumb_ip.y and thumb_tip.y < lm[5].y

    return fingers_curled and thumb_extended and thumb_points_up

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
    
    tf_msg = ROS_NODE.tf_buffer.lookup_transform('base_link', 'zed_left_camera_frame_optical', rclpy.time.Time())
    tf_rot = R.from_quat([tf_msg.transform.rotation.x, tf_msg.transform.rotation.y, tf_msg.transform.rotation.z, tf_msg.transform.rotation.w])
    tf_trans = np.array([tf_msg.transform.translation.x, tf_msg.transform.translation.y, tf_msg.transform.translation.z])
    
    return tf_rot.apply(point_3d_cam) + tf_trans


def compute_ffs_depth_map(l_rgb, r_rgb, K_s, bl):
    depth_mm, _ = compute_ffs_depth_and_disp(l_rgb, r_rgb, K_s, bl)
    return depth_mm


def compute_ffs_depth_and_disp(l_rgb, r_rgb, K_s, bl):
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
    return (depth_m * 1000.0).astype(np.uint16), disp


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


def save_zed_grasp_capture_bundle(left_rgb, tracking_bgr, depth_mm, disp, camera, workspace_mask, gg_before, gg_after, note="human2robot"):
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


def calc_point_base_from_rs_depth(depth_img, camera, pt_x, pt_y, max_depth=1.5, log_prefix="hand"):
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
    depth_scale = getattr(camera, "factor_depth", getattr(camera, "scale", None))
    if depth_scale is None or depth_scale <= 0:
        logging.error(f"Camera depth scale is invalid for {log_prefix}.")
        return None
    depth_m = depth_raw / depth_scale
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


def calc_hand_base_pos_from_rs_depth(depth_img, camera, pt_x, pt_y, max_depth=1.5, log_prefix="hand"):
    return calc_point_base_from_rs_depth(depth_img, camera, pt_x, pt_y, max_depth=max_depth, log_prefix=log_prefix)




def encode_image_to_base64(image_bgr):
    _, buffer = cv2.imencode(".jpg", image_bgr)
    return base64.b64encode(buffer).decode("utf-8")

def build_hand_object_prompt():
    return (
        "Instructions: \n"
        "1. Detect the human hand in the image. If no hand is present, reply exactly: 'No hand detected'.\n"
        "2. If a hand is present, detect both the 'hand' and the 'object' being held or interactived with by that hand.\n"
        "3. Provide their locations in JSON format as follows: \n"
        "   {\"hand\": [ymin, xmin, ymax, xmax], \"object\": [ymin, xmin, ymax, xmax], \"object_name\": \"name\"}\n"
        "4. object_name should be a short noun phrase for the detected object.\n"
        "   Coordinates should be normalized 0-1000. \n"
        "Return ONLY the JSON or 'No hand detected'."
    )


def save_hand_object_bbox_artifacts(image_bgr, targets, prefix="handover_vlm"):
    """Save parsed hand/object bbox JSON and visualization image under ./llm_output/<timestamp>/."""
    if not DEBUG_MODE:
        return None, None
    if image_bgr is None or not targets:
        return None, None

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cwd = os.getcwd()
    output_dir = os.path.join(cwd, "llm_output", ts)
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{prefix}.json")
    img_path = os.path.join(output_dir, f"{prefix}.jpg")

    payload = {
        "timestamp": ts,
        "image_size": {"width": int(image_bgr.shape[1]), "height": int(image_bgr.shape[0])},
        "targets": [],
    }
    vis = image_bgr.copy()

    for target in targets:
        tid = int(target.get("id", -1))
        label = str(target.get("label", f"id_{tid}"))
        x1, y1, x2, y2 = map(int, target.get("bbox", (0, 0, 0, 0)))
        payload["targets"].append({"id": tid, "label": label, "bbox_xyxy": [x1, y1, x2, y2]})

        color = (0, 255, 0) if tid == 2 else (255, 0, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    cv2.imwrite(img_path, vis)
    logging.info(f"Saved MLLM bbox JSON: {json_path}")
    logging.info(f"Saved MLLM bbox image: {img_path}")
    return json_path, img_path

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

def extract_doubao_text(response):
    if hasattr(response, "choices") and len(response.choices) > 0:
        return response.choices[0].message.content
    if hasattr(response, "output") and len(response.output) > 1:
        return response.output[1].content[0].text
    return str(response)

def parse_hand_object_response(content, image_bgr):
    if "No hand detected" in content:
        logging.warning("MLLM: Hand not found in frame.")
        return []

    try:
        json_str = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_str:
            logging.warning("MLLM response does not contain JSON.")
            return []

        data = json.loads(json_str.group())
        results = []
        h, w = image_bgr.shape[:2]

        if "hand" in data:
            vals = [float(v) for v in data["hand"]]
            ymin, xmin, ymax, xmax = vals
            bbox = (int(xmin * w / 1000), int(ymin * h / 1000), int(xmax * w / 1000), int(ymax * h / 1000))
            results.append({"id": 1, "label": "hand", "bbox": bbox})

        if "object" in data:
            vals = [float(v) for v in data["object"]]
            ymin, xmin, ymax, xmax = vals
            bbox = (int(xmin * w / 1000), int(ymin * h / 1000), int(xmax * w / 1000), int(ymax * h / 1000))
            object_name = str(data.get("object_name") or "object").strip() or "object"
            results.append({"id": 2, "label": object_name, "bbox": bbox})

        return results
    except Exception as je:
        logging.error(f"JSON Parse failed: {je}")
        return []

def call_doubao_hand_object(image_bgr):
    if not ARK_API_KEY:
        logging.error("ARK_API_KEY not set.")
        return []

    img_base64 = encode_image_to_base64(image_bgr)
    data_url = f"data:image/jpeg;base64,{img_base64}"
    grounding_text = build_hand_object_prompt()

    try:
        kwargs = {
            "model": DOUBAO_MODEL_NAME,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": "You are a professional vision assistant for robot hand-object interaction."}]
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
        content = extract_doubao_text(response)
        logging.info(f"Doubao Response: {content}")
        return parse_hand_object_response(content, image_bgr)
    except Exception as e:
        logging.error(f"API call failed: {e}")
        return []

def call_vlm_hand_object(image_bgr):
    return call_doubao_hand_object(image_bgr)

def vlm_thread_worker_hand(image_bgr, result_callback=None, done_callback=None):
    try:
        found_targets = call_vlm_hand_object(image_bgr)
        if found_targets and result_callback is not None:
            result_callback(found_targets)
            logging.info(f"MLLM Worker: Found {len(found_targets)} targets.")
    finally:
        if done_callback is not None:
            done_callback()


def prompt_zed_selection_mode(default_mode="mllm"):
    mode = (default_mode or "mllm").strip().lower()
    while True:
        prompt = "Select ZED hand/object selection mode before workspace selection [1=mllm, 2=manual]"
        if mode in ("mllm", "manual"):
            prompt += f" (default: {mode})"
        prompt += ": "
        ans = input(prompt).strip().lower()
        if not ans:
            return mode if mode in ("mllm", "manual") else "mllm"
        if ans in ("1", "mllm"):
            return "mllm"
        if ans in ("2", "manual"):
            return "manual"
        print("Please enter 1/mllm or 2/manual.")

def get_net(checkpoint_path, num_view=300):
    net = GraspNet(input_feature_dim=0, num_view=num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net, device

def resolve_checkpoint_path():
    ckpt_rs = os.path.join(BASELINE_DIR, 'logs', 'log_rs', 'checkpoint-rs.tar')
    ckpt_fallback = os.path.join(BASELINE_DIR, 'logs', 'log_rs', 'checkpoint.tar')
    if os.path.exists(ckpt_rs):
        return ckpt_rs
    if os.path.exists(ckpt_fallback):
        return ckpt_fallback
    raise FileNotFoundError(
        f"GraspNet checkpoint not found. Tried:\n  - {ckpt_rs}\n  - {ckpt_fallback}"
    )

def start_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    # RealSense uses its own supported RGB-D mode; do not reuse ZED's 640x360 viewer size.
    rs_width = 640
    rs_height = 480
    config.enable_stream(rs.stream.depth, rs_width, rs_height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs_width, rs_height, rs.format.rgb8, 30)
    
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale() 
    factor_depth = 1.0 / depth_scale 
    
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intrinsics = color_profile.get_intrinsics()
    
    camera = CameraInfo(float(rs_width), float(rs_height), intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy, factor_depth)
    return pipeline, align, camera

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

    # 保存用于可视化的完整点云
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
    with torch.autocast(device_type="cuda", dtype=torch.float32):
        pass 
    end_points['point_clouds'] = cloud_sampled_tensor
    end_points['cloud_colors'] = color_sampled
    
    return end_points, cloud_o3d, workspace_mask

def show_open3d_process(points, colors, gg_array):
    get_open3d_viewer().update(points, colors, gg_array, top_k=1)


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
        closing_camera = rot_mat[:, 1]

        approach_camera = approach_camera / (np.linalg.norm(approach_camera) + 1e-8)
        closing_camera = closing_camera / (np.linalg.norm(closing_camera) + 1e-8)

        approach_base = camera_rot_mat @ approach_camera
        closing_base = camera_rot_mat @ closing_camera

        approach_base = approach_base / (np.linalg.norm(approach_base) + 1e-8)
        closing_base = closing_base / (np.linalg.norm(closing_base) + 1e-8)

        cos_tilt = float(np.clip(np.dot(approach_base, camera_forward_base), -1.0, 1.0))
        is_within_tilt = cos_tilt >= max_tilt_cos
        is_not_singular = abs(np.dot(closing_base, camera_forward_base)) <= CAMERA_GRASP_MAX_CLOSING_FORWARD_COS
        keep_mask.append(is_within_tilt and is_not_singular)

    keep_mask = np.asarray(keep_mask, dtype=bool)
    return gg[keep_mask]


def filter_collision_grasps(gg, scene_points):
    if len(gg) == 0 or len(scene_points) == 0:
        return gg

    detector = ModelFreeCollisionDetector(scene_points, voxel_size=COLLISION_VOXEL_SIZE)
    collision_mask = detector.detect(
        gg,
        approach_dist=COLLISION_APPROACH_DIST,
        collision_thresh=COLLISION_THRESH
    )
    return gg[~collision_mask]





def legacy_realsense_main():
    global vlm_running, pending_targets, doubao_trigger, ROS_NODE
    
    # Init ROS2
    rclpy.init()
    ROS_NODE = RobotControllerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(ROS_NODE)
    t_ros = threading.Thread(target=executor.spin, daemon=True)
    t_ros.start()

    if ROS_NODE.rtde_c:
        ROS_NODE.get_logger().info("【初始化】移动机械臂到交接观测位置...")
        start_joint_q = [math.radians(100), math.radians(60), math.radians(-150), math.radians(-90), math.radians(-90), math.radians(0)]
        ROS_NODE.rtde_c.moveJ(start_joint_q, 0.5, 0.5)
        
        time.sleep(1.0)
        ROS_NODE.get_logger().info("【初始化】释放夹爪...")
        success = ROS_NODE.send_gripper_command(0.085)
        if not success:
            ROS_NODE.get_logger().warning("夹爪打开指令发送失败或超时，请检查 Action Server")
            
        ROS_NODE.get_logger().info("【初始化完成】机械臂已就位，启动相机流...")
    
    checkpoint_path = resolve_checkpoint_path()
    net, device = get_net(checkpoint_path)
    
    SAM2_CHECKPOINT = os.path.join(SAM2_DIR, "checkpoints/sam2.1/sam2.1_hiera_small.pt")
    SAM2_CFG = "sam2.1/sam2.1_hiera_s.yaml"
    logging.info("Loading SAM2 model...")
    sam2_predictor = build_sam2_camera_predictor(SAM2_CFG, SAM2_CHECKPOINT)
    sam2_predictor.fill_hole_area = 0
    sam2_predictor_rs = build_sam2_camera_predictor(SAM2_CFG, SAM2_CHECKPOINT)
    sam2_predictor_rs.fill_hole_area = 0
    logging.info("SAM2 model loaded")
    
    pipeline, align, camera_info = start_realsense()
    cv2.namedWindow("Realsense Viewer", cv2.WINDOW_AUTOSIZE)

    logging.info("Camera Started. UI Controls (focus on OpenCV window):")
    logging.info("  - v: Trigger Doubao prompt for hand and object selection")
    logging.info("  - Space: Pass Object Mask (ID 2) to GraspNet")
    logging.info("  - r: Reset SAM2 selection")
    logging.info("  - q: Quit")
    
    num_targets = 0 
    need_reset = False
    current_masks = {} 
    all_prompts = []
    
    try:
        while True:
            t0 = time.time()
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame: continue
            
            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())
            
            color_img_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            tracking_img = color_img_bgr.copy()

            # --- SAM2: Reset ---
            if need_reset:
                if hasattr(sam2_predictor, 'condition_state') and 'point_inputs_per_obj' in sam2_predictor.condition_state:
                    sam2_predictor.reset_state()
                num_targets = 0
                need_reset = False
                current_masks = {}
                all_prompts = []
                pending_targets = []
                logging.info("Reset tracking targets.")

            # --- Doubao Grounding ---
            if doubao_trigger and not vlm_running:
                vlm_snapshot = color_img_bgr.copy() # Snapshot taken at the moment of trigger
                vlm_running = True
                threading.Thread(target=vlm_thread_worker, args=(vlm_snapshot,), daemon=True).start()
                doubao_trigger = False

            # --- SAM2: Initialize targets ---
            if pending_targets:
                if hasattr(sam2_predictor, 'condition_state') and 'point_inputs_per_obj' in sam2_predictor.condition_state:
                    sam2_predictor.reset_state()
                
                sam2_predictor.load_first_frame(vlm_snapshot)
                all_prompts = []
                
                for tid, bbox in pending_targets:
                    x1, y1, x2, y2 = bbox
                    bbox_arr = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                    sam2_predictor.add_new_prompt(frame_idx=0, obj_id=tid, bbox=bbox_arr)
                    all_prompts.append({'id': tid, 'bbox': bbox})
                
                num_targets = len(all_prompts)
                pending_targets = []

            # --- SAM2: Track ---
            if num_targets > 0:
                out_obj_ids, out_mask_logits = sam2_predictor.track(tracking_img)
                current_masks = {}
                for i in range(len(out_obj_ids)):
                    obj_id = out_obj_ids[i]
                    current_masks[obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()

            # --- Visualization ---
            display = tracking_img.copy()
            for obj_id, mask in current_masks.items():
                if mask is not None:
                    color = MASK_COLORS_BGR.get(obj_id, [0, 255, 0])
                    overlay = display.copy()
                    overlay[mask > 0] = color
                    display = cv2.addWeighted(display, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(display, contours, -1, (0, 255, 0), 2)
                    
                    label = "Hand" if obj_id == 1 else "Object"
                    pos = np.argwhere(mask)
                    if len(pos) > 0:
                        y, x = pos[len(pos)//2]
                        cv2.putText(display, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if vlm_running:
                cv2.putText(display, "Processing", (IMG_WIDTH // 2 - 80, IMG_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            t1 = time.time()
            fps = 1.0 / (t1 - t0)
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            status = f"Targets: {num_targets} | v=Seg | SPACE=GraspNet | r=Reset | q=Quit"
            cv2.putText(display, status, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Realsense Viewer", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32: 
                print("Processing Frame for GraspNet...")
                # 获取物体的掩码 (ID 为 2)
                object_mask = None
                if 2 in current_masks and current_masks[2] is not None:
                    object_mask = current_masks[2] > 0
                
                if object_mask is None or not np.any(object_mask):
                    print("No Object mask found (ID 2), using default workspace heuristic!")
                else:
                    print("Object mask found. Proceeding with grasped object mask.")
                
                def process_grasp(c_img, d_img, cam_info, dev, c_mask, t_img):
                    end_points, cloud_o3d, workspace_mask = process_frame(c_img, d_img, cam_info, dev, sam_mask=c_mask)
                    if end_points is None:
                        print("No points found in workspace mask.")
                        return
                    
                    if DEBUG_MODE:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_dir = os.path.join("captured_data", timestamp)
                        os.makedirs(save_dir, exist_ok=True)
                        
                        cv2.imwrite(os.path.join(save_dir, 'color.png'), t_img)
                        cv2.imwrite(os.path.join(save_dir, 'depth.png'), d_img)
                        
                        mask_img = (workspace_mask.astype(np.uint8) * 255)
                        cv2.imwrite(os.path.join(save_dir, 'workspace_mask.png'), mask_img)
                        
                        try:
                            meta = {
                                'intrinsic_matrix': np.array([
                                    [cam_info.fx, 0, cam_info.cx],
                                    [0, cam_info.fy, cam_info.cy],
                                    [0, 0, 1]
                                ]),
                                'factor_depth': np.array([[cam_info.scale]])
                            }
                            scio.savemat(os.path.join(save_dir, 'meta.mat'), meta)
                            print(f"Data successfully saved to {save_dir}")
                        except Exception as e:
                            print("Failed to save meta.mat:", e)
                    
                    with torch.no_grad():
                        with torch.autocast(device_type="cuda", dtype=torch.float32):
                            end_points = net(end_points)
                            grasp_preds = pred_decode(end_points)
                    
                    gg_array = grasp_preds[0].detach().cpu().numpy()
                    
                    show_open3d_process(
                        np.asarray(cloud_o3d.points),
                        np.asarray(cloud_o3d.colors),
                        gg_array
                    )

                    gg_filtered = GraspGroup(gg_array).nms().sort_by_score()
                    if len(gg_filtered) > 0 and ROS_NODE:
                        ROS_NODE.execute_grasp(gg_filtered)

                t = threading.Thread(target=process_grasp, args=(
                    color_img.copy(), depth_img.copy(), camera_info, device, 
                    object_mask.copy() if object_mask is not None else None, 
                    tracking_img.copy()))
                t.daemon = True
                t.start()
                
            elif key == ord('v'):
                doubao_trigger = True
            elif key == ord('r'):
                need_reset = True
            elif key == ord('q'):
                break
                
    finally:
        pipeline.stop()
        if ROS_NODE and ROS_NODE.rtde_c: ROS_NODE.rtde_c.stopScript()
        rclpy.shutdown()
        close_open3d_viewer()
        cv2.destroyAllWindows()

def start_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    rs_width = 640
    rs_height = 480
    config.enable_stream(rs.stream.depth, rs_width, rs_height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs_width, rs_height, rs.format.rgb8, 30)
    
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale() 
    factor_depth = 1.0 / depth_scale 
    
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intrinsics = color_profile.get_intrinsics()
    
    camera = CameraInfo(float(rs_width), float(rs_height), intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy, factor_depth)
    return pipeline, align, camera

def process_frame(color_img, depth_img, camera, device, sam_mask=None, num_point=20000):
    color = color_img.astype(np.float32) / 255.0
    
    if sam_mask is not None and np.any(sam_mask):
        workspace_mask = (sam_mask > 0) & (depth_img > 0) & (depth_img < 2000)
    else:
        z_min_mm, z_max_mm = 200, 1000 
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

    # 保存用于可视化的完整点云
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
    with torch.autocast(device_type="cuda", dtype=torch.float32):
        pass 
    end_points['point_clouds'] = cloud_sampled_tensor
    end_points['cloud_colors'] = color_sampled
    
    return end_points, cloud_o3d, workspace_mask

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

def main(debug=False):
    global DEBUG_MODE
    DEBUG_MODE = debug
    global need_reset, num_targets, pending_bbox, pending_point, current_masks, all_prompts, drawing, ix, iy, fx_mouse, fy_mouse
    global vlm_running, doubao_trigger, workspace_bbox, pending_targets, ROS_NODE, STATE
    global zed_drawing, zed_ix, zed_iy, zed_fx_mouse, zed_fy_mouse, zed_pending_bbox, zed_pending_point

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
    
    
    SAM2_CHECKPOINT = os.path.join(SAM2_DIR, "checkpoints/sam2.1/sam2.1_hiera_small.pt")
    SAM2_CFG = "sam2.1/sam2.1_hiera_s.yaml"
    logging.info("Loading SAM2 model...")
    sam2_predictor = build_sam2_camera_predictor(SAM2_CFG, SAM2_CHECKPOINT)
    sam2_predictor.fill_hole_area = 0
    sam2_predictor_rs = build_sam2_camera_predictor(SAM2_CFG, SAM2_CHECKPOINT)
    sam2_predictor_rs.fill_hole_area = 0
    logging.info("SAM2 model loaded")

    zed_selection_mode = "mllm"
    
    zed, K_scaled, baseline, camera_info = start_zed()
    
    # GraspNet Init
    checkpoint_path = resolve_checkpoint_path()
    net, device = get_net(checkpoint_path)
    pipeline, align, camera_info_rs = start_realsense()
    image_left = sl.Mat()
    image_right = sl.Mat()

    cv2.namedWindow("Realsense Viewer", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Realsense Viewer", mouse_callback)
    cv2.namedWindow("ZED Inhand Viewer", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("ZED Inhand Viewer", zed_mouse_callback)


    logging.info("Camera Started. UI Controls (focus on OpenCV window):")
    logging.info("  - w: Trigger Doubao to detect Workpace")
    logging.info("  - Left-click drag: Draw object bounding box -> track")
    logging.info("  - Left-click: Select foreground object point -> track")
    logging.info("  - v: Trigger ZED hand/object segmentation (mode asked after workspace)")
    logging.info("  - Left fist during following: auto segmentation only")
    logging.info("  - Space (in segmentation mode): run GraspNet grasp generation")
    logging.info("  - r: Reset current selection")
    logging.info("  - q: Quit")
    
    num_targets = 0 
    need_reset = False
    current_masks = {}
    all_prompts = []
    zed_need_reset = False
    zed_vlm_running = False
    zed_pending_targets = []
    zed_doubao_trigger = False
    zed_current_masks = {}
    zed_all_prompts = []
    zed_vlm_snapshot = None
    zed_viewer_requested = False
    zed_grasp_request_pending = False
    zed_segmentation_ready_time = 0.0
    zed_selection_source = None
    zed_sam2_initialized = False
    zed_selection_mode_default = "mllm"
    
    initialized_dynamic_start = False
    handover_start_calc_running = False
    follow_calc_running = False
    handover_grasp_running = False

    def select_zed_mode_for_round():
        nonlocal zed_selection_mode, zed_selection_mode_default
        zed_selection_mode = prompt_zed_selection_mode(default_mode=zed_selection_mode_default)
        zed_selection_mode_default = zed_selection_mode
        if ROS_NODE is not None:
            ROS_NODE.get_logger().info(f"ZED hand/object selection mode for this round: {zed_selection_mode}")
        return zed_selection_mode

    if ROS_NODE is not None:
        ROS_NODE.zed_selection_mode_callback = select_zed_mode_for_round

    def launch_handover_grasp(c_img, c_bgr, l_rgb, r_rgb, masks):
        nonlocal handover_grasp_running
        if handover_grasp_running:
            return
        handover_grasp_running = True

        object_mask = None
        if 2 in masks and masks[2] is not None:
            object_mask = masks[2] > 0
            if object_mask is None or not np.any(object_mask):
                logging.warning("No object mask found (ID 2), using default depth heuristic.")
                object_mask = None
        else:
            logging.warning("No object mask found (ID 2), using default depth heuristic.")

        logging.info("Processing ZED frame for handover GraspNet...")
        zed_depth_mm, zed_disp = compute_ffs_depth_and_disp(l_rgb, r_rgb, K_scaled, baseline)

        def process_handover_grasp(c_img_i, d_img_i, cam_info_i, dev_i, c_mask_i, t_img_i):
            nonlocal handover_grasp_running, zed_need_reset
            try:
                from graspnetAPI import GraspGroup

                end_points, cloud_o3d, workspace_mask = process_frame(c_img_i, d_img_i, cam_info_i, dev_i, sam_mask=c_mask_i)
                if end_points is None:
                    logging.warning("No valid points found for GraspNet.")
                    return

                if DEBUG_MODE:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_dir = os.path.join("captured_data", timestamp)
                    os.makedirs(save_dir, exist_ok=True)

                    cv2.imwrite(os.path.join(save_dir, 'color.png'), t_img_i)
                    cv2.imwrite(os.path.join(save_dir, 'depth.png'), d_img_i)
                    cv2.imwrite(os.path.join(save_dir, 'workspace_mask.png'), workspace_mask.astype(np.uint8) * 255)

                    try:
                        meta = {
                            'intrinsic_matrix': np.array([
                                [cam_info_i.fx, 0, cam_info_i.cx],
                                [0, cam_info_i.fy, cam_info_i.cy],
                                [0, 0, 1]
                            ]),
                            'factor_depth': np.array([[cam_info_i.scale]])
                        }
                        scio.savemat(os.path.join(save_dir, 'meta.mat'), meta)
                        logging.info(f"Saved ZED inhand capture to {save_dir}")
                    except Exception as e:
                        logging.warning(f"Failed to save meta.mat: {e}")

                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.float32):
                        end_points = net(end_points)
                        grasp_preds = pred_decode(end_points)

                gg_array = grasp_preds[0].detach().cpu().numpy()
                gg_filtered = GraspGroup(gg_array).nms().sort_by_score()
                raw_count = len(gg_filtered)

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

                gg_filtered = filter_grasps_by_camera_direction(gg_filtered, camera_rot_mat).sort_by_score()
                orient_count = len(gg_filtered)
                logging.info(
                    "Handover grasp filtering: "
                    f"raw={raw_count}, orient={orient_count}"
                )

                show_open3d_process(
                    np.asarray(cloud_o3d.points),
                    np.asarray(cloud_o3d.colors),
                    gg_filtered.grasp_group_array if len(gg_filtered) > 0 else np.empty((0, 17), dtype=np.float32)
                )
                p = None

                save_zed_grasp_capture_bundle(
                    left_rgb=c_img_i,
                    tracking_bgr=t_img_i,
                    depth_mm=d_img_i,
                    disp=zed_disp,
                    camera=cam_info_i,
                    workspace_mask=workspace_mask,
                    gg_before=GraspGroup(gg_array).nms().sort_by_score(),
                    gg_after=gg_filtered,
                    note="human2robot",
                )

                if len(gg_filtered) == 0 or ROS_NODE is None:
                    logging.warning("No valid grasp candidate generated.")
                    return

                ans = input("\nPoint cloud generated. Execute handover grasp now? Press 'y' to continue, 'n' to cancel: ")
                if ans.lower() != 'y':
                    logging.info("Handover execution cancelled. Robot remains at current handover pose.")
                    return

                # Reset SAM targets immediately before grasp execution.
                # execute_grasp may block while waiting for right-hand detection, so
                # we cannot defer this reset until after execute_grasp returns.
                zed_need_reset = True
                ROS_NODE.execute_grasp(gg_filtered, viz_process=p, require_confirmation=False)
            finally:
                handover_grasp_running = False

        t = threading.Thread(
            target=process_handover_grasp,
            args=(
                c_img.copy(),
                zed_depth_mm.copy(),
                camera_info,
                device,
                object_mask.copy() if object_mask is not None else None,
                c_bgr.copy(),
            ),
            daemon=True,
        )
        t.start()

    def trigger_zed_segmentation(reason="manual trigger"):
        nonlocal zed_doubao_trigger, zed_need_reset, zed_grasp_request_pending, follow_calc_running, zed_selection_source, zed_sam2_initialized
        global STATE
        STATE = "HANDOVER_SEGMENTATION"
        zed_need_reset = True
        zed_grasp_request_pending = False
        zed_sam2_initialized = False
        follow_calc_running = False
        if zed_selection_mode == "mllm":
            zed_doubao_trigger = True
            zed_selection_source = "mllm"
            logging.info(f"{reason}: triggering ZED hand/object detection in MLLM mode.")
        else:
            zed_doubao_trigger = False
            zed_selection_source = "manual"
            logging.info(f"{reason}: entering ZED manual hand/object selection mode. Please click or box-select hand first, then object.")
    
    # 刚启动时，自动触发工作台检测
    doubao_trigger = True
    
    try:
        while True:
            t0 = time.time()
            if zed.grab() != sl.ERROR_CODE.SUCCESS:
                continue
                
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            zed.retrieve_image(image_right, sl.VIEW.RIGHT)

            raw_left_bgr = image_left.get_data()[:, :, :3]
            raw_right_bgr = image_right.get_data()[:, :, :3]

            left_bgr = cv2.resize(raw_left_bgr, (IMG_WIDTH, IMG_HEIGHT))
            right_bgr = cv2.resize(raw_right_bgr, (IMG_WIDTH, IMG_HEIGHT))

            left_rgb = left_bgr[:, :, ::-1].copy()
            right_rgb = right_bgr[:, :, ::-1].copy()

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            rs_color_frame = aligned_frames.get_color_frame()
            rs_depth_frame = aligned_frames.get_depth_frame()
            if not rs_color_frame or not rs_depth_frame:
                continue
            rs_color_image = np.asanyarray(rs_color_frame.get_data())
            rs_depth_image = np.asanyarray(rs_depth_frame.get_data())
            rs_color_image_bgr = cv2.cvtColor(rs_color_image, cv2.COLOR_RGB2BGR)

            color_img_bgr = rs_color_image_bgr.copy()
            tracking_img = color_img_bgr.copy()
            color_img = rs_color_image.copy()
            display = tracking_img.copy()

            # --- ZED in-hand viewer / segmentation ---
            # Keep ZED stream visible in all states (startup, grasping, post-grasp).
            try:
                zed_color_image = left_rgb.copy()
                zed_color_image_bgr = left_bgr.copy()
                zed_display = zed_color_image_bgr.copy()

                if zed_need_reset:
                    if hasattr(sam2_predictor_rs, 'condition_state') and 'point_inputs_per_obj' in sam2_predictor_rs.condition_state:
                        sam2_predictor_rs.reset_state()
                    zed_current_masks = {}
                    zed_all_prompts = []
                    zed_pending_targets = []
                    zed_pending_bbox = None
                    zed_pending_point = None
                    zed_drawing = False
                    zed_grasp_request_pending = False
                    zed_segmentation_ready_time = 0.0
                    zed_sam2_initialized = False
                    zed_need_reset = False
                    logging.info("Reset ZED hand/object segmentation targets.")

                if zed_doubao_trigger and not zed_vlm_running:
                    zed_vlm_snapshot = zed_color_image_bgr.copy()
                    zed_vlm_running = True

                    def on_rs_targets(found_targets):
                        nonlocal zed_pending_targets
                        zed_pending_targets = found_targets

                    def on_rs_done():
                        nonlocal zed_vlm_running
                        zed_vlm_running = False

                    threading.Thread(
                        target=vlm_thread_worker_hand,
                        args=(zed_vlm_snapshot, on_rs_targets, on_rs_done),
                        daemon=True,
                    ).start()
                    zed_doubao_trigger = False

                if zed_selection_mode == "manual" and STATE == "HANDOVER_SEGMENTATION":
                    if (zed_pending_bbox is not None or zed_pending_point is not None) and len(zed_all_prompts) < 2:
                        next_id = 1 if len(zed_all_prompts) == 0 else 2
                        next_label = "hand" if next_id == 1 else "object"
                        target = {"id": next_id, "label": next_label}
                        if zed_pending_bbox is not None:
                            target["bbox"] = zed_pending_bbox
                            zed_pending_bbox = None
                        elif zed_pending_point is not None:
                            target["point"] = zed_pending_point
                            zed_pending_point = None
                        zed_all_prompts.append(target)
                        logging.info(f"Added manual ZED target {next_id}: {next_label}")

                    if len(zed_all_prompts) == 2 and not zed_current_masks and not zed_grasp_request_pending:
                        if hasattr(sam2_predictor_rs, 'condition_state') and 'point_inputs_per_obj' in sam2_predictor_rs.condition_state:
                            sam2_predictor_rs.reset_state()
                        zed_vlm_snapshot = zed_color_image_bgr.copy()
                        sam2_predictor_rs.load_first_frame(zed_vlm_snapshot)
                        zed_sam2_initialized = True

                        for prompt in zed_all_prompts:
                            tid = int(prompt["id"])
                            if "bbox" in prompt:
                                x1, y1, x2, y2 = prompt["bbox"]
                                bbox_arr = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                                sam2_predictor_rs.add_new_prompt(frame_idx=0, obj_id=tid, bbox=bbox_arr)
                            elif "point" in prompt:
                                px, py = prompt["point"]
                                pts_arr = np.array([[px, py]], dtype=np.float32)
                                lbl_arr = np.array([1], dtype=np.int32)
                                sam2_predictor_rs.add_new_prompt(frame_idx=0, obj_id=tid, points=pts_arr, labels=lbl_arr)

                        save_hand_object_bbox_artifacts(zed_vlm_snapshot, zed_all_prompts, prefix="human2robot_manual_bbox")
                        zed_grasp_request_pending = True
                        zed_segmentation_ready_time = time.time()
                        logging.info("Manual ZED hand/object prompts are ready. Waiting for SAM2 to stabilize before GraspNet.")

                if zed_pending_targets:
                    if hasattr(sam2_predictor_rs, 'condition_state') and 'point_inputs_per_obj' in sam2_predictor_rs.condition_state:
                        sam2_predictor_rs.reset_state()

                    save_hand_object_bbox_artifacts(zed_vlm_snapshot, zed_pending_targets, prefix="human2robot_bbox")
                    sam2_predictor_rs.load_first_frame(zed_vlm_snapshot)
                    zed_sam2_initialized = True
                    zed_all_prompts = []

                    for target in zed_pending_targets:
                        tid = int(target.get("id", -1))
                        bbox = target.get("bbox")
                        if bbox is None:
                            continue
                        x1, y1, x2, y2 = bbox
                        bbox_arr = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                        sam2_predictor_rs.add_new_prompt(frame_idx=0, obj_id=tid, bbox=bbox_arr)
                        zed_all_prompts.append({'id': tid, 'bbox': bbox, 'label': target.get("label", f"id_{tid}")})

                    zed_pending_targets = []
                    zed_grasp_request_pending = True
                    zed_segmentation_ready_time = time.time()

                if zed_sam2_initialized and zed_all_prompts:
                    out_obj_ids, out_mask_logits = sam2_predictor_rs.track(zed_color_image_bgr)
                    zed_current_masks = {}
                    for i in range(len(out_obj_ids)):
                        obj_id = out_obj_ids[i]
                        zed_current_masks[obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()

                for obj_id, mask in zed_current_masks.items():
                    if mask is not None and mask.shape == zed_display.shape[:2]:
                        m_color = [0, 255, 0] if obj_id == 2 else [255, 0, 0] # 2 is object, 1 is hand
                        overlay = zed_display.copy()
                        overlay[mask > 0] = m_color
                        zed_display = cv2.addWeighted(zed_display, 0.7, overlay, 0.3, 0)
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(zed_display, contours, -1, m_color, 2)
                        label = next((p.get("label") for p in zed_all_prompts if p.get("id") == obj_id), str(obj_id))
                        if contours:
                            x, y, _, _ = cv2.boundingRect(contours[0])
                            cv2.putText(zed_display, str(label), (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, m_color, 2)

                if zed_drawing and zed_ix >= 0:
                    cv2.rectangle(zed_display, (zed_ix, zed_iy), (zed_fx_mouse, zed_fy_mouse), (255, 200, 0), 2)
                
                if zed_vlm_running:
                    cv2.putText(zed_display, "MLLM Detecting Hand/Object...", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                elif zed_selection_mode == "manual" and STATE == "HANDOVER_SEGMENTATION":
                    if len(zed_all_prompts) == 0:
                        zed_manual_hint = "Manual ZED Seg: select hand first"
                    elif len(zed_all_prompts) == 1:
                        zed_manual_hint = "Manual ZED Seg: select object second"
                    else:
                        zed_manual_hint = "Manual ZED Seg: hand/object selected"
                    cv2.putText(zed_display, zed_manual_hint, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                if STATE == "HANDOVER_SEGMENTATION":
                    if zed_grasp_request_pending:
                        wait_left = max(0.0, SAM2_SETTLE_SECONDS - (time.time() - zed_segmentation_ready_time))
                        rs_hint = f"ZED Segmentation | waiting {wait_left:.1f}s -> GraspNet | r=Reset | q=Quit"
                    else:
                        if zed_selection_mode == "manual":
                            rs_hint = "ZED Segmentation(manual) | click/box hand then object | SPACE=Run GraspNet | r=Reset | q=Quit"
                        else:
                            rs_hint = "ZED Segmentation(MLLM) | SPACE=Run GraspNet | r=Reset | q=Quit"
                else:
                    rs_hint = f"ZED Live | mode={zed_selection_mode} | v=Detect hand/object | q=Quit"
                rs_status = rs_hint
                cv2.putText(zed_display, rs_status, (10, zed_display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                cv2.imshow("ZED Inhand Viewer", zed_display)

                if zed_grasp_request_pending and zed_all_prompts and (not handover_grasp_running):
                    if time.time() - zed_segmentation_ready_time >= SAM2_SETTLE_SECONDS:
                        zed_grasp_request_pending = False
                        launch_handover_grasp(
                            zed_color_image.copy(),
                            zed_color_image_bgr.copy(),
                            left_rgb.copy(),
                            right_rgb.copy(),
                            zed_current_masks,
                        )
            except Exception as e:
                logging.error(f"ZED inhand segmentation read error: {e}")

            # --- MediaPipe Two-Hand Handover ---
            if STATE in ("WAITING_HANDOVER_START_POSE", "WAITING_DELIVERY", "FOLLOWING_HAND"):
                results = hands.process(cv2.cvtColor(color_img_bgr, cv2.COLOR_BGR2RGB))
                right_hand_landmarks, right_hand_score = select_right_hand(results)
                left_hand_landmarks, left_hand_score = select_left_hand(results)
                left_fist = False

                if left_hand_landmarks:
                    mp_drawing.draw_landmarks(display, left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    h, w, _ = display.shape
                    left_palm_lm = left_hand_landmarks.landmark[9]
                    left_cx, left_cy = int(left_palm_lm.x * w), int(left_palm_lm.y * h)
                    left_fist = is_fist(left_hand_landmarks)
                    left_label = "Physical Left fist" if left_fist else "Physical Left hand"
                    cv2.circle(display, (left_cx, left_cy), 8, (0, 255, 255), cv2.FILLED)
                    cv2.putText(display, f"{left_label} {left_hand_score:.2f}", (left_cx-85, left_cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if right_hand_landmarks:
                    mp_drawing.draw_landmarks(display, right_hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    h, w, _ = display.shape
                    wrist_lm = right_hand_landmarks.landmark[0]
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
                            t = threading.Thread(target=calc_rs_and_update_handover_start, args=(rs_depth_image.copy(), camera_info_rs, cx, cy), daemon=True)
                            t.start()

                    elif STATE == "WAITING_DELIVERY":
                        cv2.putText(display, "Starting to follow Right hand...", (cx-90, cy-35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        STATE = "FOLLOWING_HAND"

                    elif STATE == "FOLLOWING_HAND":
                        cv2.putText(display, "Following Right hand. Left fist to segment.", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        if left_fist:
                            save_mediapipe_intent_frame(
                                display.copy(),
                                prefix="human2robot_left_fist_intent",
                                detected_text="Detected left fist intent",
                            )
                            trigger_zed_segmentation(reason="Left fist detected")
                        elif not follow_calc_running:
                            follow_calc_running = True
                            t = threading.Thread(target=calc_rs_and_follow, args=(rs_depth_image.copy(), camera_info_rs, cx, cy), daemon=True)
                            t.start()
                else:
                    cv2.putText(display, "Waiting for MediaPipe Right hand...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)


            # 这里原本是 MLLM 调用，现已废弃

            # --- SAM2: Reset ---
            if need_reset:
                if hasattr(sam2_predictor, 'condition_state') and 'point_inputs_per_obj' in sam2_predictor.condition_state:
                    sam2_predictor.reset_state()
                num_targets = 0
                need_reset = False
                pending_bbox = None
                pending_point = None
                current_masks = {}
                all_prompts = []
                # workspace_bbox = None # (可选)如果重置想要一起清空工作台
                logging.info("Reset, select new targets.")

            # --- SAM2: Initialize object tracking ---
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

            # --- SAM2: Track ---
            if num_targets > 0:
                out_obj_ids, out_mask_logits = sam2_predictor.track(tracking_img)
                current_masks = {}
                for i in range(len(out_obj_ids)):
                    obj_id = out_obj_ids[i]
                    current_masks[obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()

            # --- Visualization ---
            for obj_id, mask in current_masks.items():
                if mask is not None:
                    color = MASK_COLORS_BGR.get(obj_id, [0, 255, 0])
                    overlay = display.copy()
                    overlay[mask > 0] = color
                    display = cv2.addWeighted(display, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(display, contours, -1, (0, 255, 0), 2)
            
            # 画当前工作台框 (Rotated Rect from SAM2)
            if workspace_bbox is not None:
                cv2.polylines(display, [workspace_bbox], isClosed=True, color=(0, 165, 255), thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(display, "Workspace", (workspace_bbox[0][0], max(20, workspace_bbox[0][1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                
            if drawing and ix >= 0:
                cv2.rectangle(display, (ix, iy), (fx_mouse, fy_mouse), (255, 200, 0), 2)
                
            if vlm_running:
                cv2.putText(display, "Detecting Workspace...", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            t1 = time.time()
            fps = 1.0 / (t1 - t0)
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            status = f"Targets: {num_targets} | STATE: {STATE} | w=Workspace | v=ZED Seg | LeftFist=AutoSeg | SPACE=Grasp | r=reset | q=quit"
            cv2.putText(display, status, (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Realsense Viewer", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('w'):
                if num_targets > 0:
                    logging.info("Setting current SAM2 targets as workspace...")
                    combined_mask = np.zeros(display.shape[:2], dtype=bool)
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
                            
                            def calc_rs_and_move_ws(d_img, cam_info, pt_x, pt_y, e_pt1, e_pt2):
                                logging.info("Calculating dynamic start position via RealSense depth...")
                                h, w = d_img.shape[:2]
                                px = int(np.clip(pt_x, 0, w - 1))
                                py = int(np.clip(pt_y, 0, h - 1))
                                x1, x2 = max(0, px - 2), min(w, px + 3)
                                y1, y2 = max(0, py - 2), min(h, py + 3)
                                patch = d_img[y1:y2, x1:x2].astype(np.float32)
                                valid = patch[patch > 0]
                                if valid.size == 0:
                                    logging.warning("Workspace selection failed: invalid RealSense depth.")
                                    return

                                depth_scale = getattr(cam_info, "factor_depth", getattr(cam_info, "scale", None))
                                if depth_scale is None or depth_scale <= 0:
                                    logging.warning("Workspace selection failed: invalid camera depth scale.")
                                    return
                                depth_m = float(np.median(valid)) / depth_scale
                                if not (0.1 < depth_m < 2.0):
                                    logging.warning("Workspace selection failed: computed depth out-of-bounds.")
                                    return

                                x_cam = (px - cam_info.cx) * depth_m / cam_info.fx
                                y_cam = (py - cam_info.cy) * depth_m / cam_info.fy
                                point_3d_cam = np.array([x_cam, y_cam, depth_m], dtype=np.float64)

                                x1_cam = (e_pt1[0] - cam_info.cx) * depth_m / cam_info.fx
                                y1_cam = (e_pt1[1] - cam_info.cy) * depth_m / cam_info.fy
                                x2_cam = (e_pt2[0] - cam_info.cx) * depth_m / cam_info.fx
                                y2_cam = (e_pt2[1] - cam_info.cy) * depth_m / cam_info.fy
                                pt1_cam = np.array([x1_cam, y1_cam, depth_m], dtype=np.float64)
                                pt2_cam = np.array([x2_cam, y2_cam, depth_m], dtype=np.float64)

                                tf_msg = ROS_NODE.tf_buffer.lookup_transform('base_link', 'rs_tohand_color_optical_frame', rclpy.time.Time())
                                tf_rot = R.from_quat([tf_msg.transform.rotation.x, tf_msg.transform.rotation.y, tf_msg.transform.rotation.z, tf_msg.transform.rotation.w])
                                tf_trans = np.array([tf_msg.transform.translation.x, tf_msg.transform.translation.y, tf_msg.transform.translation.z], dtype=np.float64)

                                ws_center_base = tf_rot.apply(point_3d_cam) + tf_trans
                                pt1_base = tf_rot.apply(pt1_cam) + tf_trans
                                pt2_base = tf_rot.apply(pt2_cam) + tf_trans

                                V = pt2_base - pt1_base
                                V[2] = 0
                                V_norm = np.linalg.norm(V)
                                if V_norm < 1e-5:
                                    V = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                                    V_norm = 1.0

                                X_axis = V / V_norm
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

                                ws_center_xy = np.array(ws_center_base[:2], dtype=np.float64)
                                ws_center_xy_norm = np.linalg.norm(ws_center_xy)
                                observe_xy = ws_center_xy.copy()
                                if ws_center_xy_norm >= 1e-5:
                                    observe_xy = ws_center_xy - (ws_center_xy / ws_center_xy_norm) * WORKSPACE_OBSERVE_BASE_OFFSET
                                else:
                                    logging.warning("Workspace center too close to base origin; skip 10cm base offset for observation pose.")

                                dyn_center_pose = [-ws_center_base[0], -ws_center_base[1], ws_center_base[2] + 0.30, rx, ry, rz]
                                dyn_observe_pose = [-observe_xy[0], -observe_xy[1], ws_center_base[2] + 0.30, rx, ry, rz]
                                ROS_NODE.dynamic_workspace_center_pose = dyn_center_pose
                                ROS_NODE.dynamic_start_pose = dyn_observe_pose
                                ROS_NODE.get_logger().info(
                                    f"Moving to observation pose (30cm above workspace, 10cm toward base): {dyn_observe_pose}"
                                )
                                ROS_NODE.rtde_c.moveL(dyn_observe_pose, 0.1, 0.1)

                                global STATE
                                select_zed_mode_for_round()
                                STATE = "WAITING_HANDOVER_START_POSE"
                                ROS_NODE.get_logger().info("Workspace recognized. Waiting for MediaPipe Right hand direction before handover position...")
                                handover_joint_q = ROS_NODE.wait_for_handover_joint_q()
                                ROS_NODE.get_logger().info("Moving to Handover Observation Position...")
                                ROS_NODE.rtde_c.moveJ(handover_joint_q, 0.5, 0.5)
                                ROS_NODE.handover_follow_pose = list(ROS_NODE.rtde_r.getActualTCPPose())
                                ROS_NODE.handover_follow_z_offset = None
                                ROS_NODE.get_logger().info("Standing by at handover position.")
                                STATE = "FOLLOWING_HAND"

                        t_start = threading.Thread(target=calc_rs_and_move_ws, args=(rs_depth_image.copy(), camera_info_rs, cx, cy, edge_pt1, edge_pt2), daemon=True)
                        t_start.start()
                        
                        # 清理现有 targets 留空进行抓取选择
                        need_reset = True
                else:
                    logging.warning("No SAM2 tracking target to set as workspace. Please specify target first.")
            elif key == ord('v'):
                if STATE in ("WAITING_DELIVERY", "FOLLOWING_HAND", "HANDOVER_SEGMENTATION"):
                    trigger_zed_segmentation(reason="Manual key v")
                else:
                    logging.warning("Press 'v' after the robot reaches the handover observation position.")
            elif key == 32:
                if STATE != "HANDOVER_SEGMENTATION":
                    logging.warning("Press SPACE only after pressing 'v' to enter ZED segmentation mode.")
                    continue

                if 'zed_color_image' not in locals():
                    logging.warning("ZED frame not ready yet.")
                    continue

                if not zed_all_prompts:
                    logging.warning("No ZED SAM2 targets available yet.")
                    continue

                if zed_selection_mode == "manual" and len(zed_all_prompts) < 2:
                    logging.warning("Manual mode needs both hand and object selections before running GraspNet.")
                    continue

                if not zed_sam2_initialized:
                    logging.warning("ZED SAM2 is not initialized yet. Please finish target selection first.")
                    continue

                zed_grasp_request_pending = True
                if zed_segmentation_ready_time <= 0:
                    zed_segmentation_ready_time = time.time()
                wait_left = max(0.0, SAM2_SETTLE_SECONDS - (time.time() - zed_segmentation_ready_time))
                if wait_left > 0:
                    logging.info(f"Queued handover grasp generation. Waiting {wait_left:.1f}s for SAM2 to stabilize.")
                else:
                    logging.info("Queued handover grasp generation. SAM2 is ready.")
            elif key == ord('r'):
                if STATE == "HANDOVER_SEGMENTATION":
                    trigger_zed_segmentation(reason="Reset ZED hand/object segmentation")
                else:
                    need_reset = True
            elif key == ord('q'):
                break
                
    finally:
        zed.close()
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        if ROS_NODE and ROS_NODE.rtde_c:
            ROS_NODE.rtde_c.stopScript()
        try:
            executor.shutdown()
        except Exception:
            pass
        rclpy.shutdown()
        close_open3d_viewer()
        cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="Human-to-robot handover pipeline")
    parser.add_argument("--debug", action="store_true", help="Save debug artifacts like captured_data, llm_output, mediapipe_output")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(debug=args.debug)
