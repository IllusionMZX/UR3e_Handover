import os
import sys
import math
import time
import datetime
import logging
import threading
import multiprocessing
from contextlib import nullcontext

import cv2
import numpy as np
import open3d as o3d
import scipy.io as scio
import torch
import yaml
import pyzed.sl as sl

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


logging.basicConfig(level=logging.INFO, format='%(message)s')

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

PROJECT_ROOT = os.path.dirname(ROOT_DIR)
SAM2_DIR = os.path.join(PROJECT_ROOT, "SAM2_streaming")
sys.path.insert(0, SAM2_DIR)
from sam2.build_sam import build_sam2_camera_predictor

FFS_DIR = os.path.join(PROJECT_ROOT, "Fast-FoundationStereoPhysics")
sys.path.append(FFS_DIR)
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, vis_disparity

from graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup
from data_utils import CameraInfo, create_point_cloud_from_depth_image


CAMERA_FRAME = 'zed_inhand_camera_frame_optical'
IMG_WIDTH = 640
IMG_HEIGHT = 360
SCALE_FACTOR = 0.5
VALID_ITERS = 8
MAX_DISP = 192
ZNEAR = 0.2
ZFAR = 5.0
MODEL_DIR = os.path.join(FFS_DIR, "weights/23-36-37/model_best_bp2_serialize.pth")
MASK_COLORS_BGR = {1: [75, 70, 203], 2: [203, 192, 75]}
MASK_ALPHA = 0.5
DENOISE_VOXEL_SIZE = 0.001
DENOISE_NB_POINTS = 30
DENOISE_RADIUS = 0.03
CAMERA_AXIS_MAX_TILT_DEG = 30.0
OPEN3D_BEFORE_TOP_K = 20
OPEN3D_AFTER_TOP_K = 1
BACKOFF_DIST = 0.08
Z_OFFSET_DOWN = 0.005
BASE_TO_CONTROLLER_ROT = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)


class RobotControllerNode(Node):
    def __init__(self):
        super().__init__('robot_controller_node')
        self.callback_group = ReentrantCallbackGroup()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._action_client = ActionClient(
            self,
            GripperCommand,
            '/robotiq_2f_urcap_adapter/gripper_command',
            callback_group=self.callback_group
        )
        self.ur_ip = "192.168.1.10"
        try:
            self.rtde_c = RTDEControlInterface(self.ur_ip)
            self.rtde_r = RTDEReceiveInterface(self.ur_ip)
            self.get_logger().info("✓ Connected to UR3e")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to UR: {e}")
            self.rtde_c = None

    def send_gripper_command(self, position, speed=0.15, force=140.0):
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            return False
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = force
        goal_msg.command.max_speed = speed
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        while not send_goal_future.done():
            time.sleep(0.1)
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            return False
        get_result_future = goal_handle.get_result_async()
        while not get_result_future.done():
            time.sleep(0.1)
        return get_result_future.result().result.reached_goal

    def execute_grasp(self, gg, viz_process=None):
        if not self.rtde_c:
            self.get_logger().error("RTDE not connected.")
            return

        try:
            best_grasp = gg[0]
            trans_cam = best_grasp.translation
            rot_mat_cam = best_grasp.rotation_matrix

            tf_msg = self.tf_buffer.lookup_transform('base_link', CAMERA_FRAME, rclpy.time.Time())
            tf_trans = np.array([
                tf_msg.transform.translation.x,
                tf_msg.transform.translation.y,
                tf_msg.transform.translation.z
            ])
            tf_rot = R.from_quat([
                tf_msg.transform.rotation.x,
                tf_msg.transform.rotation.y,
                tf_msg.transform.rotation.z,
                tf_msg.transform.rotation.w
            ])

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

            start_joint_q = [0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0]
            self.get_logger().info("Moving to Start Joint Position...")
            self.rtde_c.moveJ(start_joint_q, 0.5, 0.5)
            self.send_gripper_command(0.085)

            ans = input("\nReady to grasp? Press 'y' to continue, 'n' to cancel: ")
            print()
            if ans.lower() != 'y':
                self.get_logger().info("Grasp cancelled.")
                return

            self.get_logger().info("Moving to approach pose...")
            self.rtde_c.moveL(approach_pose, 0.05, 0.05)
            self.get_logger().info("Moving to target pose...")
            self.rtde_c.moveL(target_pose, 0.05, 0.05)
            self.get_logger().info("Closing gripper...")
            self.send_gripper_command(0.0)

            self.get_logger().info("Lifting...")
            self.rtde_c.moveL(approach_pose, 0.05, 0.05)
            self.get_logger().info("Returning to Start Joint Position...")
            self.rtde_c.moveJ(start_joint_q, 0.5, 0.5)
            self.get_logger().info("Grasp Executed. Waiting for release confirmation...")

            ans_release = input("\nPress 'y' to release the object, 'n' to hold: ")
            print()
            if ans_release.lower() == 'y':
                self.send_gripper_command(0.085)
                self.get_logger().info("Object released.")

        except Exception as e:
            self.get_logger().error(f"Grasp execution failed: {e}")


global ROS_NODE
ROS_NODE = None

drawing = False
ix, iy, fx_mouse, fy_mouse = -1, -1, -1, -1
pending_bbox = None
pending_point = None
num_targets = 0
need_reset = False
current_masks = {}
all_prompts = []


def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, fx_mouse, fy_mouse, pending_bbox, pending_point

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        fx_mouse, fy_mouse = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        fx_mouse, fy_mouse = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx_mouse, fy_mouse = x, y
        dx = abs(fx_mouse - ix)
        dy = abs(fy_mouse - iy)
        if dx > 8 and dy > 8:
            pending_bbox = (
                min(ix, fx_mouse),
                min(iy, fy_mouse),
                max(ix, fx_mouse),
                max(iy, fy_mouse)
            )
        else:
            pending_point = (x, y)


def get_net(checkpoint_path, num_view=300):
    net = GraspNet(
        input_feature_dim=0,
        num_view=num_view,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net, device


def load_ffs_model():
    logging.info("Loading FFS model...")
    with open(os.path.join(os.path.dirname(MODEL_DIR), "cfg.yaml"), 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['valid_iters'] = VALID_ITERS
    cfg['max_disp'] = MAX_DISP

    model = torch.load(MODEL_DIR, map_location='cpu', weights_only=False)
    model.args.valid_iters = VALID_ITERS
    model.args.max_disp = MAX_DISP
    model.cuda().eval()
    logging.info("FFS model loaded")
    return model


def warmup_ffs_model(model):
    logging.info("Warming up FFS model...")
    dummy_left = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).cuda().float()
    dummy_right = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).cuda().float()
    padder = InputPadder(dummy_left.shape, divis_by=32, force_square=False)
    dummy_left_p, dummy_right_p = padder.pad(dummy_left, dummy_right)
    with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
        _ = model.forward(
            dummy_left_p,
            dummy_right_p,
            iters=VALID_ITERS,
            test_mode=True,
            optimize_build_volume='pytorch1'
        )
    del dummy_left, dummy_right, dummy_left_p, dummy_right_p
    torch.cuda.empty_cache()
    logging.info("FFS warm-up complete")


def start_zed():
    logging.info("Initializing ZED 2i camera...")
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

    fx = left_cam.fx * SCALE_FACTOR
    fy = left_cam.fy * SCALE_FACTOR
    cx = left_cam.cx * SCALE_FACTOR
    cy = left_cam.cy * SCALE_FACTOR

    k = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    camera = CameraInfo(float(IMG_WIDTH), float(IMG_HEIGHT), fx, fy, cx, cy, 1000.0)
    logging.info(f"Using optical TF frame: {CAMERA_FRAME}")
    return zed, k, baseline, camera


def infer_depth_with_ffs(model, left_rgb, right_rgb, k, baseline):
    img0 = torch.as_tensor(left_rgb.copy()).cuda().float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(right_rgb.copy()).cuda().float()[None].permute(0, 3, 1, 2)

    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0_p, img1_p = padder.pad(img0, img1)

    with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
        disp = model.forward(
            img0_p,
            img1_p,
            iters=VALID_ITERS,
            test_mode=True,
            optimize_build_volume='pytorch1'
        )

    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(IMG_HEIGHT, IMG_WIDTH).clip(0, None)
    xx = np.arange(IMG_WIDTH)[None, :].repeat(IMG_HEIGHT, axis=0)
    invalid = (xx - disp) < 0
    disp[invalid] = np.inf

    depth_m = k[0, 0] * baseline / disp
    depth_m[(depth_m < ZNEAR) | (depth_m > ZFAR) | ~np.isfinite(depth_m)] = 0
    depth_mm = (depth_m * 1000.0).astype(np.uint16)
    return depth_mm, disp


def add_colorbar_and_text(vis_bgr, vmin, vmax, title, invalid_mask=None):
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
    cv2.putText(canvas, f"max: {vmax:.3f}", (bar_x1 + 8, top_pad + 14), font, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"min: {vmin:.3f}", (bar_x1 + 8, top_pad + h - 6), font, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
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

    vis = add_colorbar_and_text(vis, dmin, dmax, "Depth (m)", invalid_mask=~valid)
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


def process_frame(color_img, depth_img, camera, device, sam_mask=None, num_point=20000):
    color = color_img.astype(np.float32) / 255.0

    if sam_mask is not None and np.any(sam_mask):
        workspace_mask = (sam_mask > 0) & (depth_img > 0) & (depth_img < 2000)
    else:
        z_min_mm, z_max_mm = 200, 1000
        workspace_mask = (depth_img > z_min_mm) & (depth_img < z_max_mm)

    cloud = create_point_cloud_from_depth_image(depth_img, camera, organized=True)
    mask = workspace_mask & (depth_img > 0)
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
        idxs2 = np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True)
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

    end_points = {}
    end_points['point_clouds'] = torch.from_numpy(
        cloud_sampled[np.newaxis].astype(np.float32)
    ).to(device, dtype=torch.float32)
    end_points['cloud_colors'] = color_sampled
    return end_points, cloud_o3d, workspace_mask


def show_open3d_process(points, colors, gg_array, top_k=1):
    import open3d as o3d
    from graspnetAPI import GraspGroup

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    gg = GraspGroup(gg_array).nms().sort_by_score()
    view_kwargs = {"front": [0, 0, -1], "lookat": [0, 0, 0.5], "up": [0, -1, 0], "zoom": 0.8}

    if len(gg) > 0:
        show_n = min(len(gg), max(int(top_k), 1))
        grippers = gg[:show_n].to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers], **view_kwargs)
    else:
        o3d.visualization.draw_geometries([cloud], **view_kwargs)


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
    before_path = os.path.join(save_dir, 'open3d_before_filter.png')
    after_path = os.path.join(save_dir, 'open3d_after_filter.png')

    try:
        _render_open3d_snapshot(
            points, colors, gg_before_array, before_path, 'Open3D Before Filter', top_k=OPEN3D_BEFORE_TOP_K
        )
        _render_open3d_snapshot(
            points, colors, gg_after_array, after_path, 'Open3D After Filter', top_k=OPEN3D_AFTER_TOP_K
        )

        img_before = cv2.imread(before_path)
        img_after = cv2.imread(after_path)
        if img_before is not None and img_after is not None and img_before.shape == img_after.shape:
            cv2.imwrite(os.path.join(save_dir, 'open3d_filter_compare.png'), cv2.hconcat([img_before, img_after]))
    except Exception as e:
        logging.warning(f"Open3D screenshot save failed: {e}")

def show_open3d_process_named(points, colors, gg_array, window_name, top_k=1):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    gg = GraspGroup(gg_array).nms().sort_by_score()
    view_kwargs = {
        "window_name": window_name,
        "front": [0, 0, -1],
        "lookat": [0, 0, 0.5],
        "up": [0, -1, 0],
        "zoom": 0.8,
    }

    if len(gg) > 0:
        show_n = min(len(gg), max(int(top_k), 1))
        grippers = gg[:show_n].to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers], **view_kwargs)
    else:
        o3d.visualization.draw_geometries([cloud], **view_kwargs)


def estimate_workspace_normal(scene_points):
    default_normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    if len(scene_points) < 100:
        return default_normal

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene_points.astype(np.float64))

    try:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.01,
            ransac_n=3,
            num_iterations=1000
        )
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


def filter_grasps_by_camera_axis(gg):
    if len(gg) == 0:
        return gg

    camera_forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    max_tilt_cos = math.cos(math.radians(CAMERA_AXIS_MAX_TILT_DEG))

    keep_mask = []
    for grasp in gg:
        rot_mat = grasp.rotation_matrix
        approach_camera = rot_mat[:, 0]

        approach_camera = approach_camera / (np.linalg.norm(approach_camera) + 1e-8)
        cos_tilt = float(np.clip(np.dot(approach_camera, camera_forward), -1.0, 1.0))
        keep_mask.append(abs(cos_tilt) >= max_tilt_cos)

    keep_mask = np.asarray(keep_mask, dtype=bool)
    return gg[keep_mask]


def draw_grasp_projection(img_bgr, gg, camera, title):
    canvas = img_bgr.copy()
    cv2.putText(canvas, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 255, 60), 2, cv2.LINE_AA)
    count = min(len(gg), 20)
    for i in range(count):
        g = gg[i]
        t = g.translation
        if t[2] <= 1e-6:
            continue
        p0 = t
        p1 = t + 0.06 * g.rotation_matrix[:, 0]

        u0 = int(camera.fx * (p0[0] / p0[2]) + camera.cx)
        v0 = int(camera.fy * (p0[1] / p0[2]) + camera.cy)
        u1 = int(camera.fx * (p1[0] / p1[2]) + camera.cx)
        v1 = int(camera.fy * (p1[1] / p1[2]) + camera.cy)
        if not (0 <= u0 < IMG_WIDTH and 0 <= v0 < IMG_HEIGHT):
            continue

        color = (0, 220, 255) if i == 0 else (255, 170, 0)
        cv2.circle(canvas, (u0, v0), 3, color, -1, cv2.LINE_AA)
        cv2.arrowedLine(canvas, (u0, v0), (u1, v1), color, 2, cv2.LINE_AA, tipLength=0.25)

    cv2.putText(canvas, f"count={len(gg)}", (10, IMG_HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main():
    global need_reset, num_targets, pending_bbox, pending_point
    global current_masks, all_prompts, drawing, ix, iy, fx_mouse, fy_mouse, ROS_NODE

    rclpy.init()
    ROS_NODE = RobotControllerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(ROS_NODE)
    t_ros = threading.Thread(target=executor.spin, daemon=True)
    t_ros.start()

    if ROS_NODE.rtde_c:
        ROS_NODE.get_logger().info("【初始化】移动机械臂到初始观测位置...")
        start_joint_q = [0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0]
        ROS_NODE.rtde_c.moveJ(start_joint_q, 0.5, 0.5)
        time.sleep(1.0)
        ROS_NODE.get_logger().info("【初始化】释放夹爪...")
        success = ROS_NODE.send_gripper_command(0.085)
        if not success:
            ROS_NODE.get_logger().warning("夹爪打开指令发送失败或超时，请检查 Action Server")
        ROS_NODE.get_logger().info("【初始化完成】机械臂已就位，启动相机流...")

    checkpoint_path = 'logs/log_rs/checkpoint-rs.tar'
    if not os.path.exists(checkpoint_path) and os.path.exists('logs/log_rs/checkpoint.tar'):
        checkpoint_path = 'logs/log_rs/checkpoint.tar'
    net, device = get_net(checkpoint_path)

    sam2_checkpoint = os.path.join(SAM2_DIR, "checkpoints/sam2.1/sam2.1_hiera_small.pt")
    sam2_cfg = "sam2.1/sam2.1_hiera_s.yaml"
    logging.info("Loading SAM2 model...")
    sam2_predictor = build_sam2_camera_predictor(sam2_cfg, sam2_checkpoint)
    sam2_predictor.fill_hole_area = 0
    logging.info("SAM2 model loaded")

    ffs_model = load_ffs_model()
    warmup_ffs_model(ffs_model)
    zed, k_scaled, baseline, camera_info = start_zed()
    image_left = sl.Mat()
    image_right = sl.Mat()

    cv2.namedWindow("ZED2i Viewer", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("ZED2i Viewer", mouse_callback)

    logging.info("Camera Started. UI Controls (focus on OpenCV window):")
    logging.info("  - Left-click drag: Draw bounding box -> initialize tracking")
    logging.info("  - Left-click: Select foreground point -> initialize tracking")
    logging.info("  - Space: Capture and predict grasp for targeted object(s)")
    logging.info("  - r: Reset SAM2 selection")
    logging.info("  - q: Quit")

    try:
        while True:
            t0 = time.time()
            if zed.grab() != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            zed.retrieve_image(image_right, sl.VIEW.RIGHT)

            left_bgr_full = image_left.get_data()[:, :, :3]
            right_bgr_full = image_right.get_data()[:, :, :3]
            left_bgr = cv2.resize(left_bgr_full, (IMG_WIDTH, IMG_HEIGHT))
            right_bgr = cv2.resize(right_bgr_full, (IMG_WIDTH, IMG_HEIGHT))

            color_img = left_bgr[:, :, ::-1].copy()
            right_img = right_bgr[:, :, ::-1].copy()
            tracking_img = left_bgr.copy()

            if need_reset:
                if hasattr(sam2_predictor, 'condition_state') and 'point_inputs_per_obj' in sam2_predictor.condition_state:
                    sam2_predictor.reset_state()
                num_targets = 0
                need_reset = False
                pending_bbox = None
                pending_point = None
                current_masks = {}
                all_prompts = []
                logging.info("Reset, select new targets (up to 2)")

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
                for prompt in all_prompts:
                    target_id = prompt['id']
                    if 'bbox' in prompt:
                        x1, y1, x2, y2 = prompt['bbox']
                        bbox_arr = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                        sam2_predictor.add_new_prompt(frame_idx=0, obj_id=target_id, bbox=bbox_arr)
                    elif 'point' in prompt:
                        px, py = prompt['point']
                        pts_arr = np.array([[px, py]], dtype=np.float32)
                        lbl_arr = np.array([1], dtype=np.int32)
                        sam2_predictor.add_new_prompt(frame_idx=0, obj_id=target_id, points=pts_arr, labels=lbl_arr)

            if num_targets > 0:
                out_obj_ids, out_mask_logits = sam2_predictor.track(tracking_img)
                current_masks = {}
                for i in range(len(out_obj_ids)):
                    obj_id = out_obj_ids[i]
                    current_masks[obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()

            display = tracking_img.copy()
            for obj_id, mask in current_masks.items():
                if mask is not None:
                    color = MASK_COLORS_BGR.get(obj_id, [0, 255, 0])
                    overlay = display.copy()
                    overlay[mask > 0] = color
                    display = cv2.addWeighted(display, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

            if drawing and ix >= 0:
                cv2.rectangle(display, (ix, iy), (fx_mouse, fy_mouse), (255, 200, 0), 2)

            fps = 1.0 / max(time.time() - t0, 1e-6)
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            status = f"TRACKING {num_targets}/2 | SPACE=Predict | r=reset | q=quit" if num_targets > 0 else "Select targets | SPACE=Predict | q=quit"
            cv2.putText(display, status, (10, IMG_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("ZED2i Viewer", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                print("Processing ZED2i frame for GraspNet...")
                combined_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=bool)
                for mask in current_masks.values():
                    if mask is not None:
                        combined_mask |= (mask > 0)

                if not np.any(combined_mask):
                    print("No SAM mask found, using default workspace heuristic!")
                    combined_mask = None

                def process_grasp(left_rgb, right_rgb, cam_info, dev, c_mask, tracking_bgr):
                    depth_img, disp = infer_depth_with_ffs(ffs_model, left_rgb, right_rgb, k_scaled, baseline)
                    end_points, cloud_o3d, workspace_mask = process_frame(
                        left_rgb, depth_img, cam_info, dev, sam_mask=c_mask
                    )
                    if end_points is None:
                        print("No points found in workspace mask.")
                        return

                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_dir = os.path.join("captured_data", timestamp)
                    os.makedirs(save_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(save_dir, 'sam2_input.png'), tracking_bgr)
                    cv2.imwrite(os.path.join(save_dir, 'left_rgb.png'), cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_dir, 'depth_raw_mm.png'), depth_img)
                    save_disparity_vis(disp, os.path.join(save_dir, 'disparity.png'))
                    save_depth_vis(depth_img.astype(np.float32) / 1000.0, os.path.join(save_dir, 'depth.png'))

                    if c_mask is not None:
                        sam_mask_u8 = (c_mask.astype(np.uint8) * 255)
                        cv2.imwrite(os.path.join(save_dir, 'sam2_mask.png'), sam_mask_u8)
                        sam_overlay = tracking_bgr.copy()
                        ov = sam_overlay.copy()
                        ov[c_mask > 0] = (60, 200, 240)
                        sam_overlay = cv2.addWeighted(sam_overlay, 0.65, ov, 0.35, 0)
                        cv2.imwrite(os.path.join(save_dir, 'sam2_overlay.png'), sam_overlay)

                    cv2.imwrite(os.path.join(save_dir, 'workspace_mask_used.png'), workspace_mask.astype(np.uint8) * 255)

                    full_cloud = create_point_cloud_from_depth_image(depth_img, cam_info, organized=True)
                    full_valid = depth_img > 0
                    full_points = full_cloud[full_valid]
                    full_colors = (left_rgb.astype(np.float32) / 255.0)[full_valid]
                    save_pointcloud_xyzrgb(full_points, full_colors, os.path.join(save_dir, 'pointcloud_full.ply'))

                    if c_mask is not None:
                        sam_valid = (c_mask > 0) & (depth_img > 0)
                        sam_points = full_cloud[sam_valid]
                        sam_colors = (left_rgb.astype(np.float32) / 255.0)[sam_valid]
                        save_pointcloud_xyzrgb(sam_points, sam_colors, os.path.join(save_dir, 'pointcloud_sam_masked.ply'))

                    used_valid = workspace_mask & (depth_img > 0)
                    used_points = full_cloud[used_valid]
                    used_colors = (left_rgb.astype(np.float32) / 255.0)[used_valid]
                    save_pointcloud_xyzrgb(used_points, used_colors, os.path.join(save_dir, 'pointcloud_workspace_used.ply'))
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
                        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float32) if dev.type == "cuda" else nullcontext()
                        with amp_ctx:
                            end_points = net(end_points)
                            grasp_preds = pred_decode(end_points)

                    gg_array = grasp_preds[0].detach().cpu().numpy()
                    gg_ranked = GraspGroup(gg_array).nms().sort_by_score()
                    raw_count = len(gg_ranked)
                    gg_filtered = filter_grasps_by_camera_axis(gg_ranked).sort_by_score()
                    axis_count = len(gg_filtered)

                    if len(gg_filtered) == 0 and raw_count > 0:
                        gg_filtered = gg_ranked[:1]
                        logging.warning("No valid grasp after camera-axis filtering; fallback to raw top1.")

                    gg_filtered = gg_filtered.sort_by_score()

                    vis_before = draw_grasp_projection(
                        tracking_bgr,
                        gg_ranked,
                        cam_info,
                        "Before filter (NMS+score)"
                    )
                    vis_after = draw_grasp_projection(
                        tracking_bgr,
                        gg_filtered,
                        cam_info,
                        "After filter (camera-axis approach)"
                    )
                    cv2.imwrite(os.path.join(save_dir, 'grasp_before_filter.png'), vis_before)
                    cv2.imwrite(os.path.join(save_dir, 'grasp_after_filter.png'), vis_after)
                    cv2.imwrite(os.path.join(save_dir, 'grasp_filter_compare.png'), cv2.hconcat([vis_before, vis_after]))

                    logging.info(
                        "Grasp filtering: "
                        f"raw={raw_count}, axis_filtered={axis_count}, selected={len(gg_filtered)}, "
                        f"rule=abs(dot(approach,cam_z))>=cos({CAMERA_AXIS_MAX_TILT_DEG:.1f}deg)"
                    )

                    gg_before_array = gg_ranked.grasp_group_array if len(gg_ranked) > 0 else np.empty((0, 17), dtype=np.float32)
                    gg_after_array = gg_filtered.grasp_group_array if len(gg_filtered) > 0 else np.empty((0, 17), dtype=np.float32)

                    points_np = np.asarray(cloud_o3d.points)
                    colors_np = np.asarray(cloud_o3d.colors)

                    save_open3d_compare_images(points_np, colors_np, gg_before_array, gg_after_array, save_dir)

                    p_before = multiprocessing.Process(
                        target=show_open3d_process_named,
                        args=(points_np, colors_np, gg_before_array, "Open3D Before Filter", OPEN3D_BEFORE_TOP_K),
                    )
                    p_before.daemon = True
                    p_before.start()

                    p_after = multiprocessing.Process(
                        target=show_open3d_process_named,
                        args=(points_np, colors_np, gg_after_array, "Open3D After Filter", OPEN3D_AFTER_TOP_K),
                    )
                    p_after.daemon = True
                    p_after.start()

                    if len(gg_filtered) > 0 and ROS_NODE:
                        ROS_NODE.execute_grasp(gg_filtered, viz_process=p_after)
                    elif len(gg_filtered) == 0:
                        logging.warning("No valid grasp remains after camera-axis filtering.")

                t = threading.Thread(
                    target=process_grasp,
                    args=(
                        color_img.copy(),
                        right_img.copy(),
                        camera_info,
                        device,
                        combined_mask.copy() if combined_mask is not None else None,
                        tracking_img.copy(),
                    ),
                    daemon=True
                )
                t.start()
            elif key == ord('r'):
                need_reset = True
            elif key == ord('q'):
                break

    finally:
        zed.close()
        cv2.destroyAllWindows()
        if ROS_NODE and ROS_NODE.rtde_c:
            ROS_NODE.rtde_c.stopScript()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
