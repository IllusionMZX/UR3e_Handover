"""
RealSense D435i + Fast-FoundationStereo Real-time Depth Estimation and Color Point Cloud Visualization

IR left/right stereo pair → FFS inference → depth map → RGB-colored point cloud

Features:
  - IR projector provides texture, FFS learned prior provides high precision → complementary
  - RealSense SDK provides complete calibration parameters, precise RGB coloring
  - Optional IR projector toggle

Usage:
  conda activate ffs
  python d435i_ffs_realtime.py
"""

import os, sys, time, logging
import cv2
import numpy as np
import torch
import yaml
import pyrealsense2 as rs
import open3d as o3d

# Add FFS path
FFS_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(FFS_DIR)
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ===== Parameters =====
MODEL_DIR = os.path.join(FFS_DIR, "weights/23-36-37/model_best_bp2_serialize.pth")
VALID_ITERS = 8       # Accuracy=8, speed=4
MAX_DISP = 192
ZFAR = 5.0            # Max depth (meters)
ZNEAR = 0.11          # D435i min depth ~0.11m
IR_PROJECTOR_ON = True # Enable IR projector (recommended, adds texture to textureless surfaces)
IMG_WIDTH = 640
IMG_HEIGHT = 480

# ===== 1. Load FFS model =====
logging.info("Loading FFS model...")
torch.autograd.set_grad_enabled(False)

with open(os.path.join(os.path.dirname(MODEL_DIR), "cfg.yaml"), 'r') as f:
    cfg = yaml.safe_load(f)
cfg['valid_iters'] = VALID_ITERS
cfg['max_disp'] = MAX_DISP

model = torch.load(MODEL_DIR, map_location='cpu', weights_only=False)
model.args.valid_iters = VALID_ITERS
model.args.max_disp = MAX_DISP
model.cuda().eval()
logging.info("FFS model loaded")

# ===== 2. Initialize RealSense D435i =====
logging.info("Initializing RealSense D435i...")
pipeline = rs.pipeline()
config = rs.config()

# Enable IR left/right streams + RGB stream
config.enable_stream(rs.stream.infrared, 1, IMG_WIDTH, IMG_HEIGHT, rs.format.y8, 30)   # IR left
config.enable_stream(rs.stream.infrared, 2, IMG_WIDTH, IMG_HEIGHT, rs.format.y8, 30)   # IR right
config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, 30)       # RGB

profile = pipeline.start(config)

# IR projector control
device = profile.get_device()
depth_sensor = device.first_depth_sensor()
if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 1 if IR_PROJECTOR_ON else 0)
    logging.info(f"IR projector: {'ON' if IR_PROJECTOR_ON else 'OFF'}")

# ===== 3. Get camera intrinsics and extrinsics =====
# Wait for one frame to get profile
frames = pipeline.wait_for_frames()
ir_left_frame = frames.get_infrared_frame(1)
color_frame = frames.get_color_frame()

# IR left camera intrinsics
ir_left_profile = ir_left_frame.get_profile().as_video_stream_profile()
ir_intrinsics = ir_left_profile.get_intrinsics()
K_ir = np.array([
    [ir_intrinsics.fx, 0, ir_intrinsics.ppx],
    [0, ir_intrinsics.fy, ir_intrinsics.ppy],
    [0, 0, 1]
], dtype=np.float32)

# RGB camera intrinsics
color_profile = color_frame.get_profile().as_video_stream_profile()
color_intrinsics = color_profile.get_intrinsics()
K_color = np.array([
    [color_intrinsics.fx, 0, color_intrinsics.ppx],
    [0, color_intrinsics.fy, color_intrinsics.ppy],
    [0, 0, 1]
], dtype=np.float32)
D_color = np.array(color_intrinsics.coeffs, dtype=np.float32) # RGB distortion coefficients

# IR left → RGB extrinsics
ir_to_color_extrinsics = ir_left_profile.get_extrinsics_to(color_profile)
R_ir_to_color = np.array(ir_to_color_extrinsics.rotation).reshape(3, 3).astype(np.float32)
T_ir_to_color = np.array(ir_to_color_extrinsics.translation).astype(np.float32)  # Already in meters

# Baseline: IR left → IR right
ir_right_frame = frames.get_infrared_frame(2)
ir_right_profile = ir_right_frame.get_profile().as_video_stream_profile()
ir_left_to_right = ir_left_profile.get_extrinsics_to(ir_right_profile)
baseline = abs(ir_left_to_right.translation[0])  # Meters

logging.info(f"IR intrinsics K:\n{K_ir}")
logging.info(f"RGB intrinsics K:\n{K_color}")
logging.info(f"Baseline: {baseline*1000:.1f}mm")
logging.info(f"IR→RGB translation: [{T_ir_to_color[0]*1000:.1f}, {T_ir_to_color[1]*1000:.1f}, {T_ir_to_color[2]*1000:.1f}] mm")

# Pre-compute pixel grid
fx_ir, fy_ir = K_ir[0, 0], K_ir[1, 1]
cx_ir, cy_ir = K_ir[0, 2], K_ir[1, 2]
u_grid, v_grid = np.meshgrid(np.arange(IMG_WIDTH), np.arange(IMG_HEIGHT))
u_flat = u_grid.reshape(-1).astype(np.float32)
v_flat = v_grid.reshape(-1).astype(np.float32)

# ===== 4. Warm up model =====
logging.info("Warming up model (first inference will be slower)...")
dummy_left = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).cuda().float()
dummy_right = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).cuda().float()
padder = InputPadder(dummy_left.shape, divis_by=32, force_square=False)
dummy_left_p, dummy_right_p = padder.pad(dummy_left, dummy_right)
with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
    _ = model.forward(dummy_left_p, dummy_right_p, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
del dummy_left, dummy_right, dummy_left_p, dummy_right_p
torch.cuda.empty_cache()
logging.info("Warm-up complete")

# ===== 5. Open3D visualizer =====
vis = o3d.visualization.Visualizer()
vis.create_window("D435i + FFS Color Point Cloud", width=1280, height=720)
vis.get_render_option().point_size = 2.0
vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)
first_frame = True

# ===== 6. Main loop =====
logging.info("Starting real-time inference, press Ctrl+C to exit")
frame_count = 0

try:
    while True:
        t0 = time.time()

        # Capture frames
        frames = pipeline.wait_for_frames()
        ir_left = np.asanyarray(frames.get_infrared_frame(1).get_data())   # (H, W) uint8
        ir_right = np.asanyarray(frames.get_infrared_frame(2).get_data())  # (H, W) uint8
        color_bgr = np.asanyarray(frames.get_color_frame().get_data())     # (H, W, 3) uint8 BGR

        H, W = ir_left.shape[:2]

        # IR grayscale → 3-channel
        left_rgb = np.stack([ir_left] * 3, axis=-1)
        right_rgb = np.stack([ir_right] * 3, axis=-1)

        # Convert to tensor [1, 3, H, W]
        img0 = torch.as_tensor(left_rgb).cuda().float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(right_rgb).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0_p, img1_p = padder.pad(img0, img1)

        # FFS inference
        with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
            disp = model.forward(img0_p, img1_p, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H, W).clip(0, None)

        # Remove invisible regions
        xx = np.arange(W)[None, :].repeat(H, axis=0)
        invalid = (xx - disp) < 0
        disp[invalid] = np.inf

        # Disparity → depth
        depth = fx_ir * baseline / disp
        depth[(depth < ZNEAR) | (depth > ZFAR) | ~np.isfinite(depth)] = 0

        # ===== Edge / Occlusion Filtering =====
        # 计算深度的空间梯度，剔除深度剧烈跳变的边缘点（这些点在投影到RGB视角时极易引起遮挡和错位取色）
        depth_diff_x = np.abs(depth[:, :-1] - depth[:, 1:])
        depth_diff_y = np.abs(depth[:-1, :] - depth[1:, :])
        
        depth_diff = np.zeros_like(depth)
        depth_diff[:, :-1] = np.maximum(depth_diff[:, :-1], depth_diff_x)
        depth_diff[:-1, :] = np.maximum(depth_diff[:-1, :], depth_diff_y)
        
        # 深度跳变阈值设为 0.03米 (3厘米)，超过这个跳变的点被视为边缘飞点丢弃
        edge_mask = depth_diff < 0.03

        # ===== RGB coloring: per-pixel reprojection based on depth =====
        z_flat = depth.reshape(-1)
        edge_mask_flat = edge_mask.reshape(-1)
        
        # 只有在所需深度范围内，且不是突变边缘的点才保留
        valid_mask = (z_flat > 0) & edge_mask_flat

        z = z_flat[valid_mask]
        u = u_flat[valid_mask]
        v = v_flat[valid_mask]

        # IR pixel → 3D point (IR left coordinate system)
        x3d = (u - cx_ir) * z / fx_ir
        y3d = (v - cy_ir) * z / fy_ir
        pts_ir = np.stack([x3d, y3d, z], axis=-1)  # (N, 3)

        # IR coordinate system → RGB coordinate system
        pts_color = (R_ir_to_color @ pts_ir.T).T + T_ir_to_color  # (N, 3)

        # RGB coordinate system → RGB pixel WITH distortion handling
        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.zeros((3, 1), dtype=np.float32)
        img_pts, _ = cv2.projectPoints(pts_color, rvec, tvec, K_color, D_color)
        img_pts = img_pts.reshape(-1, 2)
        
        u_rgb = np.round(img_pts[:, 0]).astype(np.int32)
        v_rgb = np.round(img_pts[:, 1]).astype(np.int32)

        # Boundary check
        in_bounds = (u_rgb >= 0) & (u_rgb < W) & (v_rgb >= 0) & (v_rgb < H)

        # Sample colors from RGB image (BGR → RGB)
        colors = np.zeros((len(z), 3), dtype=np.float64)
        colors[in_bounds] = color_bgr[v_rgb[in_bounds], u_rgb[in_bounds], ::-1].astype(np.float64) / 255.0

        # Keep only points with valid colors
        final_valid = in_bounds & (colors.sum(axis=1) > 0)
        points_final = pts_ir[final_valid]
        colors_final = colors[final_valid]

        t1 = time.time()
        fps = 1.0 / (t1 - t0)

        # Update Open3D point cloud
        pcd.points = o3d.utility.Vector3dVector(points_final.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors_final)

        if first_frame:
            vis.reset_view_point(True)
            ctr = vis.get_view_control()
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, -1, 0])
            first_frame = False

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        frame_count += 1
        if frame_count % 30 == 0:
            logging.info(f"Frame {frame_count}, FPS: {fps:.1f}, points: {len(points_final)}")

        pass

except KeyboardInterrupt:
    pass
finally:
    pipeline.stop()
    vis.destroy_window()
    logging.info("Exited")