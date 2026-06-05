"""
RealSense D435i + Fast-FoundationStereo vs Native Depth Point Cloud Comparison

Features:
  - Press SPACE or 'c' to capture a frame.
  - Generates two point clouds:
      1. Native RealSense depth sensor point cloud
      2. Fast-FoundationStereo predicted depth point cloud (Denoised)
  - Displays both point clouds side-by-side in two separate Open3D windows.
  - Press ESC or 'q' to exit.
"""

import os, sys, time, logging
import numpy as np
import torch
import yaml
import pyrealsense2 as rs
import open3d as o3d
import cv2

# Add FFS path
FFS_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(FFS_DIR)
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ===== Parameters =====
MODEL_DIR = os.path.join(FFS_DIR, "weights/23-36-37/model_best_bp2_serialize.pth")
VALID_ITERS = 8
MAX_DISP = 192
ZFAR = 5.0            # Max depth (meters)
ZNEAR = 0.1           # D435i min depth ~0.1m
IR_PROJECTOR_ON = True
IMG_WIDTH = 640
IMG_HEIGHT = 480

# Denoising parameters
DENOISE_VOXEL_SIZE = 0.001
DENOISE_NB_POINTS = 30
DENOISE_RADIUS = 0.03

# ===== 1. Load FFS model =====
logging.info("Loading FFS model...")
torch.autograd.set_grad_enabled(False)

if not os.path.exists(MODEL_DIR):
    ALT_DIR = os.path.abspath(os.path.join(FFS_DIR, "../Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth"))
    if os.path.exists(ALT_DIR):
        MODEL_DIR = ALT_DIR

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

# Enable streams
config.enable_stream(rs.stream.infrared, 1, IMG_WIDTH, IMG_HEIGHT, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, IMG_WIDTH, IMG_HEIGHT, rs.format.y8, 30)
config.enable_stream(rs.stream.depth, IMG_WIDTH, IMG_HEIGHT, rs.format.z16, 30)
config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, 30)

profile = pipeline.start(config)

# Setup alignment (Align depth to color for native PC)
align_to = rs.stream.color
align = rs.align(align_to)
pc = rs.pointcloud()

device = profile.get_device()
depth_sensor = device.first_depth_sensor()
if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 1 if IR_PROJECTOR_ON else 0)

# ===== 3. Camera Intrinsics/Extrinsics for FFS =====
frames = pipeline.wait_for_frames()
ir_left_frame = frames.get_infrared_frame(1)
color_frame = frames.get_color_frame()

ir_left_profile = ir_left_frame.get_profile().as_video_stream_profile()
ir_intrinsics = ir_left_profile.get_intrinsics()
K_ir = np.array([
    [ir_intrinsics.fx, 0, ir_intrinsics.ppx],
    [0, ir_intrinsics.fy, ir_intrinsics.ppy],
    [0, 0, 1]
], dtype=np.float32)

color_profile = color_frame.get_profile().as_video_stream_profile()
color_intrinsics = color_profile.get_intrinsics()
K_color = np.array([
    [color_intrinsics.fx, 0, color_intrinsics.ppx],
    [0, color_intrinsics.fy, color_intrinsics.ppy],
    [0, 0, 1]
], dtype=np.float32)

ir_to_color_extrinsics = ir_left_profile.get_extrinsics_to(color_profile)
R_ir_to_color = np.array(ir_to_color_extrinsics.rotation).reshape(3, 3).astype(np.float32)
T_ir_to_color = np.array(ir_to_color_extrinsics.translation).astype(np.float32)

ir_right_frame = frames.get_infrared_frame(2)
ir_right_profile = ir_right_frame.get_profile().as_video_stream_profile()
ir_left_to_right = ir_left_profile.get_extrinsics_to(ir_right_profile)
baseline = abs(ir_left_to_right.translation[0])

fx_ir, fy_ir = K_ir[0, 0], K_ir[1, 1]
cx_ir, cy_ir = K_ir[0, 2], K_ir[1, 2]
u_grid, v_grid = np.meshgrid(np.arange(IMG_WIDTH), np.arange(IMG_HEIGHT))
u_flat = u_grid.reshape(-1).astype(np.float32)
v_flat = v_grid.reshape(-1).astype(np.float32)

# ===== 4. Warm up model =====
logging.info("Warming up model...")
dummy_left = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).cuda().float()
dummy_right = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).cuda().float()
padder = InputPadder(dummy_left.shape, divis_by=32, force_square=False)
dummy_left_p, dummy_right_p = padder.pad(dummy_left, dummy_right)
with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
    _ = model.forward(dummy_left_p, dummy_right_p, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
torch.cuda.empty_cache()

# ===== 5. Initializing Dual Visualizers =====
vis_ffs = o3d.visualization.Visualizer()
vis_ffs.create_window("FFS Denoised Point Cloud", width=800, height=600, left=50, top=50)
vis_ffs.get_render_option().point_size = 2.0
vis_ffs.get_render_option().background_color = np.array([1.0, 1.0, 1.0])
pcd_ffs = o3d.geometry.PointCloud()
vis_ffs.add_geometry(pcd_ffs)

vis_native = o3d.visualization.Visualizer()
vis_native.create_window("Native D435i Point Cloud", width=800, height=600, left=900, top=50)
vis_native.get_render_option().point_size = 2.0
vis_native.get_render_option().background_color = np.array([1.0, 1.0, 1.0])
pcd_native = o3d.geometry.PointCloud()
vis_native.add_geometry(pcd_native)

first_frame = True
cv2.namedWindow("D435i Preview", cv2.WINDOW_AUTOSIZE)

logging.info("Ready! Select 'D435i Preview' window and press SPACE to capture and process. Press ESC to exit.")

try:
    while True:
        vis_ffs.poll_events()
        vis_ffs.update_renderer()
        vis_native.poll_events()
        vis_native.update_renderer()

        frames = pipeline.wait_for_frames()
        # Align frames for native processing
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        ir_left = frames.get_infrared_frame(1)
        ir_right = frames.get_infrared_frame(2)

        if not color_frame or not depth_frame or not ir_left or not ir_right:
            continue
            
        color_bgr = np.asanyarray(color_frame.get_data())
        cv2.imshow("D435i Preview", color_bgr)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == 32 or key == ord('c'):
            logging.info("Processing frame...")
            t0 = time.time()
            
            # ==========================================
            # 1. Native RealSense Point Cloud Processing
            # ==========================================
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)
            
            # Extract coordinates and colors
            vtx = np.asanyarray(points.get_vertices())
            vtx = np.zeros((len(vtx), 3), dtype=np.float32)
            vtx[:, 0] = np.asanyarray(points.get_vertices(2))[:, 0] # x
            vtx[:, 1] = np.asanyarray(points.get_vertices(2))[:, 1] # y
            vtx[:, 2] = np.asanyarray(points.get_vertices(2))[:, 2] # z
            
            # Filter by Z limits
            valid_native = (vtx[:, 2] > ZNEAR) & (vtx[:, 2] < ZFAR) & np.isfinite(vtx[:, 2])
            
            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            colors_native = color_rgb.reshape(-1, 3).astype(np.float64) / 255.0

            pcd_native.points = o3d.utility.Vector3dVector(vtx[valid_native].astype(np.float64))
            pcd_native.colors = o3d.utility.Vector3dVector(colors_native[valid_native])

            # ==========================================
            # 2. FFS Point Cloud Processing
            # ==========================================
            ir_left_data = np.asanyarray(ir_left.get_data())
            ir_right_data = np.asanyarray(ir_right.get_data())
            H, W = ir_left_data.shape[:2]

            left_rgb = np.stack([ir_left_data] * 3, axis=-1)
            right_rgb = np.stack([ir_right_data] * 3, axis=-1)

            img0 = torch.as_tensor(left_rgb).cuda().float()[None].permute(0, 3, 1, 2)
            img1 = torch.as_tensor(right_rgb).cuda().float()[None].permute(0, 3, 1, 2)
            img0_p, img1_p = padder.pad(img0, img1)

            with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
                disp = model.forward(img0_p, img1_p, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
            disp = padder.unpad(disp.float())
            disp = disp.data.cpu().numpy().reshape(H, W).clip(0, None)

            xx = np.arange(W)[None, :].repeat(H, axis=0)
            invalid = (xx - disp) < 0
            disp[invalid] = np.inf

            depth = fx_ir * baseline / disp
            depth[(depth < ZNEAR) | (depth > ZFAR) | ~np.isfinite(depth)] = 0

            z_flat = depth.reshape(-1)
            valid_mask = z_flat > 0

            z = z_flat[valid_mask]
            u = u_flat[valid_mask]
            v = v_flat[valid_mask]

            x3d = (u - cx_ir) * z / fx_ir
            y3d = (v - cy_ir) * z / fy_ir
            pts_ir = np.stack([x3d, y3d, z], axis=-1)

            pts_color = (R_ir_to_color @ pts_ir.T).T + T_ir_to_color

            u_rgb = (K_color[0, 0] * pts_color[:, 0] / pts_color[:, 2] + K_color[0, 2]).astype(np.int32)
            v_rgb = (K_color[1, 1] * pts_color[:, 1] / pts_color[:, 2] + K_color[1, 2]).astype(np.int32)

            in_bounds = (u_rgb >= 0) & (u_rgb < W) & (v_rgb >= 0) & (v_rgb < H)
            color_rgb = color_bgr[:, :, ::-1] # BGR to RGB
            colors = np.zeros((len(z), 3), dtype=np.float64)
            colors[in_bounds] = color_rgb[v_rgb[in_bounds], u_rgb[in_bounds]].astype(np.float64) / 255.0

            final_valid = in_bounds & (colors.sum(axis=1) > 0)
            points_final = pts_ir[final_valid]
            colors_final = colors[final_valid]

            if len(points_final) > 0:
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(points_final.astype(np.float64))
                temp_pcd.colors = o3d.utility.Vector3dVector(colors_final)
                
                # Apply denoise to FFS
                temp_pcd = temp_pcd.voxel_down_sample(voxel_size=DENOISE_VOXEL_SIZE)
                cl, ind = temp_pcd.remove_radius_outlier(nb_points=DENOISE_NB_POINTS, radius=DENOISE_RADIUS)
                temp_pcd = temp_pcd.select_by_index(ind)

                pcd_ffs.points = temp_pcd.points
                pcd_ffs.colors = temp_pcd.colors
            else:
                logging.warning("No valid points from FFS!")

            # Update displays
            if first_frame:
                vis_ffs.reset_view_point(True)
                ctr_ffs = vis_ffs.get_view_control()
                ctr_ffs.set_front([0, 0, -1])
                ctr_ffs.set_up([0, -1, 0])
                
                vis_native.reset_view_point(True)
                ctr_native = vis_native.get_view_control()
                ctr_native.set_front([0, 0, -1])
                ctr_native.set_up([0, -1, 0])
                first_frame = False

            vis_ffs.update_geometry(pcd_ffs)
            vis_native.update_geometry(pcd_native)
            
            t1 = time.time()
            logging.info(f"Processing finished in {t1-t0:.3f}s. Native Points: {len(pcd_native.points)}, FFS Points: {len(pcd_ffs.points)}")

except KeyboardInterrupt:
    pass
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    vis_ffs.destroy_window()
    vis_native.destroy_window()
    logging.info("Exited")
