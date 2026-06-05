"""
Dual RealSense RGB + Fast-FoundationStereo vs RealSense Raw Depth Comparison

This script captures frames from left and right RealSense cameras.
It displays two Open3D windows simultaneously:
1. FFS Point Cloud (calculated from left and right RGB, using CALIB_FILE).
2. RealSense Raw Point Cloud (hardware depth from the left camera).

Usage:
  conda activate ffs
  python dual_realsense_compare.py
"""

import os, sys, time, logging
import numpy as np
import torch
import yaml
import cv2
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
CALIB_FILE = os.path.join(FFS_DIR, "calibration_realsense", "stereo_calib_rs.yaml")
VALID_ITERS = 8
MAX_DISP = 192
ZFAR = 5.0
ZNEAR = 0.05
IMG_WIDTH = 640
IMG_HEIGHT = 480
PCD_STRIDE = 1

# Denoising parameters
DENOISE_VOXEL_SIZE = 0.001
DENOISE_NB_POINTS = 30
DENOISE_RADIUS = 0.03

# ===== 1. Load calibration parameters =====
logging.info("Loading calibration parameters...")
if not os.path.exists(CALIB_FILE):
    raise FileNotFoundError(f"Calibration file not found at {CALIB_FILE}!")

with open(CALIB_FILE, 'r') as f:
    calib = yaml.safe_load(f)

K_l, K_r = np.array(calib['K_l']), np.array(calib['K_r'])
dist_l, dist_r = np.array(calib['dist_l']), np.array(calib['dist_r'])
R_l, R_r = np.array(calib['R_l']), np.array(calib['R_r'])
P_l, P_r = np.array(calib['P_l']), np.array(calib['P_r'])
baseline = float(calib['baseline'])

fx_ffs, fy_ffs = P_l[0, 0], P_l[1, 1]
cx_ffs, cy_ffs = P_l[0, 2], P_l[1, 2]

logging.info(f"Baseline: {baseline*1000:.2f}mm. Rectified focal: fx={fx_ffs:.1f}, fy={fy_ffs:.1f}")

# Compute rectification maps
map_lx, map_ly = cv2.initUndistortRectifyMap(K_l, dist_l, R_l, P_l, (IMG_WIDTH, IMG_HEIGHT), cv2.CV_32FC1)
map_rx, map_ry = cv2.initUndistortRectifyMap(K_r, dist_r, R_r, P_r, (IMG_WIDTH, IMG_HEIGHT), cv2.CV_32FC1)

u_grid, v_grid = np.meshgrid(np.arange(0, IMG_WIDTH, PCD_STRIDE), np.arange(0, IMG_HEIGHT, PCD_STRIDE))
u_flat, v_flat = u_grid.flatten().astype(np.float32), v_grid.flatten().astype(np.float32)

# ===== 2. Load FFS model =====
logging.info("Loading FFS model...")
torch.autograd.set_grad_enabled(False)
model = torch.load(MODEL_DIR, map_location='cpu', weights_only=False)
model.args.valid_iters = VALID_ITERS
model.args.max_disp = MAX_DISP
model.cuda().eval()

# ===== 3. Initialize Two RealSense Cameras =====
ctx = rs.context()
devices = ctx.query_devices()
assert len(devices) >= 2, "Failed to find at least two RealSense devices."

# 尝试读取标定阶段记录的序列号自动分配左右
SN_FILE = os.path.join(FFS_DIR, "calibration_realsense", "stereo_sn_rs.yaml")
if os.path.exists(SN_FILE):
    with open(SN_FILE, 'r') as f:
        sn_data = yaml.safe_load(f)
    serial_left = sn_data.get('serial_left')
    serial_right = sn_data.get('serial_right')
    logging.info(f"Loaded serial numbers from config: Left={serial_left}, Right={serial_right}")
else:
    logging.warning("No stereo_sn_rs.yaml found! Falling back to USB enum order.")
    serial_left = devices[0].get_info(rs.camera_info.serial_number)
    serial_right = devices[1].get_info(rs.camera_info.serial_number)

pipeline_left = rs.pipeline()
c_left = rs.config()
c_left.enable_device(serial_left)
c_left.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, 30)
# Enable depth on left camera to compare
c_left.enable_stream(rs.stream.depth, IMG_WIDTH, IMG_HEIGHT, rs.format.z16, 30)
left_profile = pipeline_left.start(c_left)

pipeline_right = rs.pipeline()
c_right = rs.config()
c_right.enable_device(serial_right)
c_right.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, 30)
pipeline_right.start(c_right)
logging.info(f"Cameras initialized: L({serial_left}) R({serial_right})")

# Align left depth to left color
align_to = rs.stream.color
align = rs.align(align_to)
depth_scale = left_profile.get_device().first_depth_sensor().get_depth_scale()

# Get hardware intrinsics of left camera for RealSense projection
frames_l = pipeline_left.wait_for_frames()
color_l_frame = align.process(frames_l).get_color_frame()
color_profile = color_l_frame.get_profile().as_video_stream_profile()
c_intrinsics = color_profile.get_intrinsics()
fx_rs, fy_rs = c_intrinsics.fx, c_intrinsics.fy
cx_rs, cy_rs = c_intrinsics.ppx, c_intrinsics.ppy

# ===== 4. Open3D visualizers (Two separate windows) =====
vis_ffs = o3d.visualization.Visualizer()
vis_ffs.create_window("FFS Denoised Point Cloud", width=800, height=600, left=50, top=50)
vis_ffs.get_render_option().point_size = 2.0
vis_ffs.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
pcd_ffs = o3d.geometry.PointCloud()
vis_ffs.add_geometry(pcd_ffs)
first_frame_ffs = True

vis_rs = o3d.visualization.Visualizer()
vis_rs.create_window("RealSense Hardware Point Cloud", width=800, height=600, left=860, top=50)
vis_rs.get_render_option().point_size = 2.0
vis_rs.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
pcd_rs = o3d.geometry.PointCloud()
vis_rs.add_geometry(pcd_rs)
first_frame_rs = True

cv2.namedWindow("Dual RealSense Preview", cv2.WINDOW_AUTOSIZE)

# ===== 5. Main loop =====
logging.info("Ready! Select 'Dual RealSense Preview' window and press SPACE to capture and compare. Press ESC to exit.")
try:
    while True:
        # Keep both Open3D windows responsive
        vis_ffs.poll_events()
        vis_ffs.update_renderer()
        vis_rs.poll_events()
        vis_rs.update_renderer()

        frames_left = pipeline_left.wait_for_frames()
        frames_right = pipeline_right.wait_for_frames()
        
        # Align left frames to get matching color and depth
        aligned_left = align.process(frames_left)
        color_frame_left = aligned_left.get_color_frame()
        depth_frame_left = aligned_left.get_depth_frame()
        color_frame_right = frames_right.get_color_frame()
        
        if not color_frame_left or not color_frame_right or not depth_frame_left: continue

        raw_left = np.asanyarray(color_frame_left.get_data())
        raw_right = np.asanyarray(color_frame_right.get_data())
        depth_left = np.asanyarray(depth_frame_left.get_data())

        cv2.imshow("Dual RealSense Preview", raw_left)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == 32 or key == ord('c'):
            logging.info("Processing comparison...")
            t0 = time.time()

            # ----------------------------------------------------
            # 1. RealSense Hardware Point Cloud
            # ----------------------------------------------------
            depth_meters = depth_left * depth_scale
            valid_rs = (depth_meters > ZNEAR) & (depth_meters < ZFAR)
            z_rs = depth_meters[valid_rs]
            u_rs = u_grid[valid_rs]
            v_rs = v_grid[valid_rs]
            
            x_rs = (u_rs - cx_rs) * z_rs / fx_rs
            y_rs = (v_rs - cy_rs) * z_rs / fy_rs
            pts_rs = np.stack([x_rs, y_rs, z_rs], axis=-1)
            colors_rs = raw_left[v_rs.astype(int), u_rs.astype(int), ::-1].astype(np.float64) / 255.0
            
            pcd_rs.points = o3d.utility.Vector3dVector(pts_rs)
            pcd_rs.colors = o3d.utility.Vector3dVector(colors_rs)
            
            if first_frame_rs:
                vis_rs.reset_view_point(True)
                ctr_rs = vis_rs.get_view_control()
                ctr_rs.set_front([0, 0, -1])
                ctr_rs.set_up([0, -1, 0])
                first_frame_rs = False
                
            vis_rs.update_geometry(pcd_rs)


            # ----------------------------------------------------
            # 2. Fast-FoundationStereo Point Cloud
            # ----------------------------------------------------
            rect_left = cv2.remap(raw_left, map_lx, map_ly, cv2.INTER_LINEAR)
            rect_right = cv2.remap(raw_right, map_rx, map_ry, cv2.INTER_LINEAR)

            H, W = rect_left.shape[:2]
            img0 = torch.as_tensor(rect_left[:, :, ::-1].copy()).cuda().float()[None].permute(0, 3, 1, 2)
            img1 = torch.as_tensor(rect_right[:, :, ::-1].copy()).cuda().float()[None].permute(0, 3, 1, 2)
            
            padder = InputPadder(img0.shape, divis_by=32, force_square=False)
            img0_p, img1_p = padder.pad(img0, img1)

            with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
                disp = model.forward(img0_p, img1_p, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
            
            disp = padder.unpad(disp.float())
            disp = disp.data.cpu().numpy().reshape(H, W).clip(0, None)

            xx = np.arange(W)[None, :].repeat(H, axis=0)
            disp[(xx - disp) < 0] = np.inf

            depth_ffs = fx_ffs * baseline / disp
            depth_ffs[(depth_ffs < ZNEAR) | (depth_ffs > ZFAR) | ~np.isfinite(depth_ffs)] = 0

            depth_ds = depth_ffs[::PCD_STRIDE, ::PCD_STRIDE]
            z_flat = depth_ds.reshape(-1)
            valid_mask = z_flat > 0

            z_f = z_flat[valid_mask]
            u_f = u_flat[valid_mask]
            v_f = v_flat[valid_mask]

            x3d_f = (u_f - cx_ffs) * z_f / fx_ffs
            y3d_f = (v_f - cy_ffs) * z_f / fy_ffs
            pts_ffs = np.stack([x3d_f, y3d_f, z_f], axis=-1)

            colors_ffs = rect_left[v_f.astype(int), u_f.astype(int), ::-1].astype(np.float64) / 255.0

            if len(pts_ffs) > 0:
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(pts_ffs.astype(np.float64))
                temp_pcd.colors = o3d.utility.Vector3dVector(colors_ffs)

                # Apply Denoise to FFS point cloud
                temp_pcd = temp_pcd.voxel_down_sample(voxel_size=DENOISE_VOXEL_SIZE)
                cl, ind = temp_pcd.remove_radius_outlier(nb_points=DENOISE_NB_POINTS, radius=DENOISE_RADIUS)
                temp_pcd = temp_pcd.select_by_index(ind)

                pcd_ffs.points = temp_pcd.points
                pcd_ffs.colors = temp_pcd.colors
            else:
                pcd_ffs.points = o3d.utility.Vector3dVector()

            if first_frame_ffs:
                vis_ffs.reset_view_point(True)
                ctr_ffs = vis_ffs.get_view_control()
                ctr_ffs.set_front([0, 0, -1])
                ctr_ffs.set_up([0, -1, 0])
                first_frame_ffs = False

            vis_ffs.update_geometry(pcd_ffs)
            
            t1 = time.time()
            logging.info(f"Comparison mapped! Time: {t1-t0:.3f}s. Left Open3D = FFS, Right Open3D = RealSense.")

except KeyboardInterrupt:
    pass
finally:
    pipeline_left.stop()
    pipeline_right.stop()
    cv2.destroyAllWindows()
    vis_ffs.destroy_window()
    vis_rs.destroy_window()
    logging.info("Exited")
