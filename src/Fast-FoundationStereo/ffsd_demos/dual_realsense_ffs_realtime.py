"""
Dual RealSense RGB + Fast-FoundationStereo Real-time Depth Estimation

Usage:
  conda activate ffs
  python dual_realsense_ffs_realtime.py
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
    raise FileNotFoundError(f"Calibration file not found at {CALIB_FILE}! Please run calibration_realsense scripts first.")

with open(CALIB_FILE, 'r') as f:
    calib = yaml.safe_load(f)

K_l, K_r = np.array(calib['K_l']), np.array(calib['K_r'])
dist_l, dist_r = np.array(calib['dist_l']), np.array(calib['dist_r'])
R_l, R_r = np.array(calib['R_l']), np.array(calib['R_r'])
P_l, P_r = np.array(calib['P_l']), np.array(calib['P_r'])
baseline = float(calib['baseline'])

fx, fy = P_l[0, 0], P_l[1, 1]
cx, cy = P_l[0, 2], P_l[1, 2]

logging.info(f"Baseline: {baseline*1000:.2f}mm. Rectified focal: fx={fx:.1f}, fy={fy:.1f}")

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

# ===== 3. Initialize Two RealSense RGB Cameras =====
ctx = rs.context()
devices = ctx.query_devices()
assert len(devices) >= 2, "Failed to find at least two RealSense devices."

serial_left = devices[0].get_info(rs.camera_info.serial_number)
serial_right = devices[1].get_info(rs.camera_info.serial_number)

pipeline_left = rs.pipeline()
c_left = rs.config()
c_left.enable_device(serial_left)
c_left.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, 30)
pipeline_left.start(c_left)

pipeline_right = rs.pipeline()
c_right = rs.config()
c_right.enable_device(serial_right)
c_right.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, 30)
pipeline_right.start(c_right)
logging.info(f"Cameras initialized: L({serial_left}) R({serial_right})")

# ===== 4. Open3D visualizer =====
vis = o3d.visualization.Visualizer()
vis.create_window("Dual RealSense FFS Manual Capture point cloud", width=1280, height=720)
vis.get_render_option().point_size = 2.0
vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)
first_frame = True

cv2.namedWindow("Dual RealSense Preview (Left Camera)", cv2.WINDOW_AUTOSIZE)

# ===== 5. Main loop =====
logging.info("Ready! Select 'Dual RealSense Preview (Left Camera)' window and press SPACE to capture and process. Press ESC to exit.")
try:
    while True:
        # Keep Open3D responsive
        vis.poll_events()
        vis.update_renderer()

        frames_left = pipeline_left.wait_for_frames()
        frames_right = pipeline_right.wait_for_frames()
        color_frame_left = frames_left.get_color_frame()
        color_frame_right = frames_right.get_color_frame()
        
        if not color_frame_left or not color_frame_right: continue

        raw_left = np.asanyarray(color_frame_left.get_data())
        raw_right = np.asanyarray(color_frame_right.get_data())

        cv2.imshow("Dual RealSense Preview (Left Camera)", raw_left)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == 32 or key == ord('c'):
            logging.info("Processing frame...")
            t0 = time.time()

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

            depth = fx * baseline / disp
            depth[(depth < ZNEAR) | (depth > ZFAR) | ~np.isfinite(depth)] = 0

            depth_ds = depth[::PCD_STRIDE, ::PCD_STRIDE]
            z_flat = depth_ds.reshape(-1)
            valid_mask = z_flat > 0

            z = z_flat[valid_mask]
            u = u_flat[valid_mask]
            v = v_flat[valid_mask]

            x3d = (u - cx) * z / fx
            y3d = (v - cy) * z / fy
            points = np.stack([x3d, y3d, z], axis=-1)

            colors = rect_left[v.astype(int), u.astype(int), ::-1].astype(np.float64) / 255.0

            if len(points) == 0:
                logging.warning("No valid points generated!")
                continue

            # Create temporary point cloud for denoising
            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            temp_pcd.colors = o3d.utility.Vector3dVector(colors)

            logging.info(f"Points before denoise: {len(temp_pcd.points)}")
            
            # Apply denoise
            temp_pcd = temp_pcd.voxel_down_sample(voxel_size=DENOISE_VOXEL_SIZE)
            cl, ind = temp_pcd.remove_radius_outlier(nb_points=DENOISE_NB_POINTS, radius=DENOISE_RADIUS)
            temp_pcd = temp_pcd.select_by_index(ind)

            logging.info(f"Points after denoise: {len(temp_pcd.points)}")

            # Update actual visualizer geometry
            pcd.points = temp_pcd.points
            pcd.colors = temp_pcd.colors

            if first_frame:
                vis.reset_view_point(True)
                ctr = vis.get_view_control()
                ctr.set_front([0, 0, -1])
                ctr.set_up([0, -1, 0])
                first_frame = False

            vis.update_geometry(pcd)
            
            t1 = time.time()
            logging.info(f"Processing finished in {t1-t0:.3f}s. Points mapped to 3D viewer.")

except KeyboardInterrupt:
    pass
finally:
    pipeline_left.stop()
    pipeline_right.stop()
    cv2.destroyAllWindows()
    vis.destroy_window()
    logging.info("Exited")
