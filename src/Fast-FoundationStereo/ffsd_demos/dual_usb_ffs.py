"""
Dual USB Camera RGB to Point Cloud using Fast-FoundationStereo

Usage:
  conda activate ffs
  python dual_usb_ffs.py
"""

import os, sys, time, logging
import numpy as np
import torch
import yaml
import cv2
import open3d as o3d

# Add FFS path
FFS_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(FFS_DIR)
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ===== Parameters =====
MODEL_DIR = os.path.join(FFS_DIR, "weights/23-36-37/model_best_bp2_serialize.pth")
CALIB_FILE = os.path.join(FFS_DIR, "calibration", "stereo_calib.yaml")
VALID_ITERS = 8
MAX_DISP = 192
ZFAR = 5.0
ZNEAR = 0.05
# Keep original calibration aspect ratio but downscale to reduce VRAM
# 1280x720 (16:9). To match width 640, height is 360. 
# Alternatively we format to 640x480 by cropping, but 640x360 is safer and maintains calibration FOV.
IMG_WIDTH = 640
IMG_HEIGHT = 384 # Multiple of 32, close to 360
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

# Original resolution was 1280x720
ORIG_W, ORIG_H = 1280, 720

# We want to rectify to 640x384. 
# We adjust P matrix to reflect the scaling and letterboxing/cropping.
# Simplest: Rectify at 1280x720, then resize to 640x384 afterwards.
# That way we don't mess up principal points carefully.
map_lx, map_ly = cv2.initUndistortRectifyMap(K_l, dist_l, R_l, P_l, (ORIG_W, ORIG_H), cv2.CV_32FC1)
map_rx, map_ry = cv2.initUndistortRectifyMap(K_r, dist_r, R_r, P_r, (ORIG_W, ORIG_H), cv2.CV_32FC1)

# P_l is for 1280x720. If we scale to 640x384 (resize 1280x720 -> 640x360 -> pad to 384)
# Scale factor is 0.5
scale_x = IMG_WIDTH / ORIG_W
scale_y = 360 / ORIG_H # 0.5

fx_ffs = P_l[0, 0] * scale_x
fy_ffs = P_l[1, 1] * scale_y
cx_ffs = P_l[0, 2] * scale_x
cy_ffs = P_l[1, 2] * scale_y + (IMG_HEIGHT - 360) / 2.0 # padded offset

logging.info(f"Baseline: {baseline*1000:.2f}mm. Target focal: fx={fx_ffs:.1f}, fy={fy_ffs:.1f}")

u_grid, v_grid = np.meshgrid(np.arange(0, IMG_WIDTH, PCD_STRIDE), np.arange(0, IMG_HEIGHT, PCD_STRIDE))
u_flat, v_flat = u_grid.flatten().astype(np.float32), v_grid.flatten().astype(np.float32)

# ===== 2. Load FFS model =====
logging.info("Loading FFS model...")
torch.autograd.set_grad_enabled(False)
model = torch.load(MODEL_DIR, map_location='cpu', weights_only=False)
model.args.valid_iters = VALID_ITERS
model.args.max_disp = MAX_DISP
model.cuda().eval()

# ===== 3. Initialize Two USB Cameras =====
def init_camera(dev_id):
    cap = cv2.VideoCapture(dev_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ORIG_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ORIG_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

logging.info("Scanning for valid USB cameras (this may take a few seconds)...")
valid_devs = []
for i in range(12):
    cap = init_camera(i)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            valid_devs.append(i)
    cap.release()
    if len(valid_devs) == 2:
        break

if len(valid_devs) < 2:
    raise RuntimeError(f"Only found {len(valid_devs)} valid cameras: {valid_devs}. Need 2. Check device IDs and permissions.")

DEV_LEFT, DEV_RIGHT = valid_devs[0], valid_devs[1]
logging.info(f"Initializing USB cameras on /dev/video{DEV_LEFT} and /dev/video{DEV_RIGHT}...")
cap_left = init_camera(DEV_LEFT)
cap_right = init_camera(DEV_RIGHT)

if not cap_left.isOpened() or not cap_right.isOpened():
    raise RuntimeError("Failed to open cameras.")

# ===== 4. Open3D visualizer =====
vis_ffs = o3d.visualization.Visualizer()
vis_ffs.create_window("FFS USB Point Cloud", width=800, height=600, left=50, top=50)
vis_ffs.get_render_option().point_size = 2.0
vis_ffs.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
pcd_ffs = o3d.geometry.PointCloud()
vis_ffs.add_geometry(pcd_ffs)
first_frame_ffs = True

cv2.namedWindow("Dual USB Preview", cv2.WINDOW_AUTOSIZE)

# ===== 5. Main loop =====
logging.info("Ready! Select 'Dual USB Preview' window and press SPACE to capture and process. Press ESC to exit.")
try:
    while True:
        vis_ffs.poll_events()
        vis_ffs.update_renderer()

        ret_l = cap_left.grab()
        ret_r = cap_right.grab()
        if not (ret_l and ret_r):
            continue

        _, raw_left = cap_left.retrieve()
        _, raw_right = cap_right.retrieve()
        
        # Resize for preview
        preview_l = cv2.resize(raw_left, (640, 360))
        preview_r = cv2.resize(raw_right, (640, 360))
        preview = cv2.hconcat([preview_l, preview_r])

        cv2.imshow("Dual USB Preview", preview)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == 32 or key == ord('c'):
            logging.info("Processing comparison...")
            t0 = time.time()

            # ----------------------------------------------------
            # Fast-FoundationStereo Point Cloud
            # ----------------------------------------------------
            rect_left = cv2.remap(raw_left, map_lx, map_ly, cv2.INTER_LINEAR)
            rect_right = cv2.remap(raw_right, map_rx, map_ry, cv2.INTER_LINEAR)

            # Resize to 640x360 (scale 0.5) to save memory
            rect_left_small = cv2.resize(rect_left, (640, 360))
            rect_right_small = cv2.resize(rect_right, (640, 360))

            # Pad to 640x384 (multiple of 32)
            pad_top = (IMG_HEIGHT - 360) // 2
            pad_bot = IMG_HEIGHT - 360 - pad_top
            rect_left_pad = cv2.copyMakeBorder(rect_left_small, pad_top, pad_bot, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
            rect_right_pad = cv2.copyMakeBorder(rect_right_small, pad_top, pad_bot, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])

            img0 = torch.as_tensor(rect_left_pad[:, :, ::-1].copy()).cuda().float()[None].permute(0, 3, 1, 2)
            img1 = torch.as_tensor(rect_right_pad[:, :, ::-1].copy()).cuda().float()[None].permute(0, 3, 1, 2)
            
            with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
                disp = model.forward(img0, img1, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
            
            disp = disp.data.cpu().numpy().reshape(IMG_HEIGHT, IMG_WIDTH).clip(0, None)

            xx = np.arange(IMG_WIDTH)[None, :].repeat(IMG_HEIGHT, axis=0)
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

            colors_ffs = rect_left_pad[v_f.astype(int), u_f.astype(int), ::-1].astype(np.float64) / 255.0

            if len(pts_ffs) > 0:
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(pts_ffs.astype(np.float64))
                temp_pcd.colors = o3d.utility.Vector3dVector(colors_ffs)

                # Denoise
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
            logging.info(f"Processing done! Time: {t1-t0:.3f}s.")

except KeyboardInterrupt:
    pass
finally:
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
    vis_ffs.destroy_window()
    logging.info("Exited")
