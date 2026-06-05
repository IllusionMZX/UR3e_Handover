"""
ZED 2i + Fast-FoundationStereo Real-time Depth Estimation and Point Cloud Visualization

Usage:
  conda activate ffs
  python zed_ffs_realtime.py
"""

import os, sys, time, logging
import numpy as np
import cv2
import torch
import yaml
import pyzed.sl as sl
import open3d as o3d

# Add FFS path
FFS_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(FFS_DIR)
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, vis_disparity, depth2xyzmap

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ===== Parameters =====
MODEL_DIR = os.path.join(FFS_DIR, "weights/23-36-37/model_best_bp2_serialize.pth")
VALID_ITERS = 8       # Accuracy=8, speed=4
MAX_DISP = 192
ZFAR = 5.0            # Max depth (meters)
ZNEAR = 0.2           # Min depth (meters)

# Denoising parameters (和 dual_usb_ffs.py 保持一致)
DENOISE_VOXEL_SIZE = 0.001
DENOISE_NB_POINTS = 30
DENOISE_RADIUS = 0.03

# 为了节省显存和加快 FFS 的推理速度，我们将 720p 缩小到 640x360 (也是 16:9)
IMG_W = 640
IMG_H = 360
SCALE_FACTOR = 0.5 

# ===== 1. Load FFS model =====
logging.info("Loading FFS model...")
torch.autograd.set_grad_enabled(False)

# 加载配置
with open(os.path.join(os.path.dirname(MODEL_DIR), "cfg.yaml"), 'r') as f:
    cfg = yaml.safe_load(f)
cfg['valid_iters'] = VALID_ITERS
cfg['max_disp'] = MAX_DISP

model = torch.load(MODEL_DIR, map_location='cpu', weights_only=False)
model.args.valid_iters = VALID_ITERS
model.args.max_disp = MAX_DISP
model.cuda().eval()
logging.info("FFS model loaded")

# ===== 2. Initialize ZED Camera =====
logging.info("Initializing ZED Camera...")
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720 # 1280x720
init_params.camera_fps = 30
init_params.depth_mode = sl.DEPTH_MODE.NONE         # 关闭自带深度，省算力
init_params.coordinate_units = sl.UNIT.METER        # 把 baseline 等参数的单位调成米

err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    raise RuntimeError(f"Failed to open ZED Camera: {err}")

# 初始化图像抓取对象
image_left = sl.Mat()
image_right = sl.Mat()

# ===== 3. Get camera intrinsics =====
cam_info = zed.get_camera_information()
calib = cam_info.camera_configuration.calibration_parameters
left_cam = calib.left_cam
baseline = calib.get_camera_baseline() # 获取双目基线长度 (米)

# 计算缩放后的内参矩阵 K
K = np.array([
    [left_cam.fx * SCALE_FACTOR, 0, left_cam.cx * SCALE_FACTOR],
    [0, left_cam.fy * SCALE_FACTOR, left_cam.cy * SCALE_FACTOR],
    [0, 0, 1]
], dtype=np.float32)

logging.info(f"Scaled Intrinsics K (640x360):\n{K}")
logging.info(f"Baseline: {baseline*1000:.1f}mm")

# ===== 4. Warm up model =====
logging.info("Warming up model (first inference will be slower)...")
dummy_left = torch.randn(1, 3, IMG_H, IMG_W).cuda().float()
dummy_right = torch.randn(1, 3, IMG_H, IMG_W).cuda().float()
padder = InputPadder(dummy_left.shape, divis_by=32, force_square=False)
dummy_left_p, dummy_right_p = padder.pad(dummy_left, dummy_right)
with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
    _ = model.forward(dummy_left_p, dummy_right_p, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
del dummy_left, dummy_right, dummy_left_p, dummy_right_p
torch.cuda.empty_cache()
logging.info("Warm-up complete")

# ===== 5. Open3D visualizer =====
vis = o3d.visualization.Visualizer()
vis.create_window("ZED + FFS Real-time Point Cloud", width=1280, height=720)
vis.get_render_option().point_size = 2.0
vis.get_render_option().background_color = np.array([1.0, 1.0, 1.0])
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)
first_frame = True

# ===== 6. Main loop =====
logging.info("Ready! Select 'ZED Preview' window and press SPACE (or 'c') to capture and process. Press ESC (or 'q') to exit.")
cv2.namedWindow("ZED Preview", cv2.WINDOW_AUTOSIZE)

frame_count = 0
enable_denoise = True
logging.info(f"Denoise: {'ON' if enable_denoise else 'OFF'} (press 'd' to toggle)")

try:
    while True:
        # 始终刷新 Open3D，防止窗口卡死
        vis.poll_events()
        vis.update_renderer()

        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # 抓取经过 ZED SDK 做了去畸变和行对齐的双目图像 (Rectified)
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            zed.retrieve_image(image_right, sl.VIEW.RIGHT)

            # BGRA -> BGR (去掉透明度通道)
            left_bgr_full = image_left.get_data()[:, :, :3]
            right_bgr_full = image_right.get_data()[:, :, :3]

            # 缩放图像以防爆显存并提升速度 (1280x720 -> 640x360)
            left_bgr = cv2.resize(left_bgr_full, (IMG_W, IMG_H))
            right_bgr = cv2.resize(right_bgr_full, (IMG_W, IMG_H))

            # 实时显示预览画面 (仅仅拼起来展示，不走 FFS)
            preview = cv2.hconcat([left_bgr, right_bgr])
            cv2.imshow("ZED Preview", preview)

            key = cv2.waitKey(1) & 0xFF
            
            # 按 ESC 或 q 退出
            if key == 27 or key == ord('q'):
                break
            # 按 d 切换去噪
            elif key == ord('d'):
                enable_denoise = not enable_denoise
                logging.info(f"Denoise toggled: {'ON' if enable_denoise else 'OFF'}")
            # 按空格或 c 触发生成点云
            elif key == 32 or key == ord('c'):
                logging.info("Processing generation...")
                t0 = time.time()

                # BGR -> RGB (FFS模型训练通常用的RGB格式)
                left_rgb = left_bgr[:, :, ::-1]
                right_rgb = right_bgr[:, :, ::-1]

                # Convert to tensor [1, 3, H, W]
                img0 = torch.as_tensor(left_rgb.copy()).cuda().float()[None].permute(0, 3, 1, 2)
                img1 = torch.as_tensor(right_rgb.copy()).cuda().float()[None].permute(0, 3, 1, 2)
                
                # 使用 Padder 自动 padding 到 32 的整数倍 (例如 360 会被 pad 到 384)
                padder = InputPadder(img0.shape, divis_by=32, force_square=False)
                img0_p, img1_p = padder.pad(img0, img1)

                # FFS inference
                with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
                    disp = model.forward(img0_p, img1_p, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
                
                # 还原尺寸并截断为 numpy 格式
                disp = padder.unpad(disp.float())
                disp = disp.data.cpu().numpy().reshape(IMG_H, IMG_W).clip(0, None)

                # 过滤无效区域（由于视差负值等）
                xx = np.arange(IMG_W)[None, :].repeat(IMG_H, axis=0)
                invalid = (xx - disp) < 0
                disp[invalid] = np.inf

                # Disparity → depth 视差转深度
                depth = K[0, 0] * baseline / disp
                depth[(depth < ZNEAR) | (depth > ZFAR) | ~np.isfinite(depth)] = 0

                # Generate point cloud (彩色点云)
                xyz_map = depth2xyzmap(depth, K) # shape: (H, W, 3)
                points = xyz_map.reshape(-1, 3)
                colors = left_rgb.reshape(-1, 3)

                # Filter invalid points 剔除无效点
                valid = points[:, 2] > 0
                points = points[valid]
                colors = colors[valid]

                if len(points) > 0:
                    temp_pcd = o3d.geometry.PointCloud()
                    temp_pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
                    temp_pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

                    # Optional denoise: 体素下采样 + 半径滤波离群点
                    if enable_denoise:
                        temp_pcd = temp_pcd.voxel_down_sample(voxel_size=DENOISE_VOXEL_SIZE)
                        _, ind = temp_pcd.remove_radius_outlier(
                            nb_points=DENOISE_NB_POINTS,
                            radius=DENOISE_RADIUS,
                        )
                        temp_pcd = temp_pcd.select_by_index(ind)

                    pcd.points = temp_pcd.points
                    pcd.colors = temp_pcd.colors
                else:
                    pcd.points = o3d.utility.Vector3dVector()
                    pcd.colors = o3d.utility.Vector3dVector()

                if first_frame:
                    vis.reset_view_point(True)
                    ctr = vis.get_view_control()
                    # 调整第一视角的位姿
                    ctr.set_front([0, 0, -1])
                    ctr.set_up([0, -1, 0])
                    first_frame = False

                vis.update_geometry(pcd)
                
                t1 = time.time()
                logging.info(f"Processing done! Time: {t1-t0:.3f}s. Points: {len(pcd.points)}")

finally:
    vis.destroy_window()
    cv2.destroyAllWindows()
    zed.close()
    logging.info("Exited")
