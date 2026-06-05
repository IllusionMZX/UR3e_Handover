import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import open3d as o3d
import os
import sys
import datetime
import scipy.io as scio
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ===== GPU config (SAM2 requires bfloat16) =====
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# === 添加相关的目录到 sys.path 中，使得 python 能找到 models, dataset, utils ===
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

# SAM2 path
FFS_ROOT = os.path.dirname(ROOT_DIR)
SAM2_DIR = os.path.join(FFS_ROOT, "SAM2_streaming")
sys.path.insert(0, SAM2_DIR)
from sam2.build_sam import build_sam2_camera_predictor

# 导入 graspnet-baseline 原有模块
from graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup
from data_utils import CameraInfo, create_point_cloud_from_depth_image

def get_net(checkpoint_path, num_view=300):
    net = GraspNet(input_feature_dim=0, num_view=num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net, device

# --- SAM2 Global State & Mouse Callback ---
drawing = False
ix, iy, fx_mouse, fy_mouse = -1, -1, -1, -1
pending_bbox = None
pending_point = None
num_targets = 0 
need_reset = False
current_masks = {} 
all_prompts = []
MASK_COLORS_BGR = {1: [75, 70, 203], 2: [203, 192, 75]}
MASK_ALPHA = 0.5 

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

def start_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    # 配置 RGB 和 Depth 分辨率，可以根据自身相机（如 D435）支持的分辨率调整
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    # 获取相机内参和深度比例
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale() # 通常为 0.001 (意味着1个单位=1mm)
    factor_depth = 1.0 / depth_scale # 转换为 graspnet 需要的 1000.0
    
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intrinsics = color_profile.get_intrinsics()
    
    camera = CameraInfo(640.0, 480.0, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy, factor_depth)
    return pipeline, align, camera

def process_frame(color_img, depth_img, camera, device, sam_mask=None, num_point=20000):
    # 1. 颜色归一化为 0~1
    color = color_img.astype(np.float32) / 255.0
    
    # 2. 生成基本的 workspace_mask，优先使用 SAM2 分割出的掩码
    if sam_mask is not None and np.any(sam_mask):
        # 结合sam对象掩膜和深度信息（忽略深度=0的空洞，以及超远距离异常点）
        workspace_mask = (sam_mask > 0) & (depth_img > 0) & (depth_img < 2000)
    else:
        # 简单的启发式过滤：假设深度大于0，且只关注深度在 0.2米 到 1.0米 的区域
        z_min_mm, z_max_mm = 200, 1000 
        workspace_mask = (depth_img > z_min_mm) & (depth_img < z_max_mm)
    
    # 还可以在图像平面上裁剪一个中间的矩形区域作为操作桌面
    # center_mask = np.zeros_like(workspace_mask)
    # center_mask[100:380, 160:480] = True
    # workspace_mask = workspace_mask & center_mask
    
    # 3. 把 depth 图片转换成伪点云格式 (H, W, 3) 
    # data_utils.py 中的 create_point_cloud_from_depth_image
    cloud = create_point_cloud_from_depth_image(depth_img, camera, organized=True)
    
    # 4. 获取有效的 mask 点云
    mask = (workspace_mask & (depth_img > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]
    
    if len(cloud_masked) == 0:
        return None, None, None
        
    # 5. 采样统一的点数输入神经网络
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
    
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_full.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_full.astype(np.float32))
    
    # 组装网络输入
    end_points = dict()
    # Ensure point_clouds is float32
    cloud_sampled_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device, dtype=torch.float32)
    # Ensure inference runs with float32 despite SAM2 autocast
    with torch.autocast(device_type="cuda", dtype=torch.float32):
        pass # just a dummy block as we will apply this context manager inside main() instead
    end_points['point_clouds'] = cloud_sampled_tensor
    end_points['cloud_colors'] = color_sampled
    
    return end_points, cloud_o3d, workspace_mask

import threading
import multiprocessing

def show_open3d_process(points, colors, gg_array):
    import open3d as o3d
    import numpy as np
    from graspnetAPI import GraspGroup
    
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    
    gg = GraspGroup(gg_array)
    gg = gg.nms()
    gg = gg.sort_by_score()
    
    view_kwargs = {"front": [0, 0, -1], "lookat": [0, 0, 0.5], "up": [0, -1, 0], "zoom": 0.8}
    
    if len(gg) > 0:
        grippers = gg[:1].to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers], **view_kwargs)
    else:
        o3d.visualization.draw_geometries([cloud], **view_kwargs)

def main():
    global need_reset, num_targets, pending_bbox, pending_point, current_masks, all_prompts, drawing, ix, iy, fx_mouse, fy_mouse
    
    checkpoint_path = 'logs/log_rs/checkpoint-rs.tar' # 替换为你的模型权重路径
    if not os.path.exists(checkpoint_path) and os.path.exists('logs/log_rs/checkpoint.tar'):
        checkpoint_path = 'logs/log_rs/checkpoint.tar'
    net, device = get_net(checkpoint_path)
    
    # Load SAM2
    SAM2_CHECKPOINT = os.path.join(SAM2_DIR, "checkpoints/sam2.1/sam2.1_hiera_small.pt")
    SAM2_CFG = "sam2.1/sam2.1_hiera_s.yaml"
    logging.info("Loading SAM2 model...")
    sam2_predictor = build_sam2_camera_predictor(SAM2_CFG, SAM2_CHECKPOINT)
    sam2_predictor.fill_hole_area = 0
    logging.info("SAM2 model loaded")
    
    pipeline, align, camera_info = start_realsense()
    
    cv2.namedWindow("Realsense Viewer", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Realsense Viewer", mouse_callback)

    logging.info("Camera Started. UI Controls (focus on OpenCV window):")
    logging.info("  - Left-click drag: Draw bounding box -> initialize tracking")
    logging.info("  - Left-click: Select foreground point -> initialize tracking")
    logging.info("  - Space: Capture and predict grasp for targeted object(s)")
    logging.info("  - r: Reset SAM2 selection")
    logging.info("  - q: Quit")
    
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
                pending_bbox = None
                pending_point = None
                current_masks = {}
                all_prompts = []
                logging.info("Reset, select new targets (up to 2)")

            # --- SAM2: Initialize targets ---
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

            t1 = time.time()
            fps = 1.0 / (t1 - t0)
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if num_targets > 0:
                status = f"TRACKING {num_targets}/2 | SPACE=Predict | r=reset | q=quit"
            else:
                status = "Select targets | SPACE=Predict | q=quit"
            cv2.putText(display, status, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Realsense Viewer", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            # 按空格提取当前帧进行抓取检测
            if key == 32: 
                print("Processing Frame for GraspNet...")
                # 合并 SAM 掩码
                combined_mask = np.zeros((480, 640), dtype=bool)
                for mask in current_masks.values():
                    if mask is not None:
                        combined_mask |= (mask > 0)
                
                if not np.any(combined_mask):
                    print("No SAM mask found, using default workspace heuristic!")
                    combined_mask = None # 触发 fallback 默认框
                
                import threading
                def process_grasp(c_img, d_img, cam_info, dev, c_mask, t_img):
                    end_points, cloud_o3d, workspace_mask = process_frame(c_img, d_img, cam_info, dev, sam_mask=c_mask)
                    if end_points is None:
                        print("No points found in workspace mask.")
                        return
                    
                    # --- 保存数据 ---
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_dir = os.path.join("captured_data", timestamp)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # 保存 clean color.png 不带分割外框
                    cv2.imwrite(os.path.join(save_dir, 'color.png'), t_img)
                    
                    # 保存 depth.png (16-bit)
                    cv2.imwrite(os.path.join(save_dir, 'depth.png'), d_img)
                    
                    # 保存 workspace_mask.png (将 bool 转成 255 的 uint8 图)
                    mask_img = (workspace_mask.astype(np.uint8) * 255)
                    cv2.imwrite(os.path.join(save_dir, 'workspace_mask.png'), mask_img)
                    
                    # 保存 meta.mat
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
                    # ----------------
                    
                    with torch.no_grad():
                        # Temporarily disable bfloat16 autocast for PointNet2 C++ extensions
                        with torch.autocast(device_type="cuda", dtype=torch.float32):
                            end_points = net(end_points)
                            grasp_preds = pred_decode(end_points)
                    
                    # 提取预测结果
                    gg_array = grasp_preds[0].detach().cpu().numpy()
                    
                    # 独立进程可视化，防止阻塞主线程的 cv2.waitKey 和 X11 窗口
                    p = multiprocessing.Process(target=show_open3d_process, args=(
                        np.asarray(cloud_o3d.points),
                        np.asarray(cloud_o3d.colors),
                        gg_array
                    ))
                    p.daemon = True
                    p.start()

                # 使用后台线程运行 graspnet 处理，避免阻塞主线程及相机流
                t = threading.Thread(target=process_grasp, args=(
                    color_img.copy(), depth_img.copy(), camera_info, device, 
                    combined_mask.copy() if combined_mask is not None else None, 
                    tracking_img.copy()))
                t.daemon = True
                t.start()
                
            elif key == ord('r'):
                need_reset = True
            elif key == ord('q'):
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()