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
import base64, json, re
import threading
import multiprocessing
from volcenginesdkarkruntime import Ark

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

# ===== Volcengine Config =====
ARK_API_KEY = os.getenv('ARK_API_KEY')
client = Ark(
    base_url='https://ark.cn-beijing.volces.com/api/v3',
    api_key=ARK_API_KEY,
)
MODEL_NAME = "doubao-seed-2-0-mini-260215"
REASONING_EFFORT = "minimal"

# ===== Parameters =====
IMG_WIDTH = 640
IMG_HEIGHT = 480
MASK_ALPHA = 0.5
MASK_COLORS_BGR = {1: [75, 70, 203], 2: [203, 192, 75]} # 1: Hand, 2: Object

vlm_running = False
pending_targets = []
doubao_trigger = False

def call_doubao_hand_object(image_bgr):
    if not ARK_API_KEY:
        logging.error("ARK_API_KEY not set.")
        return None

    _, buffer = cv2.imencode('.jpg', image_bgr)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{img_base64}"

    grounding_text = (
        "Instructions: \n"
        "1. Detect the human hand in the image. If no hand is present, reply exactly: 'No hand detected'.\n"
        "2. If a hand is present, detect both the 'hand' and the 'object' being held or interactived with by that hand.\n"
        "3. Provide their locations in JSON format as follows: \n"
        "   {\"hand\": [ymin, xmin, ymax, xmax], \"object\": [ymin, xmin, ymax, xmax]}\n"
        "   Coordinates should be normalized 0-1000. \n"
        "Return ONLY the JSON or 'No hand detected'."
    )
    
    try:
        kwargs = {
            "model": MODEL_NAME,
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
        
        content = ""
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
        elif hasattr(response, 'output') and len(response.output) > 1:
             content = response.output[1].content[0].text
        else:
            content = str(response)

        logging.info(f"Doubao Response: {content}")

        if "No hand detected" in content:
            logging.warning("VLM: Hand not found in frame.")
            return []

        try:
            json_str = re.search(r'\{.*\}', content, re.DOTALL)
            if json_str:
                data = json.loads(json_str.group())
                results = []
                h, w = image_bgr.shape[:2]
                
                if "hand" in data:
                    ymin, xmin, ymax, xmax = data["hand"]
                    bbox = (int(xmin*w/1000), int(ymin*h/1000), int(xmax*w/1000), int(ymax*h/1000))
                    results.append((1, bbox))
                
                if "object" in data:
                    ymin, xmin, ymax, xmax = data["object"]
                    bbox = (int(xmin*w/1000), int(ymin*h/1000), int(xmax*w/1000), int(ymax*h/1000))
                    results.append((2, bbox))
                
                return results
        except Exception as je:
            logging.error(f"JSON Parse failed: {je}")
            
        return []
    except Exception as e:
        logging.error(f"API call failed: {e}")
        return []

def vlm_thread_worker(image_bgr):
    global pending_targets, vlm_running
    try:
        found_targets = call_doubao_hand_object(image_bgr)
        if found_targets:
            pending_targets = found_targets
            logging.info(f"VLM Worker: Found {len(found_targets)} targets.")
    finally:
        vlm_running = False

def get_net(checkpoint_path, num_view=300):
    net = GraspNet(input_feature_dim=0, num_view=num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net, device

def start_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, IMG_WIDTH, IMG_HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.rgb8, 30)
    
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale() 
    factor_depth = 1.0 / depth_scale 
    
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intrinsics = color_profile.get_intrinsics()
    
    camera = CameraInfo(640.0, 480.0, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy, factor_depth)
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
    
    end_points = dict()
    cloud_sampled_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device, dtype=torch.float32)
    with torch.autocast(device_type="cuda", dtype=torch.float32):
        pass 
    end_points['point_clouds'] = cloud_sampled_tensor
    end_points['cloud_colors'] = color_sampled
    
    return end_points, cloud_o3d, workspace_mask

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
    global vlm_running, pending_targets, doubao_trigger
    
    checkpoint_path = 'logs/log_rs/checkpoint-rs.tar' 
    if not os.path.exists(checkpoint_path) and os.path.exists('logs/log_rs/checkpoint.tar'):
        checkpoint_path = 'logs/log_rs/checkpoint.tar'
    net, device = get_net(checkpoint_path)
    
    SAM2_CHECKPOINT = os.path.join(SAM2_DIR, "checkpoints/sam2.1/sam2.1_hiera_small.pt")
    SAM2_CFG = "sam2.1/sam2.1_hiera_s.yaml"
    logging.info("Loading SAM2 model...")
    sam2_predictor = build_sam2_camera_predictor(SAM2_CFG, SAM2_CHECKPOINT)
    sam2_predictor.fill_hole_area = 0
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
                    
                    p = multiprocessing.Process(target=show_open3d_process, args=(
                        np.asarray(cloud_o3d.points),
                        np.asarray(cloud_o3d.colors),
                        gg_array
                    ))
                    p.daemon = True
                    p.start()

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
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
