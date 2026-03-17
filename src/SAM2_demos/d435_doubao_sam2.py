"""
RealSense D435 + Volcengine Doubao-Seed + SAM2 Tracking Real-time Segmentation Demo

Features:
  - Real-time RGB streaming from D435
  - Volcengine Doubao-Seed (Multimodal) for object selection via prompt
  - SAM2 interactive segmentation (Box/Point)
  - Mask overlay and contour visualization

Usage:
  export ARK_API_KEY="your_api_key"
  python d435_doubao_sam2.py

Controls:
  - m: Toggle Mouse Mode (default: OFF)
  - v: Trigger Doubao prompt input (enter prompt in terminal) → initialize tracking
  - r: Reset selection
  - q: Quit
"""

import os, sys, time, logging, base64, json, re, threading
import numpy as np
import torch
import cv2
import pyrealsense2 as rs
from volcenginesdkarkruntime import Ark

# SAM2 path
FFS_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SAM2_DIR = os.path.join(FFS_ROOT, "SAM2_streaming")
sys.path.insert(0, SAM2_DIR)
from sam2.build_sam import build_sam2_camera_predictor

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ===== Volcengine Config =====
ARK_API_KEY = os.getenv('ARK_API_KEY')
client = Ark(
    base_url='https://ark.cn-beijing.volces.com/api/v3',
    api_key=ARK_API_KEY,
)
MODEL_NAME = "doubao-seed-2-0-mini-260215"
REASONING_EFFORT = "minimal" # Options: "minimal", "low", "medium", "high"

# ===== GPU config (SAM2 requires bfloat16) =====
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ===== Parameters =====
SAM2_CHECKPOINT = os.path.join(SAM2_DIR, "checkpoints/sam2.1/sam2.1_hiera_small.pt")
SAM2_CFG = "sam2.1/sam2.1_hiera_s.yaml"

IMG_WIDTH = 640
IMG_HEIGHT = 480
MASK_ALPHA = 0.5
MASK_COLORS_BGR = {1: [75, 70, 203], 2: [203, 192, 75]} 

# ===== 1. Load SAM2 model =====
logging.info("Loading SAM2 model...")
sam2_predictor = build_sam2_camera_predictor(SAM2_CFG, SAM2_CHECKPOINT)
sam2_predictor.fill_hole_area = 0
logging.info("SAM2 model loaded")

# ===== 2. Initialize RealSense D435 =====
logging.info("Initializing RealSense D435...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, 30)

try:
    profile = pipeline.start(config)
    device = profile.get_device()
    logging.info(f"Using device: {device.get_info(rs.camera_info.name)}")
except Exception as e:
    logging.error(f"Failed to start RealSense: {e}")
    sys.exit(1)

# ===== 3. OpenCV window + mouse interaction =====
cv2.namedWindow("D435 + Doubao + SAM2", cv2.WINDOW_AUTOSIZE)

drawing = False
ix, iy, fx_mouse, fy_mouse = -1, -1, -1, -1
pending_bbox = None
pending_point = None
num_targets = 0 
need_reset = False
current_masks = {} 
all_prompts = [] 
doubao_prompt_trigger = False
vlm_running = False
vlm_snapshot = None
vlm_input_requested = False
mouse_mode = True # Default enabled

def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, fx_mouse, fy_mouse, pending_bbox, pending_point, mouse_mode

    if not mouse_mode:
        return

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

cv2.setMouseCallback("D435 + Doubao + SAM2", mouse_callback)

def call_doubao_grounding(image_bgr, prompt_text):
    if not ARK_API_KEY:
        logging.error("ARK_API_KEY environment variable not set.")
        return None

    # Encode image to base64
    _, buffer = cv2.imencode('.jpg', image_bgr)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{img_base64}"

    grounding_text = f"You are a vision assistant. Your task is to locate the '{prompt_text}' in this image. Provide the bounding box coordinates as a normalized array [ymin, xmin, ymax, xmax] (range 0-1000). Return ONLY the array, for example: [200, 150, 800, 600]."
    
    try:
        kwargs = {
            "model": MODEL_NAME,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": "You are a professional object detection assistant. Always return coordinates in [ymin, xmin, ymax, xmax] format."}]
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
        
        match = re.search(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', content)
        if match:
            ymin, xmin, ymax, xmax = map(int, match.groups())
            h, w = image_bgr.shape[:2]
            x1 = int(xmin * w / 1000)
            y1 = int(ymin * h / 1000)
            x2 = int(xmax * w / 1000)
            y2 = int(ymax * h / 1000)
            return (x1, y1, x2, y2)
        else:
            logging.warning("Failed to parse coordinates from Doubao response.")
            return None
    except Exception as e:
        logging.error(f"Doubao API call failed: {e}")
        return None

def vlm_thread_worker(image_bgr, prompt_text):
    global pending_bbox, vlm_running
    try:
        detected_bbox = call_doubao_grounding(image_bgr, prompt_text)
        if detected_bbox:
            pending_bbox = detected_bbox
            logging.info(f"Doubao detected target at {detected_bbox}")
    finally:
        vlm_running = False

def input_thread_worker():
    """Dedicated thread for terminal input to avoid UI freezing."""
    global vlm_input_requested, vlm_snapshot, vlm_running
    while True:
        if vlm_input_requested:
            user_prompt = input("\n[VLM] Enter object name to segment: ")
            if user_prompt.strip():
                logging.info(f"Starting VLM inference for: {user_prompt}...")
                vlm_running = True
                threading.Thread(target=vlm_thread_worker, args=(vlm_snapshot, user_prompt), daemon=True).start()
            vlm_input_requested = False
        time.sleep(0.1)

# Start input listener thread
threading.Thread(target=input_thread_worker, daemon=True).start()

logging.info("Controls: m=Toggle Mouse mode, v=Doubao Prompt, r=reset, q=quit")

# ===== 4. Main loop =====
try:
    while True:
        t0 = time.time()

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
            
        color_bgr = np.asanyarray(color_frame.get_data())
        tracking_img = color_bgr.copy()

        # --- Reset ---
        if need_reset:
            if hasattr(sam2_predictor, 'condition_state') and 'point_inputs_per_obj' in sam2_predictor.condition_state:
                sam2_predictor.reset_state()
            num_targets = 0
            need_reset = False
            pending_bbox = None
            pending_point = None
            current_masks = {}
            all_prompts = []
            logging.info("Reset successful")

        # --- Doubao Grounding ---
        if doubao_prompt_trigger and not vlm_running:
            mouse_mode = False
            vlm_snapshot = color_bgr.copy()
            vlm_input_requested = True
            doubao_prompt_trigger = False

        # --- SAM2 Initialization ---
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
            
            # --- Key Fix: Use Snapshot for initialization to align with VLM coordinates ---
            if vlm_snapshot is not None:
                sam2_predictor.load_first_frame(vlm_snapshot)
                vlm_snapshot = None # Important: Reset so subsequent mouse prompts use current frame
            else:
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

        # --- SAM2 Track ---
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

        fps = 1.0 / (time.time() - t0)
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Display tracking info and controls
        if num_targets > 0:
            status = f"TRACKING {num_targets}/2 | r=reset q=quit"
        elif vlm_running:
            status = "VLM PROCESSING... | PLEASE WAIT"
        elif vlm_input_requested:
            status = ">>> ENTER PROMPT IN TERMINAL <<<"
        else:
            status = "Select up to 2 targets | q=quit"

        mode_text = f"Mode: {'MOUSE' if mouse_mode else 'VLM'} (m=mouse, v=doubao)"
        if vlm_running:
            mode_text += " | VLM PROCESSING..."
            
        cv2.putText(display, status, (10, IMG_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display, mode_text, (10, IMG_HEIGHT - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("D435 + Doubao + SAM2", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            need_reset = True
        elif key == ord('v'):
            doubao_prompt_trigger = True
        elif key == ord('m'):
            mouse_mode = not mouse_mode
            logging.info(f"Mouse mode: {'ENABLED' if mouse_mode else 'DISABLED'}")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
