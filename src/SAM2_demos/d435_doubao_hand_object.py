"""
RealSense D435 + Volcengine Doubao-Seed + SAM2 Dual Target Tracking

Features:
  - Real-time RGB streaming from D435
  - Volcengine Doubao-Seed (Multimodal) for dual-target selection:
    1. Human Hand
    2. Object held in the hand
  - Automatic error handling if hand is not detected
  - SAM2 interactive segmentation for both targets
  - Support for multi-object tracking

Usage:
  export ARK_API_KEY="your_api_key"
  python d435_doubao_hand_object.py

Controls:
  - v: Trigger Doubao prompt for hand and object selection
  - r: Reset all tracking targets
  - q: Quit
"""

import os, sys, time, logging, base64, json, re
import numpy as np
import torch
import cv2
import pyrealsense2 as rs
from volcenginesdkarkruntime import Ark
import threading

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
REASONING_EFFORT = "minimal" # Set to low/medium if needed for complex scenes

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
MASK_COLORS_BGR = {1: [75, 70, 203], 2: [203, 192, 75]} # 1: Hand (Reddish), 2: Object (Bluish)

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

# ===== 3. Interaction logic =====
cv2.namedWindow("Hand + Object Tracking", cv2.WINDOW_AUTOSIZE)

num_targets = 0 
need_reset = False
current_masks = {} 
all_prompts = [] 
doubao_trigger = False
pending_targets = [] # List of (id, bbox)
vlm_running = False

def call_doubao_hand_object(image_bgr):
    if not ARK_API_KEY:
        logging.error("ARK_API_KEY not set.")
        return None

    _, buffer = cv2.imencode('.jpg', image_bgr)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{img_base64}"

    # Complex prompt for dual-target grounding
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

        # Parse JSON from response
        try:
            # Clean possible markdown code blocks
            json_str = re.search(r'\{.*\}', content, re.DOTALL)
            if json_str:
                data = json.loads(json_str.group())
                results = []
                h, w = image_bgr.shape[:2]
                
                if "hand" in data:
                    ymin, xmin, ymax, xmax = data["hand"]
                    bbox = (int(xmin*w/1000), int(ymin*h/1000), int(xmax*w/1000), int(ymax*h/1000))
                    results.append((1, bbox)) # ID 1 for hand
                
                if "object" in data:
                    ymin, xmin, ymax, xmax = data["object"]
                    bbox = (int(xmin*w/1000), int(ymin*h/1000), int(xmax*w/1000), int(ymax*h/1000))
                    results.append((2, bbox)) # ID 2 for object
                
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

logging.info("Controls: v=Detect Hand & Object, r=Reset, q=Quit")

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
            current_masks = {}
            all_prompts = []
            pending_targets = []
            logging.info("Tracking reset.")

        # --- Doubao Grounding ---
        if doubao_trigger and not vlm_running:
            vlm_snapshot = color_bgr.copy() # Snapshot taken at the moment of trigger
            vlm_running = True
            threading.Thread(target=vlm_thread_worker, args=(vlm_snapshot,), daemon=True).start()
            doubao_trigger = False

        # --- SAM2 Initialization ---
        if pending_targets:
            # --- Key Fix: Use the original vlm_snapshot to initialize SAM2 ---
            # This ensures coordinates and visual features are perfectly aligned.
            if hasattr(sam2_predictor, 'condition_state') and 'point_inputs_per_obj' in sam2_predictor.condition_state:
                sam2_predictor.reset_state()
            
            # Load the frame where VLM detection actually occurred
            sam2_predictor.load_first_frame(vlm_snapshot)
            all_prompts = []
            
            for tid, bbox in pending_targets:
                x1, y1, x2, y2 = bbox
                bbox_arr = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                sam2_predictor.add_new_prompt(frame_idx=0, obj_id=tid, bbox=bbox_arr)
                all_prompts.append({'id': tid, 'bbox': bbox})
            
            num_targets = len(all_prompts)
            pending_targets = []

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
                
                # Draw label
                label = "Hand" if obj_id == 1 else "Object"
                # Find mask center for text
                pos = np.argwhere(mask)
                if len(pos) > 0:
                    y, x = pos[len(pos)//2]
                    cv2.putText(display, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        fps = 1.0 / (time.time() - t0)
        cv2.putText(display, f"FPS: {fps:.1f} | v=Detect r=Reset", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Hand + Object Tracking", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            need_reset = True
        elif key == ord('v'):
            doubao_trigger = True

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
