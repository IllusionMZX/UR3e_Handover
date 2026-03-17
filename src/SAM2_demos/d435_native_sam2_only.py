"""
RealSense D435 + SAM2 Tracking Real-time Segmentation Demo

Features:
  - Real-time RGB streaming from D435
  - SAM2 interactive segmentation (Box/Point)
  - Mask overlay and contour visualization

Usage:
  conda activate ffs
  python d435_native_sam2_only.py

Controls (focus on OpenCV window):
  - Left-click drag: Draw bounding box → initialize tracking
  - Left-click: Select foreground point → initialize tracking
  - r: Reset selection
  - q: Quit
"""

import os, sys, time, logging
import numpy as np
import torch
import cv2
import pyrealsense2 as rs

# SAM2 path
FFS_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SAM2_DIR = os.path.join(FFS_ROOT, "SAM2_streaming")
sys.path.insert(0, SAM2_DIR)
from sam2.build_sam import build_sam2_camera_predictor

logging.basicConfig(level=logging.INFO, format='%(message)s')

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
MASK_COLOR_BGR = [75, 70, 203]  # Red highlight (BGR)

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

profile = pipeline.start(config)
device = profile.get_device()
logging.info(f"Using device: {device.get_info(rs.camera_info.name)}")

# ===== 3. OpenCV window + mouse interaction =====
cv2.namedWindow("D435 + SAM2 Real-time", cv2.WINDOW_AUTOSIZE)

# Global state for dual-target tracking
drawing = False
ix, iy, fx_mouse, fy_mouse = -1, -1, -1, -1
pending_bbox = None
pending_point = None
num_targets = 0 
need_reset = False
current_masks = {} 
all_prompts = [] # Store all prompts to re-apply them if needed

# Colors for two targets
MASK_COLORS_BGR = {1: [75, 70, 203], 2: [203, 192, 75]} 

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

cv2.setMouseCallback("D435 + SAM2 Real-time", mouse_callback)

logging.info("Controls: Drag/click to select target, r=reset, q=quit")

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

        # --- SAM2: Reset ---
        if need_reset:
            # Check if condition_state is properly initialized before reset
            if hasattr(sam2_predictor, 'condition_state') and 'point_inputs_per_obj' in sam2_predictor.condition_state:
                sam2_predictor.reset_state()
            num_targets = 0
            need_reset = False
            pending_bbox = None
            pending_point = None
            current_masks = {}
            all_prompts = []
            logging.info("Reset, select new targets (up to 2)")

        # --- SAM2: Initialize targets (limit to 2) ---
        if (pending_bbox is not None or pending_point is not None) and num_targets < 2:
            # First, store the prompt
            target_id = num_targets + 1
            if pending_bbox is not None:
                all_prompts.append({'id': target_id, 'bbox': pending_bbox})
                pending_bbox = None
            elif pending_point is not None:
                all_prompts.append({'id': target_id, 'point': pending_point})
                pending_point = None
            num_targets += 1

            # IMPORTANT: Many SAM2 implementations don't allow add_new_prompt 
            # AFTER track() has been called for an existing session.
            # Safety check before reset_state to avoid KeyError in some internal states
            if hasattr(sam2_predictor, 'condition_state') and 'point_inputs_per_obj' in sam2_predictor.condition_state:
                sam2_predictor.reset_state()
            
            sam2_predictor.load_first_frame(tracking_img)
            
            for p in all_prompts:
                tid = p['id']
                if 'bbox' in p:
                    x1, y1, x2, y2 = p['bbox']
                    bbox_arr = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                    sam2_predictor.add_new_prompt(frame_idx=0, obj_id=tid, bbox=bbox_arr)
                    logging.info(f"Re-applied Target {tid} (bbox)")
                elif 'point' in p:
                    px, py = p['point']
                    pts_arr = np.array([[px, py]], dtype=np.float32)
                    lbl_arr = np.array([1], dtype=np.int32)
                    sam2_predictor.add_new_prompt(frame_idx=0, obj_id=tid, points=pts_arr, labels=lbl_arr)
                    logging.info(f"Re-applied Target {tid} (point)")

        # --- SAM2: Track ---
        if num_targets > 0:
            out_obj_ids, out_mask_logits = sam2_predictor.track(tracking_img)
            # Clear old masks for safety, though SAM2 usually handles this
            current_masks = {}
            for i in range(len(out_obj_ids)):
                obj_id = out_obj_ids[i]
                # Logits to mask [1, H, W] -> [H, W]
                current_masks[obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()

        # --- Visualization ---
        display = tracking_img.copy()
        for obj_id, mask in current_masks.items():
            if mask is not None:
                color = MASK_COLORS_BGR.get(obj_id, [0, 255, 0])
                overlay = display.copy()
                overlay[mask > 0] = color
                display = cv2.addWeighted(display, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)
                
                # Draw green contour for all
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

        if drawing and ix >= 0:
            cv2.rectangle(display, (ix, iy), (fx_mouse, fy_mouse), (255, 200, 0), 2)

        t1 = time.time()
        fps = 1.0 / (t1 - t0)
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display tracking info for dual targets
        if num_targets > 0:
            status = f"TRACKING {num_targets}/2 | r=reset q=quit"
        else:
            status = "Select up to 2 targets | q=quit"
        cv2.putText(display, status, (10, IMG_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("D435 + SAM2 Real-time", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            need_reset = True

except KeyboardInterrupt:
    pass
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    logging.info("Exited")
