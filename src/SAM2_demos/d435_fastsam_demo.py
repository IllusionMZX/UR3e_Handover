"""
RealSense D435 + FastSAM (Ultralytics) Real-time Segmentation Demo

Features:
  - Real-time RGB streaming from D435
  - FastSAM (via Ultralytics YOLOv8-seg)
  - Interactive Point Prompt selection (Click to select object)
  - Visualizes mask and contours

Usage:
  pip install ultralytics
  python d435_fastsam_demo.py
"""

import os, sys, time, logging
import numpy as np
import torch
import cv2
import pyrealsense2 as rs

from ultralytics import FastSAM
# We will dynamically handle the prompt tool later in the loop 
# to avoid path issues across different ultralytics versions

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ===== Parameters =====
MODEL_PATH = "FastSAM-s.pt"  # Will download automatically
IMG_WIDTH = 640
IMG_HEIGHT = 480
MASK_ALPHA = 0.5
MASK_COLOR_BGR = [75, 70, 203] # Red

# ===== 1. Load FastSAM model =====
logging.info(f"Loading FastSAM model ({MODEL_PATH})...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FastSAM(MODEL_PATH)
logging.info(f"FastSAM loaded on {device}")

# ===== 2. Initialize RealSense D435 =====
logging.info("Initializing RealSense D435...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, 30)

try:
    profile = pipeline.start(config)
except Exception as e:
    logging.error(f"Could not start RealSense pipeline: {e}")
    sys.exit(1)

# ===== 3. Interaction State =====
target_point = None  # (x, y)
current_mask = None

def mouse_callback(event, x, y, flags, param):
    global target_point
    if event == cv2.EVENT_LBUTTONDOWN:
        target_point = (x, y)
        logging.info(f"New target point set: {target_point}")

cv2.namedWindow("D435 + FastSAM Real-time", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("D435 + FastSAM Real-time", mouse_callback)

logging.info("Controls: Click on an object to segment it. Press 'q' to quit.")

# ===== 4. Main loop =====
try:
    while True:
        t0 = time.time()

        # Capture frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
            
        img = np.asanyarray(color_frame.get_data())
        
        # --- FastSAM Inference ---
        # 1. Everything mode (Generate all masks in the image)
        results = model(img, device=device, retina_masks=True, imgsz=640, conf=0.4, iou=0.9, verbose=False)
        
        # 2. Filter by Point Prompt
        if target_point is not None:
            # Replaced FastSAMPrompt with a more direct way using Predictor in newer Ultralytics
            # If FastSAMPrompt isn't available, we can manually find the mask that contains the point
            if current_mask is None or target_point is not None:
                masks = results[0].masks
                if masks is not None:
                    # masks.data is [N, H, W]
                    all_masks = masks.data.cpu().numpy().astype(bool)
                    # Find which mask contains the target point
                    # target_point is (x, y), image is (H, W)
                    px, py = target_point
                    # Ensure point is within bounds
                    if py < all_masks.shape[1] and px < all_masks.shape[2]:
                        found = False
                        for i in range(len(all_masks)):
                            if all_masks[i, int(py), int(px)]:
                                current_mask = all_masks[i]
                                found = True
                                break
                        if not found:
                            current_mask = None

        # --- Visualization ---
        display = img.copy()
        if current_mask is not None:
            # Resize mask if it doesn't match image (FastSAM might output at different scale)
            if current_mask.shape[:2] != (IMG_HEIGHT, IMG_WIDTH):
                current_mask_resized = cv2.resize(current_mask.astype(np.uint8), (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                current_mask_resized = current_mask

            overlay = display.copy()
            overlay[current_mask_resized] = MASK_COLOR_BGR
            display = cv2.addWeighted(display, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)
            
            # Draw contours
            contours, _ = cv2.findContours(current_mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, (0, 255, 0), 2)
        
        if target_point is not None:
            cv2.circle(display, target_point, 5, (0, 255, 255), -1)

        t1 = time.time()
        fps = 1.0 / (t1 - t0)
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("D435 + FastSAM Real-time", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    logging.info("Exited")
