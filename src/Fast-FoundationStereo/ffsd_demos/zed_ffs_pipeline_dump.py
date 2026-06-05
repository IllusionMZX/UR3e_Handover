"""
ZED 2i + FFS pipeline dump demo

Capture and save intermediate outputs:
left/right stereo -> disparity -> depth -> point cloud

Usage:
  conda activate ffs
  python zed_ffs_pipeline_dump.py
"""

from datetime import datetime
from pathlib import Path
import os
import sys
import time
import logging

import cv2
import numpy as np
import open3d as o3d
import pyzed.sl as sl
import torch
import yaml

FFS_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(FFS_DIR)

from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, vis_disparity, depth2xyzmap

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ===== Parameters =====
MODEL_DIR = os.path.join(FFS_DIR, "weights/23-36-37/model_best_bp2_serialize.pth")
VALID_ITERS = 8
MAX_DISP = 192
ZNEAR = 0.2
ZFAR = 5.0
IMG_W = 640
IMG_H = 360
SCALE_FACTOR = 0.5

OUT_ROOT = Path(__file__).resolve().parent / "outputs" / "zed_ffs_pipeline"


def add_colorbar_and_text(
    vis_bgr: np.ndarray,
    vmin: float,
    vmax: float,
    title: str,
    invalid_mask: np.ndarray | None = None,
) -> np.ndarray:
    h, w = vis_bgr.shape[:2]
    bar_w = 30
    gap = 10
    top_pad = 32
    bottom_pad = 26
    canvas_h = h + top_pad + bottom_pad
    canvas_w = w + gap + bar_w + 90
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    canvas[top_pad : top_pad + h, :w] = vis_bgr

    bar_x0 = w + gap
    bar_x1 = bar_x0 + bar_w
    grad = np.linspace(255, 0, h, dtype=np.uint8).reshape(h, 1)
    grad = np.repeat(grad, bar_w, axis=1)
    bar = cv2.applyColorMap(grad, cv2.COLORMAP_TURBO)
    canvas[top_pad : top_pad + h, bar_x0:bar_x1] = bar

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"{title}", (8, 24), font, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"max: {vmax:.3f}", (bar_x1 + 8, top_pad + 14), font, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"min: {vmin:.3f}",
        (bar_x1 + 8, top_pad + h - 6),
        font,
        0.45,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    if invalid_mask is not None:
        invalid_ratio = float(invalid_mask.mean()) * 100.0
        cv2.putText(
            canvas,
            f"invalid: {invalid_ratio:.1f}%",
            (8, canvas_h - 8),
            font,
            0.45,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return canvas


def save_depth_vis(depth_m: np.ndarray, out_path: Path) -> tuple[float, float]:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(valid):
        vis = np.full((depth_m.shape[0], depth_m.shape[1], 3), 255, dtype=np.uint8)
        dmin, dmax = ZNEAR, ZFAR
    else:
        # Invert depth for visualization: near=hot, far=cold.
        d = depth_m.copy()
        d[~valid] = 0
        dmin = max(ZNEAR, float(d[valid].min()))
        dmax = min(ZFAR, float(d[valid].max()))
        denom = max(dmax - dmin, 1e-6)
        inv = (1.0 - (d - dmin) / denom).clip(0, 1)
        vis = cv2.applyColorMap((inv * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        vis[~valid] = 255
        dmax = min(ZFAR, float(d[valid].max()))

    vis = add_colorbar_and_text(vis, dmin, dmax, "Depth (m)", invalid_mask=~valid)
    cv2.imwrite(str(out_path), vis)
    return dmin, dmax


def make_stereo_epiline_vis(
    left_bgr: np.ndarray, right_bgr: np.ndarray, line_count: int = 6
) -> np.ndarray:
    canvas = cv2.hconcat([left_bgr, right_bgr])
    h, w = canvas.shape[:2]
    step = h / (line_count + 1)
    color = (255, 220, 60)
    for i in range(1, line_count + 1):
        y = int(round(i * step))
        cv2.line(canvas, (0, y), (w - 1, y), color, 1, cv2.LINE_AA)
        # dashed effect
        for x in range(0, w, 24):
            x2 = min(x + 12, w - 1)
            cv2.line(canvas, (x, y), (x2, y), color, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Left", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 255, 40), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Right",
        (left_bgr.shape[1] + 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (40, 255, 40),
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        canvas,
        (left_bgr.shape[1], 0),
        (left_bgr.shape[1], h - 1),
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    torch.autograd.set_grad_enabled(False)

    logging.info("Loading FFS model...")
    with open(os.path.join(os.path.dirname(MODEL_DIR), "cfg.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    cfg["valid_iters"] = VALID_ITERS
    cfg["max_disp"] = MAX_DISP

    model = torch.load(MODEL_DIR, map_location="cpu", weights_only=False)
    model.args.valid_iters = VALID_ITERS
    model.args.max_disp = MAX_DISP
    model.cuda().eval()
    logging.info("FFS model loaded")

    logging.info("Initializing ZED camera...")
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.coordinate_units = sl.UNIT.METER

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera: {err}")

    image_left = sl.Mat()
    image_right = sl.Mat()

    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters
    left_cam = calib.left_cam
    baseline = calib.get_camera_baseline()
    K = np.array(
        [
            [left_cam.fx * SCALE_FACTOR, 0, left_cam.cx * SCALE_FACTOR],
            [0, left_cam.fy * SCALE_FACTOR, left_cam.cy * SCALE_FACTOR],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    logging.info(f"Scaled K:\n{K}")
    logging.info(f"Baseline: {baseline * 1000:.1f} mm")

    logging.info("Warming up model...")
    dummy_left = torch.randn(1, 3, IMG_H, IMG_W).cuda().float()
    dummy_right = torch.randn(1, 3, IMG_H, IMG_W).cuda().float()
    padder = InputPadder(dummy_left.shape, divis_by=32, force_square=False)
    dl, dr = padder.pad(dummy_left, dummy_right)
    with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        _ = model.forward(dl, dr, iters=VALID_ITERS, test_mode=True, optimize_build_volume="pytorch1")
    del dummy_left, dummy_right, dl, dr
    torch.cuda.empty_cache()
    logging.info("Warm-up complete")

    vis = o3d.visualization.Visualizer()
    vis.create_window("ZED + FFS Point Cloud", width=1280, height=720)
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.array([1.0, 1.0, 1.0])
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    first_frame = True

    cv2.namedWindow("ZED Preview", cv2.WINDOW_AUTOSIZE)
    logging.info("Press SPACE/c to capture and save pipeline outputs. Press q/ESC to quit.")

    try:
        while True:
            vis.poll_events()
            vis.update_renderer()

            if zed.grab() != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            zed.retrieve_image(image_right, sl.VIEW.RIGHT)

            left_bgr_full = image_left.get_data()[:, :, :3]
            right_bgr_full = image_right.get_data()[:, :, :3]
            left_bgr = cv2.resize(left_bgr_full, (IMG_W, IMG_H))
            right_bgr = cv2.resize(right_bgr_full, (IMG_W, IMG_H))

            cv2.imshow("ZED Preview", cv2.hconcat([left_bgr, right_bgr]))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key not in (32, ord("c")):
                continue

            t0 = time.time()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_dir = OUT_ROOT / f"capture_{ts}"
            out_dir.mkdir(parents=True, exist_ok=True)

            left_rgb = left_bgr[:, :, ::-1]
            right_rgb = right_bgr[:, :, ::-1]

            img0 = torch.as_tensor(left_rgb.copy()).cuda().float()[None].permute(0, 3, 1, 2)
            img1 = torch.as_tensor(right_rgb.copy()).cuda().float()[None].permute(0, 3, 1, 2)
            padder = InputPadder(img0.shape, divis_by=32, force_square=False)
            img0_p, img1_p = padder.pad(img0, img1)

            with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
                disp = model.forward(
                    img0_p, img1_p, iters=VALID_ITERS, test_mode=True, optimize_build_volume="pytorch1"
                )
            disp = padder.unpad(disp.float())
            disp = disp.data.cpu().numpy().reshape(IMG_H, IMG_W).clip(0, None)

            xx = np.arange(IMG_W)[None, :].repeat(IMG_H, axis=0)
            invalid = (xx - disp) < 0
            disp[invalid] = np.inf

            depth = K[0, 0] * baseline / disp
            depth[(depth < ZNEAR) | (depth > ZFAR) | ~np.isfinite(depth)] = 0

            xyz_map = depth2xyzmap(depth, K)
            points = xyz_map.reshape(-1, 3)
            colors = left_rgb.reshape(-1, 3)
            valid = points[:, 2] > 0
            points = points[valid]
            colors = colors[valid]

            if len(points) > 0:
                pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
                pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
                vis.update_geometry(pcd)
                if first_frame:
                    vis.reset_view_point(True)
                    ctr = vis.get_view_control()
                    ctr.set_front([0, 0, -1])
                    ctr.set_up([0, -1, 0])
                    first_frame = False
            else:
                pcd.points = o3d.utility.Vector3dVector()
                pcd.colors = o3d.utility.Vector3dVector()
                logging.warning("No valid 3D points in this capture.")

            cv2.imwrite(str(out_dir / "left.png"), left_bgr)
            cv2.imwrite(str(out_dir / "right.png"), right_bgr)
            stereo_lines = make_stereo_epiline_vis(left_bgr, right_bgr, line_count=6)
            cv2.imwrite(str(out_dir / "stereo_epilines.png"), stereo_lines)
            disp_stats = {}
            disp_vis_rgb = vis_disparity(disp, invalid_thres=np.inf, other_output=disp_stats)
            disp_valid = np.isfinite(disp) & (disp > 0)
            if disp_stats.get("min_val") is None or disp_stats.get("max_val") is None:
                disp_min, disp_max = 0.0, float(MAX_DISP)
            else:
                disp_min = float(disp_stats["min_val"])
                disp_max = float(disp_stats["max_val"])
            disp_vis_bgr = cv2.cvtColor(disp_vis_rgb, cv2.COLOR_RGB2BGR)
            disp_vis_bgr[~disp_valid] = 255
            disp_vis_bgr = add_colorbar_and_text(
                disp_vis_bgr, disp_min, disp_max, "Disparity (px)", invalid_mask=~disp_valid
            )
            cv2.imwrite(str(out_dir / "disparity.png"), disp_vis_bgr)
            depth_min, depth_max = save_depth_vis(depth, out_dir / "depth.png")
            o3d.io.write_point_cloud(str(out_dir / "pointcloud.ply"), pcd)

            logging.info(
                f"Saved pipeline outputs: {out_dir} | points={len(points)} | "
                f"disp=[{disp_min:.3f},{disp_max:.3f}] px | depth=[{depth_min:.3f},{depth_max:.3f}] m | "
                f"time={time.time()-t0:.3f}s"
            )
    finally:
        vis.destroy_window()
        cv2.destroyAllWindows()
        zed.close()
        logging.info("Exited")


if __name__ == "__main__":
    main()
