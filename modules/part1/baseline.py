"""
Part 1: Baseline - Hand-crafted Video Object Removal Pipeline
YOLO v8 Segmentation + Optical Flow Dynamic Judgment + cv2.inpaint
"""

import os
import glob
import numpy as np
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class YOLOSegmentor:
    """YOLOv8 instance segmentation for mask extraction."""

    # COCO classes for dynamic objects
    DYNAMIC_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
        5: 'bus', 7: 'truck', 14: 'bird', 15: 'cat', 16: 'dog',
    }

    def __init__(self, model_name=None, device="cuda:0", conf=0.2):
        from ultralytics import YOLO
        if model_name is None:
            model_name = str(PROJECT_ROOT / "weights" / "part1" / "yolov8x-seg.pt")
        self.model = YOLO(model_name)
        self.model.to(device)
        self.device = device
        self.conf = conf
        print(f"[Baseline] YOLOv8-Seg loaded on {device}")

    def detect(self, frame):
        """Detect and return masks + bboxes for dynamic classes."""
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        masks = []
        bboxes = []
        classes = []
        if results.masks is not None:
            for i, cls_id in enumerate(results.boxes.cls.cpu().numpy().astype(int)):
                if cls_id in self.DYNAMIC_CLASSES:
                    mask = results.masks.data[i].cpu().numpy()
                    # Resize mask to frame size
                    h, w = frame.shape[:2]
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    masks.append(mask > 0.5)
                    bboxes.append(results.boxes.xyxy[i].cpu().numpy())
                    classes.append(cls_id)
        return masks, bboxes, classes


def compute_optical_flow_magnitude(prev_gray, curr_gray, mask):
    """
    Compute sparse optical flow (Lucas-Kanade) within mask region.
    Returns mean motion magnitude.
    """
    # Detect feature points within mask
    mask_u8 = (mask * 255).astype(np.uint8)
    corners = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=200, qualityLevel=0.01,
        minDistance=5, mask=mask_u8
    )
    if corners is None or len(corners) < 3:
        return 0.0

    # Lucas-Kanade optical flow
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, corners, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    good_old = corners[status.flatten() == 1]
    good_new = next_pts[status.flatten() == 1]
    if len(good_old) < 2:
        return 0.0

    displacements = np.linalg.norm(good_new - good_old, axis=1)
    return float(np.mean(displacements))


def dynamic_filter(masks, bboxes, classes, prev_gray, curr_gray, motion_threshold=0.8):
    """Filter masks: keep only dynamic objects (motion > threshold)."""
    dynamic_masks = []
    dynamic_meta = []
    for mask, bbox, cls_id in zip(masks, bboxes, classes):
        mag = compute_optical_flow_magnitude(prev_gray, curr_gray, mask)
        is_dynamic = mag > motion_threshold
        if is_dynamic:
            dynamic_masks.append(mask)
            dynamic_meta.append({
                'class': YOLOSegmentor.DYNAMIC_CLASSES.get(cls_id, 'unknown'),
                'motion': mag, 'bbox': bbox
            })
    return dynamic_masks, dynamic_meta


def merge_masks(masks, h, w):
    """Merge multiple binary masks into a single mask."""
    combined = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        combined = np.maximum(combined, m.astype(np.uint8))
    return combined


def dilate_mask(mask, kernel_size=7, iterations=3):
    """Apply morphological dilation to mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=iterations)


def temporal_background_inpaint(frames, masks, method='telea'):
    """
    Temporal-aware inpainting: borrow clean pixels from neighboring frames first,
    then fallback to cv2.inpaint for remaining holes.
    """
    n_frames = len(frames)
    results = []

    for i in range(n_frames):
        frame = frames[i].copy()
        mask = masks[i]
        if mask.sum() == 0:
            results.append(frame)
            continue

        mask_bool = mask > 0
        # Try to borrow from neighboring frames (look ±5 frames)
        borrowed = frame.copy()
        still_missing = mask_bool.copy()

        for offset in [1, -1, 2, -2, 3, -3, 5, -5]:
            j = i + offset
            if j < 0 or j >= n_frames:
                continue
            neighbor_mask = masks[j] > 0
            # Pixels that are clean in neighbor but masked in current
            can_borrow = still_missing & (~neighbor_mask)
            if can_borrow.any():
                borrowed[can_borrow] = frames[j][can_borrow]
                still_missing = still_missing & (~can_borrow)

        # Fallback: cv2.inpaint for remaining holes
        if still_missing.any():
            remaining_mask = still_missing.astype(np.uint8) * 255
            if method == 'telea':
                inpainted = cv2.inpaint(borrowed, remaining_mask, 10, cv2.INPAINT_TELEA)
            else:
                inpainted = cv2.inpaint(borrowed, remaining_mask, 10, cv2.INPAINT_NS)
            results.append(inpainted)
        else:
            results.append(borrowed)

    return results


def run_baseline_pipeline(
    video_dir, output_dir=None, device="cuda:0",
    conf=0.3, motion_threshold=2.0,
    dilation_kernel=7, dilation_iter=5,
    inpaint_method='telea', save_fps=24,
):
    """
    Full Part 1 baseline pipeline.
    
    Args:
        video_dir: Path to frame directory (JPG/PNG images).
        output_dir: Output directory.
        device: GPU for YOLO.
        conf: YOLO confidence threshold.
        motion_threshold: Optical flow motion threshold for dynamic filtering.
        dilation_kernel: Mask dilation kernel size.
        dilation_iter: Mask dilation iterations.
        inpaint_method: 'telea' or 'ns' (Navier-Stokes).
        save_fps: Output video FPS.
    """
    video_path = Path(video_dir)
    video_name = video_path.name
    
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "data" / "outputs" / "part1" / video_name)
    os.makedirs(output_dir, exist_ok=True)
    mask_dir = str(PROJECT_ROOT / "data" / "interim_masks" / "part1" / video_name)
    os.makedirs(mask_dir, exist_ok=True)

    # Load frames
    frame_files = sorted(glob.glob(os.path.join(str(video_path), "*.[jp][pn]g")))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {video_dir}")
    
    frames = [cv2.imread(f) for f in frame_files]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    h, w = frames[0].shape[:2]
    print(f"[Baseline] Loaded {len(frames)} frames ({w}x{h}) from {video_name}")

    # Step 1: YOLO segmentation + optical flow dynamic filtering
    print("[Baseline] Step 1: YOLO Segmentation + Optical Flow Dynamic Filter")
    segmentor = YOLOSegmentor(device=device, conf=conf)
    
    all_masks = []
    for i in range(len(frames)):
        masks_i, bboxes_i, classes_i = segmentor.detect(frames[i])
        
        if i > 0 and masks_i:
            # Apply optical flow dynamic filter
            dyn_masks, dyn_meta = dynamic_filter(
                masks_i, bboxes_i, classes_i,
                grays[i-1], grays[i], motion_threshold
            )
            if dyn_meta:
                for meta in dyn_meta:
                    pass  # could log
        else:
            # First frame: keep all detected objects
            dyn_masks = masks_i
        
        combined = merge_masks(dyn_masks, h, w) if dyn_masks else np.zeros((h, w), dtype=np.uint8)
        
        # Dilation
        if combined.sum() > 0:
            combined = dilate_mask(combined, dilation_kernel, dilation_iter)
        
        all_masks.append(combined)
        
        # Save mask
        cv2.imwrite(os.path.join(mask_dir, f"{i:05d}.png"), combined * 255)
    
    del segmentor
    import torch
    torch.cuda.empty_cache()
    
    mask_count = sum(1 for m in all_masks if m.sum() > 0)
    print(f"[Baseline] Generated {mask_count}/{len(all_masks)} non-empty masks")

    # Step 1.5: Morphological refinement + temporal smoothing (reuse Part 2 preprocessor)
    print("[Baseline] Step 1.5: Mask refinement (closing + temporal smoothing)")
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from modules.part2.preprocessor import refine_masks
        refine_masks(
            mask_dir=mask_dir,
            dilation_kernel_size=dilation_kernel,
            dilation_iterations=2,
            use_closing=True,
            closing_kernel_size=11,
            temporal_smooth=True,
            temporal_window=5,
            adaptive_dilation=True,
        )
        # Reload refined masks
        refined_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        if len(refined_files) == len(all_masks):
            all_masks = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) // 255 for f in refined_files]
            print(f"[Baseline] Reloaded {len(all_masks)} refined masks")
    except Exception as e:
        print(f"[Baseline] Warning: mask refinement failed ({e}), using raw masks")

    # Step 2: Temporal Background Inpainting
    print(f"[Baseline] Step 2: Temporal Background + cv2.inpaint ({inpaint_method})")
    results = temporal_background_inpaint(frames, all_masks, method=inpaint_method)

    # Save output video
    out_path = os.path.join(output_dir, "inpaint_out.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, save_fps, (w, h))
    for r in results:
        writer.write(r)
    writer.release()
    
    # Save masked input video for visualization
    masked_path = os.path.join(output_dir, "masked_in.mp4")
    writer2 = cv2.VideoWriter(masked_path, fourcc, save_fps, (w, h))
    for frame, mask in zip(frames, all_masks):
        vis = frame.copy()
        vis[mask > 0] = [255, 255, 255]
        writer2.write(vis)
    writer2.release()

    print(f"[Baseline] Done! Output: {out_path}")
    return output_dir, mask_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Part 1: Baseline Pipeline")
    parser.add_argument("--video", required=True, help="Frame directory path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--motion_threshold", type=float, default=0.8)
    parser.add_argument("--dilation_kernel", type=int, default=7)
    parser.add_argument("--dilation_iter", type=int, default=5)
    parser.add_argument("--inpaint_method", default="telea", choices=["telea", "ns"])
    parser.add_argument("--save_fps", type=int, default=24)
    args = parser.parse_args()

    run_baseline_pipeline(
        video_dir=args.video, device=args.device,
        conf=args.conf, motion_threshold=args.motion_threshold,
        dilation_kernel=args.dilation_kernel, dilation_iter=args.dilation_iter,
        inpaint_method=args.inpaint_method, save_fps=args.save_fps,
    )
