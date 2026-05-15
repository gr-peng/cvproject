"""
SAM 2 Video Tracker Module
Encapsulates SAM 2 video prediction for dynamic object mask extraction.
Supports bidirectional propagation for improved mask quality.
"""

import os
import sys
import numpy as np
import cv2
import torch
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Add SAM2 to path
SAM2_DIR = PROJECT_ROOT / "modules" / "sam2"
if str(SAM2_DIR) not in sys.path:
    sys.path.insert(0, str(SAM2_DIR))


class VideoTracker:
    """SAM 2 based video object tracker with bidirectional propagation."""

    def __init__(
        self,
        device: str = "cuda:0",
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        checkpoint: str = None,
    ):
        self.device = device
        if checkpoint is None:
            checkpoint = str(PROJECT_ROOT / "weights" / "part2" / "sam2" / "sam2.1_hiera_large.pt")
        self.checkpoint = checkpoint
        self.model_cfg = model_cfg
        self.predictor = None

    def _load_model(self):
        if self.predictor is not None:
            return
        from sam2.build_sam import build_sam2_video_predictor

        self.predictor = build_sam2_video_predictor(
            self.model_cfg,
            self.checkpoint,
            device=self.device,
        )
        print(f"[Tracker] SAM 2 loaded on {self.device}")

    def _extract_frames(self, video_path: str, frame_dir: str) -> int:
        """Extract video frames to a temporary JPEG directory (SAM2 requirement)."""
        os.makedirs(frame_dir, exist_ok=True)
        # Check if frames already extracted (avoid re-extraction)
        existing = [f for f in os.listdir(frame_dir) if f.endswith(".jpg")]
        if existing:
            print(f"[Tracker] Reusing {len(existing)} existing frames in {frame_dir}")
            return len(existing)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(frame_dir, f"{idx:05d}.jpg"), frame)
            idx += 1
        cap.release()
        print(f"[Tracker] Extracted {idx} frames from {video_path}")
        return idx

    def _propagate_direction(self, inference_state, reverse=False):
        """Run propagation in one direction and collect masks."""
        segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
            inference_state, reverse=reverse
        ):
            segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        return segments

    def _merge_segments(self, forward_segments, backward_segments):
        """Merge forward and backward propagation results via union."""
        all_frames = set(forward_segments.keys()) | set(backward_segments.keys())
        merged = {}
        for frame_idx in all_frames:
            fwd = forward_segments.get(frame_idx, {})
            bwd = backward_segments.get(frame_idx, {})
            all_obj_ids = set(fwd.keys()) | set(bwd.keys())
            merged[frame_idx] = {}
            for obj_id in all_obj_ids:
                fwd_mask = fwd.get(obj_id)
                bwd_mask = bwd.get(obj_id)
                if fwd_mask is not None and bwd_mask is not None:
                    # Union of forward and backward masks
                    merged[frame_idx][obj_id] = np.logical_or(fwd_mask, bwd_mask)
                elif fwd_mask is not None:
                    merged[frame_idx][obj_id] = fwd_mask
                else:
                    merged[frame_idx][obj_id] = bwd_mask
        return merged

    def process(
        self,
        video: str,
        prompts: dict,
        output_dir: str = None,
        frame_dir: str = None,
        bidirectional: bool = True,
    ) -> str:
        """
        Run SAM 2 video tracking with optional bidirectional propagation.

        Args:
            video: Path to input video file or frame directory.
            prompts: Dict mapping obj_id -> dict with keys:
                - 'points': list of [x, y] coords
                - 'labels': list of 1=positive, 0=negative
                - 'frame_idx': int, the frame to annotate (default 0)
                Optional:
                - 'box': list [x_min, y_min, x_max, y_max]
            output_dir: Where to save masks.
            frame_dir: Temp dir for extracted frames.
            bidirectional: If True, propagate both forward and backward from the
                           prompt frame and merge results. This significantly
                           improves mask quality for frames before the prompt.

        Returns:
            output_dir path.
        """
        self._load_model()

        video_name = Path(video).stem
        is_frame_dir = os.path.isdir(video)

        if output_dir is None:
            output_dir = str(PROJECT_ROOT / "data" / "interim_masks" / video_name)
        os.makedirs(output_dir, exist_ok=True)

        if is_frame_dir:
            # SAM2 requires JPEG frames. If the directory has non-JPEG files
            # (e.g. PNG), convert them to a temp JPEG directory.
            has_jpg = any(f.endswith('.jpg') for f in os.listdir(video))
            if has_jpg:
                frame_dir = video
            else:
                if frame_dir is None:
                    frame_dir = str(PROJECT_ROOT / "data" / "temp_frames" / video_name)
                os.makedirs(frame_dir, exist_ok=True)
                existing_jpgs = [f for f in os.listdir(frame_dir) if f.endswith('.jpg')]
                if not existing_jpgs:
                    src_files = sorted([f for f in os.listdir(video) if f.endswith(('.png', '.bmp', '.tif', '.tiff'))])
                    for f in src_files:
                        img = cv2.imread(os.path.join(video, f))
                        if img is not None:
                            jpg_name = Path(f).stem + '.jpg'
                            cv2.imwrite(os.path.join(frame_dir, jpg_name), img)
                    print(f"[Tracker] Converted {len(src_files)} frames to JPEG in {frame_dir}")
                else:
                    print(f"[Tracker] Reusing {len(existing_jpgs)} existing JPEG frames in {frame_dir}")
        else:
            if frame_dir is None:
                frame_dir = str(PROJECT_ROOT / "data" / "temp_frames" / video_name)
            self._extract_frames(video, frame_dir)

        # Initialize SAM2 video state
        with torch.inference_mode():
            inference_state = self.predictor.init_state(video_path=frame_dir)

            # Add prompts for each object.
            # Each value may be a single dict OR a list of dicts (multi-frame prompts).
            for obj_id, prompt_or_list in prompts.items():
                prompt_list = prompt_or_list if isinstance(prompt_or_list, list) else [prompt_or_list]
                for prompt in prompt_list:
                    frame_idx = prompt.get("frame_idx", 0)
                    points = prompt.get("points", None)
                    labels = prompt.get("labels", None)
                    box = prompt.get("box", None)

                    pts_np = np.array(points, dtype=np.float32) if points is not None else None
                    lbl_np = np.array(labels, dtype=np.int32)   if labels is not None else None
                    box_np = np.array(box,    dtype=np.float32) if box    is not None else None

                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=int(obj_id),
                        points=pts_np,
                        labels=lbl_np,
                        box=box_np,
                    )
                    print(f"[Tracker] Added prompt for obj {obj_id} on frame {frame_idx}")

            # Propagate through video
            if bidirectional:
                print("[Tracker] Running bidirectional propagation (forward + backward)")
                forward_segments = self._propagate_direction(inference_state, reverse=False)
                backward_segments = self._propagate_direction(inference_state, reverse=True)
                video_segments = self._merge_segments(forward_segments, backward_segments)
                print(f"[Tracker] Forward: {len(forward_segments)} frames, Backward: {len(backward_segments)} frames")
            else:
                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                    inference_state
                ):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

        # Save combined masks (union of all object masks)
        for frame_idx in sorted(video_segments.keys()):
            masks = video_segments[frame_idx]
            combined = None
            for obj_id, mask in masks.items():
                if combined is None:
                    combined = mask.astype(np.uint8)
                else:
                    combined = np.maximum(combined, mask.astype(np.uint8))
            if combined is not None:
                mask_img = (combined * 255).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(output_dir, f"{frame_idx:05d}.png"), mask_img
                )

        print(f"[Tracker] Saved {len(video_segments)} masks to {output_dir}")
        return output_dir
