#!/usr/bin/env python3
"""
Video Object Removal & Inpainting Pipeline
============================================
Main entry point that orchestrates:
  1. SAM 2 video tracking (GPU 0) -> bidirectional mask extraction
  2. Mask refinement (CPU) -> adaptive dilation, closing, temporal smoothing
  3. (Optional) SDXL generative fill (GPU 3) -> persistent occlusion inpainting
  4. ProPainter inpainting (GPU 1) -> temporal video inpainting with keyframe blending

Usage:
    python pipeline.py --video data/raw/tennis.mp4 \\
                       --points "[[350,400]]" --labels "[1]" \\
                       [--use_sdxl]

    python pipeline.py --video data/raw/bmx-trees \\
                       --points "[[150,200]]" --labels "[1]" \\
                       --frame_idx 0
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Video Object Removal & Inpainting Pipeline"
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to input video file or frame directory."
    )
    parser.add_argument(
        "--points", type=str, default=None,
        help='JSON list of [x,y] click points. Auto-detected if --auto_prompt is used.'
    )
    parser.add_argument(
        "--labels", type=str, default="[1]",
        help='JSON list of labels (1=positive, 0=negative)'
    )
    parser.add_argument(
        "--frame_idx", type=int, default=0,
        help="Frame index to annotate the prompt on (default: 0)."
    )
    parser.add_argument(
        "--box", type=str, default=None,
        help='Optional bounding box [x_min,y_min,x_max,y_max], e.g. "[100,50,400,350]"'
    )
    parser.add_argument(
        "--obj_ids", type=str, default="[1]",
        help='JSON list of object IDs, e.g. "[1,2]". Must match number of point groups.'
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for final results."
    )
    # Mask refinement
    parser.add_argument(
        "--dilation_kernel", type=int, default=7,
        help="Base dilation kernel size (default: 7)."
    )
    parser.add_argument(
        "--dilation_iter", type=int, default=2,
        help="Dilation iterations (default: 2)."
    )
    parser.add_argument(
        "--no_closing", action="store_true",
        help="Disable morphological closing for mask holes."
    )
    parser.add_argument(
        "--no_temporal_smooth", action="store_true",
        help="Disable temporal smoothing of masks."
    )
    parser.add_argument(
        "--no_adaptive_dilation", action="store_true",
        help="Disable adaptive dilation (use fixed kernel size)."
    )
    parser.add_argument(
        "--temporal_window", type=int, default=5,
        help="Temporal smoothing window size (default: 5)."
    )
    # ProPainter
    parser.add_argument(
        "--neighbor_length", type=int, default=10,
        help="ProPainter neighbor length (default: 10)."
    )
    parser.add_argument(
        "--subvideo_length", type=int, default=80,
        help="ProPainter subvideo length (default: 80)."
    )
    parser.add_argument(
        "--ref_stride", type=int, default=10,
        help="ProPainter global reference stride (default: 10)."
    )
    parser.add_argument(
        "--pp_mask_dilation", type=int, default=8,
        help="ProPainter internal mask dilation (default: 8)."
    )
    parser.add_argument(
        "--save_fps", type=int, default=24,
        help="Output video FPS."
    )
    parser.add_argument(
        "--no_fp16", action="store_true",
        help="Disable fp16 inference for ProPainter."
    )
    # Tracker
    parser.add_argument(
        "--no_bidirectional", action="store_true",
        help="Disable bidirectional propagation in SAM 2 tracker."
    )
    # GPU assignment
    parser.add_argument("--gpu_tracker", type=str, default="cuda:0")
    parser.add_argument("--gpu_inpainter", type=str, default="cuda:1")
    parser.add_argument("--gpu_sdxl", type=str, default="cuda:3")
    # Part 3
    parser.add_argument(
        "--use_sdxl", action="store_true",
        help="Enable SDXL generative fill for Part 3."
    )
    # Skip steps
    parser.add_argument(
        "--skip_tracking", action="store_true",
        help="Skip mask extraction (reuse existing masks)."
    )
    parser.add_argument(
        "--skip_refinement", action="store_true",
        help="Skip mask refinement."
    )
    parser.add_argument(
        "--auto_prompt", action="store_true",
        help="Auto-detect target object via YOLO and use as SAM2 prompt (overrides --points/--labels)."
    )

    args = parser.parse_args()

    video_path = args.video
    video_name = Path(video_path).stem

    # Parse points & labels (or auto-detect via YOLO)
    obj_ids = json.loads(args.obj_ids)

    if args.auto_prompt:
        import glob, numpy as np, cv2 as _cv2
        from ultralytics import YOLO as _YOLO
        _DYNAMIC_CLASSES = {0,1,2,3,4,5,6,7,14,15,16,17}
        frame_files = sorted(glob.glob(os.path.join(video_path, "*.[jp][pn]g")))
        if not frame_files:
            frame_files = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
        _frame0 = _cv2.imread(frame_files[args.frame_idx] if frame_files else video_path)
        _yolo_model = _YOLO(str(PROJECT_ROOT / "weights" / "part1" / "yolov8x-seg.pt"))
        _results = _yolo_model(_frame0, conf=0.2, verbose=False)[0]
        _candidates = []
        if _results.boxes is not None:
            for _i, _cls in enumerate(_results.boxes.cls.cpu().numpy().astype(int)):
                if _cls in _DYNAMIC_CLASSES:
                    _box = _results.boxes.xyxy[_i].cpu().numpy()
                    _cx = int((_box[0]+_box[2])//2)
                    _cy = int((_box[1]+_box[3])//2)
                    _area = float((_box[2]-_box[0])*(_box[3]-_box[1]))
                    _conf = float(_results.boxes.conf[_i])
                    _candidates.append((_conf*_area, _cx, _cy, _YOLO.model.names[_cls] if hasattr(_YOLO, "model") else str(_cls)))
        del _yolo_model
        if _candidates:
            _candidates.sort(reverse=True)
            # Take top-2 candidates to cover compound objects (e.g., person+bike)
            _top = _candidates[:min(2, len(_candidates))]
            points = [[c[1], c[2]] for c in _top]
            labels = [1] * len(points)
            print(f"[Auto-prompt] Detected {len(_candidates)} candidates, using top-{len(points)} as prompts: {points}")
        else:
            print("[Auto-prompt] No dynamic objects detected! Falling back to default center.")
            _h, _w = _frame0.shape[:2]
            points = [[_w//2, _h//2]]
            labels = [1]
    elif args.skip_tracking:
        # When skipping tracking, points are irrelevant (masks already exist)
        points = [[0, 0]]
        labels = [1]
    elif args.points is None:
        raise ValueError("Must specify --points or use --auto_prompt")
    else:
        points = json.loads(args.points)
        labels = json.loads(args.labels)

    mask_dir = str(PROJECT_ROOT / "data" / "interim_masks" / "part2" / video_name)
    if args.output_dir is None:
        if args.use_sdxl:
            output_dir = str(PROJECT_ROOT / "data" / "outputs" / "part3")
        else:
            output_dir = str(PROJECT_ROOT / "data" / "outputs" / "part2")
    else:
        output_dir = args.output_dir

    # ==========================================================
    # Step 1: SAM 2 Mask Extraction (GPU 0)
    # ==========================================================
    if not args.skip_tracking:
        print("=" * 60)
        print("Step 1: SAM 2 Mask Extraction")
        print("=" * 60)
        from modules.part2.tracker import VideoTracker

        tracker = VideoTracker(device=args.gpu_tracker)

        # Build prompts dict
        prompts = {}
        if len(obj_ids) == 1:
            # Single object with all points
            prompt_data = {
                "points": points,
                "labels": labels,
                "frame_idx": args.frame_idx,
            }
            if args.box:
                prompt_data["box"] = json.loads(args.box)
            prompts[obj_ids[0]] = prompt_data
        else:
            # Multiple objects: assume points split evenly
            pts_per_obj = len(points) // len(obj_ids)
            for i, oid in enumerate(obj_ids):
                start = i * pts_per_obj
                end = start + pts_per_obj
                prompts[oid] = {
                    "points": points[start:end],
                    "labels": labels[start:end],
                    "frame_idx": args.frame_idx,
                }

        mask_dir = tracker.process(
            video=video_path,
            prompts=prompts,
            output_dir=mask_dir,
            bidirectional=not args.no_bidirectional,
        )

        # Free GPU memory
        del tracker
        import torch
        torch.cuda.empty_cache()
    else:
        print("[Pipeline] Skipping tracking, reusing masks from:", mask_dir)

    # ==========================================================
    # Step 2: Mask Refinement (CPU)
    # ==========================================================
    if not args.skip_refinement:
        print("\n" + "=" * 60)
        print("Step 2: Mask Refinement")
        print("=" * 60)
        from modules.part2.preprocessor import refine_masks

        refine_masks(
            mask_dir=mask_dir,
            dilation_kernel_size=args.dilation_kernel,
            dilation_iterations=args.dilation_iter,
            use_closing=not args.no_closing,
            temporal_smooth=not args.no_temporal_smooth,
            temporal_window=args.temporal_window,
            adaptive_dilation=not args.no_adaptive_dilation,
        )
    else:
        print("[Pipeline] Skipping mask refinement.")

    # ==========================================================
    # Step 3 (Optional): SDXL Generative Fill (GPU 3)
    # ==========================================================
    gen_keyframe_dir = None
    if args.use_sdxl:
        print("\n" + "=" * 60)
        print("Step 3: SDXL Generative Inpainting (Part 3)")
        print("=" * 60)
        from modules.part3.generative_fill import generative_inpaint_keyframes

        gen_keyframe_dir = generative_inpaint_keyframes(
            video_path=video_path,
            mask_dir=mask_dir,
            device=args.gpu_sdxl,
            keyframe_stride=5,
            area_threshold=0.005,
            window_size=3,
            model_id="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        )

    # ==========================================================
    # Step 4: ProPainter Video Inpainting (GPU 1)
    # ==========================================================
    print("\n" + "=" * 60)
    print("Step 4: ProPainter Video Inpainting")
    print("=" * 60)
    from modules.part2.inpainter import ProPainterEngine

    inpainter = ProPainterEngine(device=args.gpu_inpainter)
    inpainter.restore(
        video=video_path,
        mask=mask_dir,
        output=output_dir,
        neighbor_length=args.neighbor_length,
        subvideo_length=args.subvideo_length,
        ref_stride=args.ref_stride,
        mask_dilation=args.pp_mask_dilation,
        save_fps=args.save_fps,
        fp16=not args.no_fp16,
        gen_keyframe_dir=gen_keyframe_dir,
    )

    print("\n" + "=" * 60)
    print(f"Pipeline complete! Results saved to: {output_dir}/{video_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
