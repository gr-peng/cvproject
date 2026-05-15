"""
Generative Inpainting Module (Part 3 Extension)
Uses SDXL inpainting for cases where temporal propagation fails.
Improved with better keyframe selection, mask padding, and higher quality settings.
"""

import os
import numpy as np
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def detect_persistent_occlusion(
    mask_dir: str,
    window_size: int = 10,
    area_threshold: float = 0.03,
    keyframe_stride: int = 15,
) -> list:
    """
    Detect frames where the mask covers a significant area persistently,
    indicating that temporal propagation may not have clean reference frames.

    Uses a stride-based approach to select evenly-spaced keyframes in
    persistently occluded regions instead of only picking window centers.

    Returns:
        List of keyframe indices suitable for generative inpainting.
    """
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])
    if not mask_files:
        return []

    mask_areas = []
    for fname in mask_files:
        mask = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask_areas.append(0.0)
            continue
        area_ratio = np.sum(mask > 127) / mask.size
        mask_areas.append(area_ratio)

    # Find contiguous runs of persistently occluded frames
    persistent = [a > area_threshold for a in mask_areas]

    # Smooth: require at least `window_size` consecutive frames above threshold
    smoothed = [False] * len(persistent)
    for i in range(len(persistent)):
        start = max(0, i - window_size // 2)
        end = min(len(persistent), i + window_size // 2 + 1)
        if all(persistent[j] for j in range(start, end)):
            smoothed[i] = True

    # Select keyframes at regular stride within persistent regions
    keyframes = []
    in_run = False
    run_start = 0
    for i in range(len(smoothed) + 1):
        if i < len(smoothed) and smoothed[i]:
            if not in_run:
                run_start = i
                in_run = True
        else:
            if in_run:
                # Select keyframes evenly within this run
                run_len = i - run_start
                if run_len <= keyframe_stride:
                    keyframes.append(run_start + run_len // 2)
                else:
                    for k in range(run_start, i, keyframe_stride):
                        keyframes.append(min(k + keyframe_stride // 2, i - 1))
                in_run = False

    # Deduplicate and sort
    keyframes = sorted(set(keyframes))
    return keyframes


def generative_inpaint_keyframes(
    video_path: str,
    mask_dir: str,
    output_dir: str = None,
    device: str = "cuda:3",
    keyframes: list = None,
    prompt: str = "clean background, high quality, photorealistic, detailed",
    negative_prompt: str = "blurry, artifacts, text, watermark, person, human, distorted",
    num_inference_steps: int = 40,
    guidance_scale: float = 8.0,
    strength: float = 0.99,
    mask_padding: int = 30,
    keyframe_stride: int = 15,
    area_threshold: float = 0.03,
    window_size: int = 10,
    model_id: str = "runwayml/stable-diffusion-inpainting",
) -> str:
    """
    Use SDXL inpainting to fill keyframes where temporal propagation fails.

    Improvements over baseline:
    - Mask padding to give the model more context at boundaries
    - Higher inference steps and guidance scale for quality
    - Strength near 1.0 to fully regenerate masked regions
    - Better prompt engineering

    Args:
        video_path: Input video path.
        mask_dir: Directory with binary masks.
        output_dir: Where to save inpainted keyframes.
        device: GPU device for SDXL.
        keyframes: List of frame indices, or None to auto-detect.
        prompt: Text prompt for SDXL inpainting.
        negative_prompt: Negative prompt.
        num_inference_steps: Number of denoising steps (higher = better quality).
        guidance_scale: Classifier-free guidance scale.
        strength: Inpainting strength (0-1, higher = more regeneration).
        mask_padding: Pixels to pad the mask by to avoid boundary artifacts.

    Returns:
        Output directory with inpainted keyframe images.
    """
    import torch
    from PIL import Image
    from diffusers import AutoPipelineForInpainting

    video_name = Path(video_path).stem
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "data" / "generative_keyframes" / video_name)
    os.makedirs(output_dir, exist_ok=True)

    if keyframes is None:
        keyframes = detect_persistent_occlusion(
            mask_dir,
            window_size=window_size,
            area_threshold=area_threshold,
            keyframe_stride=keyframe_stride,
        )

    if not keyframes:
        print("[GenerativeFill] No persistent occlusion detected, skipping.")
        return output_dir

    print(f"[GenerativeFill] Processing {len(keyframes)} keyframes on {device}")

    # Extract needed frames from video
    is_frame_dir = os.path.isdir(video_path)
    frame_dict = {}

    if is_frame_dir:
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))])
        keyframe_set = set(keyframes)
        for fname in frame_files:
            idx = int(Path(fname).stem)
            if idx in keyframe_set:
                frame = cv2.imread(os.path.join(video_path, fname))
                if frame is not None:
                    frame_dict[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        keyframe_set = set(keyframes)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in keyframe_set:
                frame_dict[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            idx += 1
        cap.release()

    # Load inpainting pipeline (SD 1.5 by default; SDXL if model_id specified)
    # Always use fp16 safetensors to avoid PyTorch >=2.6 requirement for .bin files
    load_kwargs = {"torch_dtype": torch.float16, "use_safetensors": True, "variant": "fp16"}
    pipe = AutoPipelineForInpainting.from_pretrained(
        model_id,
        **load_kwargs,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

    for kf_idx in keyframes:
        if kf_idx not in frame_dict or kf_idx >= len(mask_files):
            continue

        frame_rgb = frame_dict[kf_idx]
        image = Image.fromarray(frame_rgb)

        # Load and prepare mask with padding
        mask_path = os.path.join(mask_dir, mask_files[kf_idx])
        mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_np is None:
            continue

        # Pad the mask to give the model more context at boundaries
        if mask_padding > 0:
            pad_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (mask_padding * 2 + 1, mask_padding * 2 + 1)
            )
            mask_np = cv2.dilate(mask_np, pad_kernel, iterations=1)

        mask = Image.fromarray(mask_np).convert("L")

        # Resize to multiples of 8 (required by SDXL)
        w, h = image.size
        target_w = (w // 8) * 8
        target_h = (h // 8) * 8
        image_resized = image.resize((target_w, target_h), Image.LANCZOS)
        mask_resized = mask.resize((target_w, target_h), Image.NEAREST)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_resized,
            mask_image=mask_resized,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
        ).images[0]

        # Resize back to original and save
        result = result.resize((w, h), Image.LANCZOS)
        result.save(os.path.join(output_dir, f"{kf_idx:05d}.png"))
        print(f"[GenerativeFill] Inpainted keyframe {kf_idx}")

    del pipe
    torch.cuda.empty_cache()

    print(f"[GenerativeFill] Done. Saved {len(keyframes)} keyframes to {output_dir}")
    return output_dir
