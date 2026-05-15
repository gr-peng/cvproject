"""
Mask Preprocessor Module
Applies morphological operations and temporal smoothing to refine SAM 2 masks.
"""

import os
import cv2
import numpy as np
from pathlib import Path


def _adaptive_dilation_size(mask: np.ndarray, base_kernel: int, min_kernel: int = 3, max_kernel: int = 15) -> int:
    """Compute adaptive dilation kernel size based on mask area ratio."""
    area_ratio = np.sum(mask > 127) / mask.size
    if area_ratio < 0.01:
        return max_kernel  # small object -> more aggressive dilation to cover edges
    elif area_ratio > 0.15:
        return min_kernel  # large object -> minimal dilation to avoid over-expansion
    return base_kernel


def refine_masks(
    mask_dir: str,
    output_dir: str = None,
    dilation_kernel_size: int = 7,
    dilation_iterations: int = 2,
    gaussian_blur_size: int = 0,
    use_closing: bool = True,
    closing_kernel_size: int = 9,
    temporal_smooth: bool = True,
    temporal_window: int = 5,
    adaptive_dilation: bool = True,
) -> str:
    """
    Refine binary masks with morphological operations and temporal smoothing.

    Args:
        mask_dir: Directory containing mask PNGs (00000.png, 00001.png, ...).
        output_dir: Output directory. If None, overwrites in-place.
        dilation_kernel_size: Base dilation kernel size (odd number).
        dilation_iterations: Number of dilation iterations.
        gaussian_blur_size: Gaussian blur kernel size (0 = disabled, must be odd).
        use_closing: Apply morphological closing to fill holes in masks.
        closing_kernel_size: Kernel size for morphological closing.
        temporal_smooth: Apply temporal smoothing across frames to reduce flicker.
        temporal_window: Window size for temporal smoothing (odd number).
        adaptive_dilation: Adapt dilation size based on mask area.

    Returns:
        Output directory path.
    """
    if output_dir is None:
        output_dir = mask_dir
    os.makedirs(output_dir, exist_ok=True)

    mask_files = sorted(
        [f for f in os.listdir(mask_dir) if f.endswith(".png")]
    )
    if not mask_files:
        raise FileNotFoundError(f"No mask files found in {mask_dir}")

    # Phase 1: Load and apply per-frame morphological refinement
    masks = []
    for fname in mask_files:
        mask = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            masks.append(None)
            continue

        # Binarize
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Morphological closing to fill holes inside the mask
        if use_closing:
            close_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

        # Adaptive or fixed dilation
        if adaptive_dilation:
            k_size = _adaptive_dilation_size(mask, dilation_kernel_size)
        else:
            k_size = dilation_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)

        # Optional Gaussian blur for soft edges
        if gaussian_blur_size > 0:
            blur_size = gaussian_blur_size if gaussian_blur_size % 2 == 1 else gaussian_blur_size + 1
            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        masks.append(mask)

    # Phase 2: Temporal smoothing to reduce mask flicker between frames
    if temporal_smooth and len(masks) > 1:
        half_w = temporal_window // 2
        smoothed = []
        for i in range(len(masks)):
            if masks[i] is None:
                smoothed.append(None)
                continue
            start = max(0, i - half_w)
            end = min(len(masks), i + half_w + 1)
            valid_masks = [masks[j] for j in range(start, end) if masks[j] is not None]
            if not valid_masks:
                smoothed.append(masks[i])
                continue
            # Average the masks and re-binarize — this smooths out single-frame noise
            avg = np.mean(np.stack(valid_masks, axis=0).astype(np.float32), axis=0)
            # Threshold at 0.3 (lower than 0.5) to be conservative — include a pixel
            # if it's masked in at least ~1/3 of the window
            _, smoothed_mask = cv2.threshold(avg, 0.3 * 255, 255, cv2.THRESH_BINARY)
            smoothed.append(smoothed_mask.astype(np.uint8))
        masks = smoothed

    # Phase 3: Save refined masks
    saved_count = 0
    for fname, mask in zip(mask_files, masks):
        if mask is not None:
            cv2.imwrite(os.path.join(output_dir, fname), mask)
            saved_count += 1

    print(f"[Preprocessor] Refined {saved_count} masks in {output_dir}")
    print(f"[Preprocessor]   closing={use_closing}, adaptive_dilation={adaptive_dilation}, temporal_smooth={temporal_smooth}")
    return output_dir
