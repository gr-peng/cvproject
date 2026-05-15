"""
Compare SAM2 / SAM3 / personal masks against GT (real) masks.

Step 1: Resize ALL masks to GT resolution and save to data/mask/unified/.
Step 2: Compute IoU from the unified masks.

GT is in data/mask/real/{dataset}_mask/  (resolution: 240x432 for bmx-trees/tennis)
wild_video has no GT; its unified masks are saved at native 480x854.

Usage:
    python eval_mask_iou.py
"""
import os
import numpy as np
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Mapping: dataset_name -> GT subfolder name
GT_MAP = {
    "bmx-trees": "bmx-trees_mask",
    "tennis":    "tennis_mask",
    # wild_video has no GT
}

ALL_DATASETS = ["bmx-trees", "tennis", "wild_video"]

METHODS = {
    "personal": "personal",
    "sam2":     "sam2",
    "sam3":     "sam3",
}


def get_gt_hw(ds):
    """Return (H, W) of the GT masks for a dataset, or None if no GT."""
    if ds not in GT_MAP:
        return None
    gt_dir = PROJECT_ROOT / "data" / "mask" / "real" / GT_MAP[ds]
    if not gt_dir.is_dir():
        return None
    files = sorted(f for f in os.listdir(gt_dir) if f.endswith(".png"))
    if not files:
        return None
    m = cv2.imread(str(gt_dir / files[0]), cv2.IMREAD_GRAYSCALE)
    return m.shape if m is not None else None  # (H, W)


def save_unified_masks(src_dir, dst_dir, target_hw):
    """
    Read all PNG masks from src_dir, resize to target_hw (H, W) if needed,
    and write binary (0/255) PNGs to dst_dir.
    Returns list of binary uint8 arrays at target_hw.
    """
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".png"))
    masks = []
    for f in files:
        m = cv2.imread(os.path.join(src_dir, f), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != target_hw:
            m = cv2.resize(m, (target_hw[1], target_hw[0]),
                           interpolation=cv2.INTER_NEAREST)
        binary = ((m > 127).astype(np.uint8)) * 255
        cv2.imwrite(os.path.join(dst_dir, f), binary)
        masks.append((binary > 0).astype(np.uint8))
    return masks


def compute_iou_stats(pred_masks, gt_masks):
    n = min(len(pred_masks), len(gt_masks))
    ious = []
    for i in range(n):
        p, g = pred_masks[i], gt_masks[i]
        inter = int(np.logical_and(p, g).sum())
        union = int(np.logical_or(p, g).sum())
        ious.append(inter / union if union > 0 else 1.0)
    return np.array(ious)


def main():
    unified_root = PROJECT_ROOT / "data" / "mask" / "unified"

    # ── Step 1: save unified masks ──────────────────────────────────────────
    print("=== Step 1: 统一分辨率并保存到 data/mask/unified/ ===")
    for ds in ALL_DATASETS:
        gt_hw = get_gt_hw(ds)
        if gt_hw is None:
            # wild_video: no GT, use native resolution from sam2
            sam2_dir = PROJECT_ROOT / "data" / "mask" / "sam2" / ds
            files = sorted(f for f in os.listdir(sam2_dir) if f.endswith(".png"))
            m0 = cv2.imread(str(sam2_dir / files[0]), cv2.IMREAD_GRAYSCALE)
            gt_hw = m0.shape  # e.g. (480, 854)
            label = f"native {gt_hw[0]}x{gt_hw[1]} (no GT)"
        else:
            label = f"GT {gt_hw[0]}x{gt_hw[1]}"

        for method_name, method_sub in METHODS.items():
            src = PROJECT_ROOT / "data" / "mask" / method_sub / ds
            dst = unified_root / method_sub / ds
            if not src.is_dir():
                continue
            orig_files = sorted(f for f in os.listdir(src) if f.endswith(".png"))
            orig_m = cv2.imread(str(src / orig_files[0]), cv2.IMREAD_GRAYSCALE) if orig_files else None
            orig_hw = orig_m.shape if orig_m is not None else "?"
            masks = save_unified_masks(str(src), str(dst), gt_hw)
            changed = "→ resized" if orig_hw != gt_hw else "  (already ok)"
            print(f"  {ds}/{method_name:<10}: {orig_hw} {changed} → {gt_hw}  "
                  f"({len(masks)} frames, {label})")

    # Also copy real GT into unified/ for reference
    for ds, gt_sub in GT_MAP.items():
        src = PROJECT_ROOT / "data" / "mask" / "real" / gt_sub
        dst = unified_root / "real" / ds
        gt_hw = get_gt_hw(ds)
        if gt_hw:
            save_unified_masks(str(src), str(dst), gt_hw)

    print()

    # ── Step 2: compute IoU from unified masks ───────────────────────────────
    print("=== Step 2: IoU 指标（基于 unified/ 目录） ===")
    print("=" * 65)
    print(f"{'Dataset':<14} {'Method':<12} {'N':>4} {'Mean IoU':>10} "
          f"{'Min IoU':>9} {'Max IoU':>9}")
    print("=" * 65)

    for ds in GT_MAP:
        gt_dir = unified_root / "real" / ds
        if not gt_dir.is_dir():
            continue
        gt_files = sorted(f for f in os.listdir(gt_dir) if f.endswith(".png"))
        gt_masks = []
        for f in gt_files:
            m = cv2.imread(str(gt_dir / f), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                gt_masks.append((m > 127).astype(np.uint8))
        if not gt_masks:
            continue

        for method_name, method_sub in METHODS.items():
            pred_dir = unified_root / method_sub / ds
            if not pred_dir.is_dir():
                continue
            pred_files = sorted(f for f in os.listdir(pred_dir) if f.endswith(".png"))
            pred_masks = []
            for f in pred_files:
                m = cv2.imread(str(pred_dir / f), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    pred_masks.append((m > 127).astype(np.uint8))
            if not pred_masks:
                continue

            ious = compute_iou_stats(pred_masks, gt_masks)
            print(f"  {ds:<14} {method_name:<12} {len(ious):>4} "
                  f"{np.mean(ious):>10.4f} {np.min(ious):>9.4f} "
                  f"{np.max(ious):>9.4f}")

        print()

    print("=" * 65)
    for ds in GT_MAP:
        hw = get_gt_hw(ds)
        if hw:
            print(f"  * {ds}: 统一分辨率 {hw[0]}x{hw[1]}")
    print("  * wild_video 无 GT，unified/ 中保存为 480x854")
    print(f"  * 已统一保存至: {unified_root}")


if __name__ == "__main__":
    main()
