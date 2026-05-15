"""
Generate SAM3-refined masks for all datasets.
Pipeline: SAM2 masks (coarse) -> SAM3/SAM-vit refinement -> data/mask/sam3/

Requires SAM2 masks in data/mask/sam2/. Falls back to facebook/sam-vit-base if SAM3 is unavailable.

Usage:
    python generate_sam3_masks.py [--gpu cuda:1] [--datasets bmx-trees tennis wild_video]
                                  [--stride 3] [--extend_tracking]
"""

import sys
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SAM2_DIR = PROJECT_ROOT / "modules" / "sam2"
if str(SAM2_DIR) not in sys.path:
    sys.path.insert(0, str(SAM2_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


DATASETS = ["bmx-trees", "tennis", "wild_video"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu",             default="cuda:1")
    ap.add_argument("--datasets",        nargs="+", default=DATASETS)
    ap.add_argument("--coarse_source",   default="sam2",
                    help="Subfolder under data/mask/ to use as coarse input (default: sam2)")
    ap.add_argument("--stride",          type=int, default=3,
                    help="Run SAM on every N-th frame (default: 3)")
    ap.add_argument("--extend_tracking", action="store_true")
    ap.add_argument("--model_id",        default="facebook/sam3")
    args = ap.parse_args()

    from modules.part3.sam3_mask_refiner import refine_dataset_masks

    for ds in args.datasets:
        raw_dir    = str(PROJECT_ROOT / "data" / "raw"  / ds)
        coarse_dir = str(PROJECT_ROOT / "data" / "mask" / args.coarse_source / ds)
        out_dir    = str(PROJECT_ROOT / "data" / "mask" / "sam3" / ds)

        if not os.path.isdir(coarse_dir) or len(os.listdir(coarse_dir)) == 0:
            print(f"[SAM3] Coarse masks not found at {coarse_dir}, skipping {ds}.")
            print(f"       Run generate_sam2_masks.py first.")
            continue

        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"[SAM3] Dataset: {ds}")
        print(f"       raw_dir:    {raw_dir}")
        print(f"       coarse_dir: {coarse_dir}")
        print(f"       out_dir:    {out_dir}")
        print(f"{'='*50}")

        refine_dataset_masks(
            raw_dir=raw_dir,
            mask_dir=coarse_dir,
            output_dir=out_dir,
            device=args.gpu,
            model_id=args.model_id,
            concept="person",
            stride=args.stride,
            extend_tracking=args.extend_tracking,
        )
        print(f"[SAM3] Done: {ds} -> {out_dir}")

    print("\n[SAM3] All datasets complete.")


if __name__ == "__main__":
    main()
