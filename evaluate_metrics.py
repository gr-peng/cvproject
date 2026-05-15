"""
Comprehensive evaluation: Part 1 (Baseline) vs Part 2 (SAM2+ProPainter)
Metrics: Mask IoU/Dice/Precision/Recall + Video PSNR/SSIM + Temporal Consistency
"""
import os, glob, json
import numpy as np
import cv2
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

PROJECT = Path(__file__).resolve().parent
EVAL_DIR = PROJECT / "data" / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────
def load_masks(d, n=None):
    fs = sorted(glob.glob(os.path.join(str(d), "*.png")))
    if n: fs = fs[:n]
    return [(cv2.imread(f, 0) > 127).astype(np.uint8) for f in fs]

def load_frames(d, n=None):
    fs = sorted(glob.glob(os.path.join(str(d), "*.[jp][pn]g")))
    if n: fs = fs[:n]
    return [cv2.imread(f) for f in fs]

def extract_video_frames(vp):
    """Extract frames from mp4."""
    cap = cv2.VideoCapture(str(vp))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()
    return frames

def iou(p, g):
    i = np.logical_and(p, g).sum()
    u = np.logical_or(p, g).sum()
    return i / u if u else 1.0

def dice(p, g):
    i = np.logical_and(p, g).sum()
    t = p.sum() + g.sum()
    return 2*i / t if t else 1.0

def precision_recall(p, g):
    tp = np.logical_and(p, g).sum()
    prec = tp / p.sum() if p.sum() else 1.0
    rec = tp / g.sum() if g.sum() else 1.0
    return prec, rec

# ── Mask evaluation ──────────────────────────────────────────────
def eval_masks(pred_dir, gt_dir, name):
    gt = load_masks(gt_dir)
    pred = load_masks(pred_dir, len(gt))
    ious, dices, precs, recs = [], [], [], []
    for s, g in zip(pred, gt):
        if s.shape != g.shape:
            s = cv2.resize(s, (g.shape[1], g.shape[0]), cv2.INTER_NEAREST)
        ious.append(iou(s, g))
        dices.append(dice(s, g))
        p, r = precision_recall(s, g)
        precs.append(p); recs.append(r)
    res = dict(name=name, n=len(gt),
               iou_mean=float(np.mean(ious)), iou_std=float(np.std(ious)),
               dice_mean=float(np.mean(dices)),
               precision=float(np.mean(precs)),
               recall=float(np.mean(recs)),
               ious=list(map(float, ious)))
    return res

# ── Video quality: PSNR / SSIM ──────────────────────────────────
def eval_video_quality(original_frames, inpaint_frames, masks, name):
    """
    Compute PSNR/SSIM on:
      - full frame (overall quality)
      - non-masked region only (background preservation)
    Also compute temporal consistency (consecutive frame PSNR in masked region)
    """
    n = min(len(original_frames), len(inpaint_frames), len(masks))
    psnr_full, ssim_full = [], []
    psnr_bg, ssim_bg = [], []
    temporal_psnr = []

    for i in range(n):
        orig = original_frames[i]
        inp = inpaint_frames[i]
        m = masks[i]

        # Resize if needed
        h, w = orig.shape[:2]
        if inp.shape[:2] != (h, w):
            inp = cv2.resize(inp, (w, h))
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

        # Full frame
        p = psnr_fn(orig, inp, data_range=255)
        s = ssim_fn(orig, inp, channel_axis=2, data_range=255)
        psnr_full.append(p); ssim_full.append(s)

        # Non-masked (background) region
        bg_mask = (m == 0)
        if bg_mask.sum() > 100:
            # Flatten to 1D for bg pixels
            bg_orig = orig[bg_mask]
            bg_inp = inp[bg_mask]
            p_bg = psnr_fn(bg_orig, bg_inp, data_range=255)
            psnr_bg.append(p_bg)
            # For SSIM on bg, use masked version
            orig_bg = orig.copy(); orig_bg[m > 0] = 0
            inp_bg = inp.copy(); inp_bg[m > 0] = 0
            s_bg = ssim_fn(orig_bg, inp_bg, channel_axis=2, data_range=255)
            ssim_bg.append(s_bg)

        # Temporal consistency (PSNR between consecutive inpainted frames in masked region)
        if i > 0:
            prev_inp = inpaint_frames[i-1]
            if prev_inp.shape[:2] != (h, w):
                prev_inp = cv2.resize(prev_inp, (w, h))
            if m.sum() > 100:
                tp = psnr_fn(prev_inp, inp, data_range=255)
                temporal_psnr.append(tp)

    res = dict(
        name=name, n=n,
        psnr_full=float(np.mean(psnr_full)) if psnr_full else 0,
        ssim_full=float(np.mean(ssim_full)) if ssim_full else 0,
        psnr_bg=float(np.mean(psnr_bg)) if psnr_bg else 0,
        ssim_bg=float(np.mean(ssim_bg)) if ssim_bg else 0,
        temporal_psnr=float(np.mean(temporal_psnr)) if temporal_psnr else 0,
        psnr_full_list=list(map(float, psnr_full)),
        ssim_full_list=list(map(float, ssim_full)),
    )
    return res

# ── Main evaluation ──────────────────────────────────────────────
def main():
    results = {}

    datasets = [
        ("bmx-trees", "modules/ProPainter/inputs/object_removal/bmx-trees_mask"),
        ("tennis", "modules/ProPainter/inputs/object_removal/tennis_mask"),
    ]

    methods = {
        "Part1_Baseline": {
            "mask_dir": "data/interim_masks/part1",
            "output_dir": "data/outputs/part1",
        },
        "Part2_SAM2_ProPainter": {
            "mask_dir": "data/interim_masks/part2",
            "output_dir": "data/outputs/part2",
        },
    }

    # Check for Part 3 outputs
    part3_out = PROJECT / "data" / "outputs" / "part3"
    if part3_out.exists() and any(part3_out.iterdir()):
        methods["Part3_SDXL_ProPainter"] = {
            "mask_dir": "data/interim_masks/part2",  # uses same masks as Part 2
            "output_dir": "data/outputs/part3",
        }

    all_mask_results = []
    all_video_results = []

    for ds_name, gt_mask_dir in datasets:
        raw_dir = PROJECT / "data" / "raw" / ds_name
        gt_dir = PROJECT / gt_mask_dir
        orig_frames = load_frames(str(raw_dir))

        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name} ({len(orig_frames)} frames)")
        print(f"{'='*60}")

        for method_name, paths in methods.items():
            mask_dir = PROJECT / paths["mask_dir"] / ds_name
            out_dir = PROJECT / paths["output_dir"] / ds_name

            if not mask_dir.exists():
                print(f"  [{method_name}] SKIP - no masks at {mask_dir}")
                continue

            # --- Mask evaluation ---
            if gt_dir.exists():
                mr = eval_masks(str(mask_dir), str(gt_dir), f"{method_name}/{ds_name}")
                all_mask_results.append(mr)
                print(f"  [{method_name}] Mask IoU={mr['iou_mean']:.4f}±{mr['iou_std']:.4f}  "
                      f"Dice={mr['dice_mean']:.4f}  Prec={mr['precision']:.4f}  Rec={mr['recall']:.4f}")

            # --- Video quality ---
            out_video = out_dir / "inpaint_out.mp4"
            if out_video.exists():
                inp_frames = extract_video_frames(str(out_video))
                masks = load_masks(str(mask_dir), len(orig_frames))
                vr = eval_video_quality(orig_frames, inp_frames, masks, f"{method_name}/{ds_name}")
                all_video_results.append(vr)
                print(f"  [{method_name}] PSNR(full)={vr['psnr_full']:.2f}dB  SSIM(full)={vr['ssim_full']:.4f}  "
                      f"PSNR(bg)={vr['psnr_bg']:.2f}dB  SSIM(bg)={vr['ssim_bg']:.4f}  "
                      f"Temporal={vr['temporal_psnr']:.2f}dB")
            else:
                print(f"  [{method_name}] SKIP video eval - no output at {out_video}")

    # Also evaluate running_car_short (no GT mask, only video quality between methods)
    rc_raw = PROJECT / "data" / "raw" / "running_car_short"
    if rc_raw.exists():
        orig_frames_rc = load_frames(str(rc_raw))
        ds_name = "running_car_short"
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name} ({len(orig_frames_rc)} frames) [No GT mask - video quality only]")
        print(f"{'='*60}")
        for method_name, paths in methods.items():
            mask_dir = PROJECT / paths["mask_dir"] / ds_name
            out_dir = PROJECT / paths["output_dir"] / ds_name
            out_video = out_dir / "inpaint_out.mp4"
            if out_video.exists() and mask_dir.exists():
                inp_frames = extract_video_frames(str(out_video))
                masks = load_masks(str(mask_dir), len(orig_frames_rc))
                vr = eval_video_quality(orig_frames_rc, inp_frames, masks, f"{method_name}/{ds_name}")
                all_video_results.append(vr)
                print(f"  [{method_name}] PSNR(full)={vr['psnr_full']:.2f}dB  SSIM(full)={vr['ssim_full']:.4f}  "
                      f"PSNR(bg)={vr['psnr_bg']:.2f}dB  Temporal={vr['temporal_psnr']:.2f}dB")

    # ── Save results ─────────────────────────────────────────────
    results = {"mask_metrics": all_mask_results, "video_metrics": all_video_results}
    with open(str(EVAL_DIR / "all_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {EVAL_DIR / 'all_results.json'}")

    # ── Generate comparison visualization ────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Mask IoU bar chart
        mask_data = {}
        for mr in all_mask_results:
            method, ds = mr['name'].split('/')
            mask_data.setdefault(ds, {})[method] = mr['iou_mean']

        if mask_data:
            fig, axes = plt.subplots(1, len(mask_data), figsize=(7*len(mask_data), 5))
            if len(mask_data) == 1:
                axes = [axes]
            for ax, (ds, mdata) in zip(axes, mask_data.items()):
                methods_list = list(mdata.keys())
                values = [mdata[m] for m in methods_list]
                colors = ['#ff6b6b', '#4ecdc4', '#45b7d1'][:len(methods_list)]
                bars = ax.bar(methods_list, values, color=colors, edgecolor='black', linewidth=0.5)
                ax.set_title(f'{ds} - Mask IoU', fontsize=13, fontweight='bold')
                ax.set_ylim(0, 1)
                ax.set_ylabel('IoU')
                ax.grid(axis='y', alpha=0.3)
                for bar, v in zip(bars, values):
                    ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f'{v:.3f}', ha='center', fontsize=11)
            plt.tight_layout()
            plt.savefig(str(EVAL_DIR / "mask_iou_comparison.png"), dpi=150)
            plt.close()
            print(f"Saved mask_iou_comparison.png")

        # Video PSNR/SSIM comparison
        vid_data = {}
        for vr in all_video_results:
            method, ds = vr['name'].split('/')
            vid_data.setdefault(ds, {})[method] = vr

        if vid_data:
            metrics_to_plot = [('psnr_full', 'PSNR (Full Frame) dB'), ('ssim_full', 'SSIM (Full Frame)'),
                               ('psnr_bg', 'PSNR (Background) dB'), ('temporal_psnr', 'Temporal PSNR dB')]
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()

            for ax, (metric, title) in zip(axes, metrics_to_plot):
                x_labels = []
                vals_by_method = {}
                for ds in vid_data:
                    for method in vid_data[ds]:
                        vals_by_method.setdefault(method, []).append(vid_data[ds][method][metric])
                    x_labels.append(ds)

                x = np.arange(len(x_labels))
                width = 0.35
                colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
                for i, (method, vals) in enumerate(vals_by_method.items()):
                    offset = (i - len(vals_by_method)/2 + 0.5) * width
                    bars = ax.bar(x + offset, vals, width, label=method, color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
                    for bar, v in zip(bars, vals):
                        ax.text(bar.get_x()+bar.get_width()/2, v+0.3, f'{v:.1f}', ha='center', fontsize=8)
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(x_labels, fontsize=9)
                ax.legend(fontsize=8)
                ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            plt.savefig(str(EVAL_DIR / "video_quality_comparison.png"), dpi=150)
            plt.close()
            print(f"Saved video_quality_comparison.png")

        # Per-frame IoU curves
        fig, axes = plt.subplots(1, len(datasets), figsize=(7*len(datasets), 5))
        if len(datasets) == 1:
            axes = [axes]
        colors_line = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        for ax, (ds_name, _) in zip(axes, datasets):
            for ci, mr in enumerate(all_mask_results):
                if ds_name in mr['name']:
                    label = mr['name'].split('/')[0]
                    ax.plot(mr['ious'], color=colors_line[ci % len(colors_line)],
                            linewidth=1.2, label=f"{label} (mean={mr['iou_mean']:.3f})")
            ax.set_title(f'{ds_name}: Per-frame IoU', fontsize=13, fontweight='bold')
            ax.set_xlabel('Frame')
            ax.set_ylabel('IoU')
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(EVAL_DIR / "iou_curves_all.png"), dpi=150)
        plt.close()
        print(f"Saved iou_curves_all.png")

        # Side-by-side frame comparison (orig | Part1 | Part2)
        for ds_name, gt_mask_dir in datasets:
            raw_dir = PROJECT / "data" / "raw" / ds_name
            raw_files = sorted(glob.glob(os.path.join(str(raw_dir), "*.[jp][pn]g")))
            n_frames = len(raw_files)
            sample_idxs = [0, n_frames//4, n_frames//2, 3*n_frames//4, n_frames-1]
            
            for idx in sample_idxs:
                row_images = []
                labels = []
                orig = cv2.imread(raw_files[idx])
                row_images.append(orig)
                labels.append("Original")

                for method_name, paths in methods.items():
                    out_video = PROJECT / paths["output_dir"] / ds_name / "inpaint_out.mp4"
                    if out_video.exists():
                        cap = cv2.VideoCapture(str(out_video))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        cap.release()
                        if ret:
                            if frame.shape[:2] != orig.shape[:2]:
                                frame = cv2.resize(frame, (orig.shape[1], orig.shape[0]))
                            row_images.append(frame)
                            labels.append(method_name.replace("_", "\n"))

                if len(row_images) > 1:
                    row = np.hstack(row_images)
                    h = orig.shape[0]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    for li, label in enumerate(labels):
                        x_pos = li * orig.shape[1] + 5
                        cv2.putText(row, label.split('\n')[0], (x_pos, 18), font, 0.45, (0,255,255), 1)
                    comp_dir = EVAL_DIR / "frame_comparisons" / ds_name
                    comp_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(comp_dir / f"frame_{idx:04d}.png"), row)

            print(f"Saved frame comparisons for {ds_name}")

    except Exception as e:
        print(f"Visualization error: {e}")
        import traceback; traceback.print_exc()

    # ── Summary table ────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"{'COMPREHENSIVE EVALUATION SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"\n{'Method':<30} {'Dataset':<18} {'IoU':>8} {'Dice':>8} {'PSNR':>8} {'SSIM':>8} {'Temp':>8}")
    print(f"{'-'*80}")
    for ds_name, _ in datasets:
        for method_name in methods:
            key = f"{method_name}/{ds_name}"
            mr = next((m for m in all_mask_results if m['name'] == key), None)
            vr = next((v for v in all_video_results if v['name'] == key), None)
            iou_s = f"{mr['iou_mean']:.4f}" if mr else "N/A"
            dice_s = f"{mr['dice_mean']:.4f}" if mr else "N/A"
            psnr_s = f"{vr['psnr_full']:.2f}" if vr else "N/A"
            ssim_s = f"{vr['ssim_full']:.4f}" if vr else "N/A"
            temp_s = f"{vr['temporal_psnr']:.2f}" if vr else "N/A"
            print(f"  {method_name:<28} {ds_name:<18} {iou_s:>8} {dice_s:>8} {psnr_s:>8} {ssim_s:>8} {temp_s:>8}")

    # running_car_short
    ds_name = "running_car_short"
    for method_name in methods:
        key = f"{method_name}/{ds_name}"
        vr = next((v for v in all_video_results if v['name'] == key), None)
        if vr:
            psnr_s = f"{vr['psnr_full']:.2f}"
            ssim_s = f"{vr['ssim_full']:.4f}"
            temp_s = f"{vr['temporal_psnr']:.2f}"
            print(f"  {method_name:<28} {ds_name:<18} {'N/A':>8} {'N/A':>8} {psnr_s:>8} {ssim_s:>8} {temp_s:>8}")

    print(f"{'='*80}")

if __name__ == "__main__":
    main()
