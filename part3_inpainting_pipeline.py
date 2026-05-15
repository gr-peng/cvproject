#!/usr/bin/env python3
"""
Unified Inpainting Pipeline (Part 3 new)
=========================================
Uses pre-generated masks from data/mask/unified/ to run:

  SAM2 masks  -> ProPainter (1x)            -> part3_new/sam2_propainter/
  SAM3 masks  -> ProPainter (1x)            -> part3_new/sam3_propainter/
  SAM3 PP out -> SD base model (pre-FT)     -> part3_new/sam3_pp_sd_base/
  SAM3 PP out -> SD LoRA fine-tuned (FT)    -> part3_new/sam3_pp_sd_lora/

SAM2 = 1 inpainting pass, SAM3 = 3 inpainting passes total.

Usage:
  python run_inpainting_pipeline.py \
      [--datasets bmx-trees tennis wild_video] \
      [--gpu_pp cuda:0] [--gpu_sd cuda:1] \
      [--skip_sam2_pp] [--skip_sam3_pp] [--skip_sd_base] [--skip_sd_lora]
"""

import os
import sys

# ── Fix sys.path: ensure conda env torch is found before user-local torch ──────
# User-local packages (/home/pgr/.local/...) may shadow conda env, causing
# torch version conflicts that break diffusers.
_conda_sp = "/public/software/anaconda3/envs/cvproject/lib/python3.10/site-packages"
if _conda_sp in sys.path:
    sys.path.remove(_conda_sp)
sys.path.insert(0, _conda_sp)
# ────────────────────────────────────────────────────────────────────────────────

import argparse
import subprocess
import shutil
import cv2
import numpy as np
import torch
from pathlib import Path

# path setup
_conda_sp = "/public/software/anaconda3/envs/cvproject/lib/python3.10/site-packages"
if _conda_sp not in sys.path:
    sys.path.insert(0, _conda_sp)

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "modules"))

PROPAINTER_DIR = PROJECT / "modules" / "ProPainter"
PP_WEIGHTS_DIR = PROJECT / "weights" / "part2" / "propainter"

DATASETS = ["bmx-trees", "tennis", "wild_video"]

PROMPT_MAP = {
    "bmx-trees":  "forest trail background, dirt path, green trees, natural daylight, photorealistic, 4k",
    "tennis":     "clay tennis court background, red clay surface, baseline markings, stadium, photorealistic",
    "wild_video": "road background, asphalt street, urban scene, natural lighting, photorealistic",
}
NEG_PROMPT = (
    "person, human, face, hands, artifacts, blur, distortion, watermark, "
    "text, overexposed, deformed, low quality, cartoon"
)
SEED_MAP = {"bmx-trees": 2025, "tennis": 2026, "wild_video": 2027}

# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_frames_dir(d):
    files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    frames, names = [], []
    for f in files:
        img = cv2.imread(os.path.join(d, f))
        if img is not None:
            frames.append(img)
            names.append(f)
    return frames, names


def read_video_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames


def load_masks(mask_dir, n=None):
    files = sorted(f for f in os.listdir(mask_dir) if f.endswith(".png"))
    if n is not None:
        files = files[:n]
    result = []
    for f in files:
        m = cv2.imread(os.path.join(mask_dir, f), cv2.IMREAD_GRAYSCALE)
        result.append((m > 127).astype(np.uint8) if m is not None else None)
    return result


def save_video(frames, path, fps, w, h):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        if f is None:
            continue
        f2 = cv2.resize(f, (w, h)) if f.shape[:2] != (h, w) else f
        writer.write(f2)
    writer.release()
    print(f"  Saved: {path}")


def get_video_fps(path, default=24.0):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else default


# ─────────────────────────────────────────────────────────────────────────────
# ProPainter inference
# ─────────────────────────────────────────────────────────────────────────────

def run_propainter(frames_dir, mask_dir, output_dir, gpu="cuda:0",
                   height=-1, width=-1, fp16=True,
                   neighbor_length=10, subvideo_length=80,
                   ref_stride=10, mask_dilation=4, save_fps=24):
    # Ensure weights are symlinked inside modules/ProPainter/weights/
    pp_w = PROPAINTER_DIR / "weights"
    pp_w.mkdir(exist_ok=True)
    for wname in ["ProPainter.pth", "recurrent_flow_completion.pth", "raft-things.pth"]:
        dst = pp_w / wname
        if not dst.exists():
            for src_candidate in [PP_WEIGHTS_DIR / wname, PROJECT / "weights" / wname]:
                if src_candidate.exists():
                    os.symlink(str(src_candidate), str(dst))
                    break

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROPAINTER_DIR / "inference_propainter.py"),
        "--video",           str(Path(frames_dir).resolve()),
        "--mask",            str(Path(mask_dir).resolve()),
        "--output",          str(Path(output_dir).resolve()),
        "--height",          str(height),
        "--width",           str(width),
        "--neighbor_length", str(neighbor_length),
        "--subvideo_length", str(subvideo_length),
        "--ref_stride",      str(ref_stride),
        "--mask_dilation",   str(mask_dilation),
        "--save_fps",        str(save_fps),
    ]
    if fp16:
        cmd.append("--fp16")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu.replace("cuda:", "")

    print(f"\n[ProPainter] GPU={gpu}")
    print(f"[ProPainter] frames={frames_dir}")
    print(f"[ProPainter] mask={mask_dir}")
    print(f"[ProPainter] output={output_dir}")
    result = subprocess.run(cmd, cwd=str(PROPAINTER_DIR), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"ProPainter failed (exit code {result.returncode})")
    print(f"[ProPainter] Done.")
    return Path(output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# SD refinement
# ─────────────────────────────────────────────────────────────────────────────

def artifact_score(frame, mask):
    if mask is None or mask.sum() == 0:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    m = mask.astype(bool)
    if m.sum() < 16:
        return 0.0
    return float(min(1.0, max(0.0, (np.abs(lap[m]).mean() / (np.abs(lap[~m]).mean() + 1e-8) - 1.0) * 0.5)))


def select_refinement_frames(pp_frames, masks, max_frames=12, min_mask_area=0.02):
    candidates = []
    for i, (f, m) in enumerate(zip(pp_frames, masks)):
        if m is None or f is None:
            continue
        if m.sum() / m.size < min_mask_area:
            continue
        candidates.append((i, artifact_score(f, m)))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = set(i for i, s in candidates if s >= 0.05)
    by_time = sorted(candidates, key=lambda x: x[0])
    if by_time:
        selected.add(by_time[0][0])
        selected.add(by_time[-1][0])

    if len(selected) < max_frames:
        remaining = [i for i, _ in by_time if i not in selected]
        budget = max_frames - len(selected)
        if remaining:
            idxs = np.linspace(0, len(remaining) - 1, budget, dtype=int)
            selected.update(remaining[j] for j in idxs)

    result = sorted(selected)[:max_frames]
    print(f"  [SD] {len(result)} frames selected for refinement")
    return result


def load_sd_pipeline(model_id, lora_weights, device):
    from diffusers import StableDiffusionInpaintPipeline

    print(f"[SD] Loading '{model_id}' on {device} ...")
    # Use cached model; avoid network calls that fail with SOCKS proxy
    _load_kw = dict(torch_dtype=torch.float16, safety_checker=None, local_files_only=True)
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id, variant="fp16", **_load_kw,
        ).to(device)
    except Exception:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id, **_load_kw,
        ).to(device)

    if lora_weights and Path(lora_weights).exists():
        print(f"[SD] Loading LoRA from {lora_weights} ...")
        try:
            from peft import PeftModel
            pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_weights)
            print("[SD] LoRA loaded via PeftModel.")
        except Exception as e:
            print(f"[SD] LoRA load failed ({e}); using base model.")

    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    return pipe


def sd_refine_frames(pp_frames, orig_frames, masks, frame_idxs, pipe, device,
                     prompt, seed=2025, num_steps=30, guidance=7.5,
                     strength=0.55, mask_dilation=15):
    from PIL import Image
    results = {}
    dil_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mask_dilation * 2 + 1, mask_dilation * 2 + 1))

    for idx in frame_idxs:
        if idx >= len(pp_frames) or pp_frames[idx] is None:
            continue
        mask = masks[idx] if idx < len(masks) else None
        if mask is None or mask.sum() == 0:
            continue

        base_rgb = cv2.cvtColor(pp_frames[idx], cv2.COLOR_BGR2RGB)
        h, w = base_rgb.shape[:2]
        mk_dilated = cv2.dilate((mask * 255).astype(np.uint8), dil_kern)
        tw, th = (w // 8) * 8, (h // 8) * 8
        image_pil = Image.fromarray(base_rgb).resize((tw, th), Image.LANCZOS)
        mask_pil  = Image.fromarray(mk_dilated).resize((tw, th), Image.NEAREST)

        try:
            gen = torch.Generator(device=device).manual_seed(seed + idx)
            out = pipe(
                prompt=prompt, negative_prompt=NEG_PROMPT,
                image=image_pil, mask_image=mask_pil,
                num_inference_steps=num_steps, guidance_scale=guidance,
                strength=strength, generator=gen,
            ).images[0]
            results[idx] = cv2.cvtColor(
                np.array(out.resize((w, h), Image.LANCZOS)), cv2.COLOR_RGB2BGR
            )
            print(f"    frame {idx:3d} done")
        except Exception as e:
            print(f"    frame {idx:3d} failed: {e}")

    return results


def blend_sd_into_pp(pp_frames, sd_results, masks, h, w, feather_sigma=7.0):
    output = []
    for f in pp_frames:
        if f is None:
            output.append(np.zeros((h, w, 3), np.uint8))
        elif f.shape[:2] != (h, w):
            output.append(cv2.resize(f, (w, h)))
        else:
            output.append(f.copy())

    sorted_kf = sorted(sd_results.keys())
    if not sorted_kf:
        return output

    def _blend_at(idx, sd_bgr, weight=1.0):
        if idx >= len(output):
            return
        mk = masks[idx] if idx < len(masks) else None
        if mk is None:
            return
        mk_f = mk.astype(np.float32)
        if mk_f.shape != (h, w):
            mk_f = cv2.resize(mk_f, (w, h))
        sf = cv2.resize(sd_bgr, (w, h)) if sd_bgr.shape[:2] != (h, w) else sd_bgr
        blur = cv2.GaussianBlur(mk_f, (0, 0), feather_sigma)
        mk3 = np.stack([blur * weight] * 3, axis=-1)
        output[idx] = (mk3 * sf + (1.0 - mk3) * output[idx]).clip(0, 255).astype(np.uint8)

    for kf in sorted_kf:
        _blend_at(kf, sd_results[kf])

    for ki in range(len(sorted_kf) - 1):
        a, b = sorted_kf[ki], sorted_kf[ki + 1]
        if b - a <= 1:
            continue
        for t in range(a + 1, b):
            if t >= len(output):
                break
            mk = masks[t] if t < len(masks) else None
            if mk is None or mk.sum() == 0:
                continue
            alpha = (t - a) / (b - a)
            cos_w = 0.5 * (1 - np.cos(np.pi * alpha))
            sd_a = cv2.resize(sd_results[a], (w, h))
            sd_b = cv2.resize(sd_results[b], (w, h))
            interp = ((1 - cos_w) * sd_a + cos_w * sd_b).clip(0, 255).astype(np.uint8)
            _blend_at(t, interp)

    for t in range(sorted_kf[0]):
        mk = masks[t] if t < len(masks) else None
        if mk is not None and mk.sum() > 0:
            _blend_at(t, sd_results[sorted_kf[0]])
    for t in range(sorted_kf[-1] + 1, len(output)):
        mk = masks[t] if t < len(masks) else None
        if mk is not None and mk.sum() > 0:
            _blend_at(t, sd_results[sorted_kf[-1]])

    return output


def run_sd_on_pp_output(dataset, pp_out_dir, mask_dir, output_dir, gpu,
                        sd_model, lora_weights=None, sd_strength=0.55,
                        sd_steps=30, max_refine_frames=12):
    tag = "LoRA" if lora_weights else "base"
    print(f"\n{'='*60}")
    print(f"  SD [{tag}]  dataset={dataset}  gpu={gpu}")
    print(f"{'='*60}")

    # ProPainter nests output one extra level: output/{dataset}/inpaint_out.mp4
    pp_vid = None
    for _cand in [
        Path(pp_out_dir) / "inpaint_out.mp4",
        Path(pp_out_dir) / dataset / "inpaint_out.mp4",
    ]:
        if _cand.exists():
            pp_vid = _cand
            break
    if pp_vid is None:
        print(f"[!] ProPainter output not found under: {pp_out_dir}")
        return
    print(f"  PP video: {pp_vid}")

    orig_frames, _ = read_frames_dir(str(PROJECT / "data" / "raw" / dataset))
    if not orig_frames:
        print(f"[!] No raw frames for {dataset}")
        return

    n = len(orig_frames)
    h0, w0 = orig_frames[0].shape[:2]
    fps = get_video_fps(str(pp_vid))

    pp_frames = read_video_frames(str(pp_vid))
    pp_frames = [cv2.resize(f, (w0, h0)) if f.shape[:2] != (h0, w0) else f for f in pp_frames]
    while len(pp_frames) < n:
        pp_frames.append(pp_frames[-1].copy() if pp_frames else orig_frames[-1].copy())
    pp_frames = pp_frames[:n]

    masks_bin = load_masks(str(mask_dir), n)
    while len(masks_bin) < n:
        masks_bin.append(np.zeros((h0, w0), np.uint8))

    refine_idxs = select_refinement_frames(pp_frames, masks_bin, max_refine_frames)
    if not refine_idxs:
        print("  No frames need refinement; copying PP output.")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy(str(pp_vid), str(Path(output_dir) / "inpaint_out.mp4"))
        return

    pipe = load_sd_pipeline(sd_model, lora_weights, gpu)
    prompt = PROMPT_MAP.get(dataset, "clean natural background, photorealistic")
    seed   = SEED_MAP.get(dataset, 2025)

    print(f"\n  Prompt  : {prompt[:80]}")
    print(f"  Strength: {sd_strength}  Steps: {sd_steps}  Frames: {len(refine_idxs)}")

    sd_results = sd_refine_frames(
        pp_frames=pp_frames, orig_frames=orig_frames, masks=masks_bin,
        frame_idxs=refine_idxs, pipe=pipe, device=gpu,
        prompt=prompt, seed=seed, num_steps=sd_steps,
        guidance=7.5, strength=sd_strength,
    )

    del pipe
    torch.cuda.empty_cache()

    final_frames = blend_sd_into_pp(pp_frames, sd_results, masks_bin, h0, w0)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_video(final_frames, str(output_dir / "inpaint_out.mp4"), fps, w0, h0)

    # masked-overlay visualization
    masked_vis = []
    for frm, mk in zip(orig_frames, masks_bin):
        vis = frm.copy()
        if mk is not None:
            vis[mk > 0] = [255, 255, 255]
        masked_vis.append(vis)
    save_video(masked_vis, str(output_dir / "masked_in.mp4"), fps, w0, h0)

    print(f"  [SD/{tag}] Done -> {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Unified inpainting pipeline (SAM2/SAM3 + ProPainter + SD)")
    ap.add_argument("--datasets",          nargs="+", default=DATASETS)
    ap.add_argument("--gpu_pp",            default="cuda:0",  help="GPU for ProPainter")
    ap.add_argument("--gpu_sd",            default="cuda:1",  help="GPU for SD")
    ap.add_argument("--sd_model",          default="runwayml/stable-diffusion-inpainting")
    ap.add_argument("--lora_weights",      default=str(PROJECT / "weights" / "sd_lora_davis" / "final"))
    ap.add_argument("--sd_strength",       type=float, default=0.55)
    ap.add_argument("--sd_steps",          type=int,   default=30)
    ap.add_argument("--max_refine_frames", type=int,   default=12)
    ap.add_argument("--skip_sam2_pp",      action="store_true")
    ap.add_argument("--skip_sam3_pp",      action="store_true")
    ap.add_argument("--skip_sd_base",      action="store_true")
    ap.add_argument("--skip_sd_lora",      action="store_true")
    args = ap.parse_args()

    UNIFIED  = PROJECT / "data" / "mask" / "unified"
    RAW      = PROJECT / "data" / "raw"
    OUT_ROOT = PROJECT / "data" / "outputs" / "part3_new"

    print(f"\n{'#'*64}")
    print(f"  Unified Inpainting Pipeline")
    print(f"  Datasets: {args.datasets}")
    print(f"  GPU PP: {args.gpu_pp}   GPU SD: {args.gpu_sd}")
    print(f"{'#'*64}")

    for ds in args.datasets:
        raw_dir   = str(RAW / ds)
        sam2_mask = str(UNIFIED / "sam2" / ds)
        sam3_mask = str(UNIFIED / "sam3" / ds)

        if not Path(raw_dir).is_dir():
            print(f"\n[!] Raw frames not found: {raw_dir}, skipping {ds}.")
            continue

        print(f"\n{'='*64}")
        print(f"  Dataset: {ds}")
        print(f"{'='*64}")

        # A1: SAM2 -> ProPainter
        sam2_pp_out = OUT_ROOT / "sam2_propainter" / ds
        if not args.skip_sam2_pp:
            if not Path(sam2_mask).is_dir() or not os.listdir(sam2_mask):
                print(f"\n[!] SAM2 masks missing: {sam2_mask}")
            else:
                print(f"\n  [A1] SAM2 -> ProPainter")
                run_propainter(raw_dir, sam2_mask, str(sam2_pp_out), gpu=args.gpu_pp)
        else:
            print(f"  [skip A1] SAM2->PP")

        # A2: SAM3 -> ProPainter
        sam3_pp_out = OUT_ROOT / "sam3_propainter" / ds
        if not args.skip_sam3_pp:
            if not Path(sam3_mask).is_dir() or not os.listdir(sam3_mask):
                print(f"\n[!] SAM3 masks missing: {sam3_mask}")
            else:
                print(f"\n  [A2] SAM3 -> ProPainter")
                run_propainter(raw_dir, sam3_mask, str(sam3_pp_out), gpu=args.gpu_pp)
        else:
            print(f"  [skip A2] SAM3->PP")

        # B1: SAM3 PP -> SD base (pre-finetune)
        if not args.skip_sd_base:
            print(f"\n  [B1] SAM3 PP -> SD base (pre-finetune comparison)")
            run_sd_on_pp_output(
                dataset=ds, pp_out_dir=sam3_pp_out, mask_dir=sam3_mask,
                output_dir=OUT_ROOT / "sam3_pp_sd_base" / ds,
                gpu=args.gpu_sd, sd_model=args.sd_model, lora_weights=None,
                sd_strength=args.sd_strength, sd_steps=args.sd_steps,
                max_refine_frames=args.max_refine_frames,
            )
        else:
            print(f"  [skip B1] SD-base")

        # B2: SAM3 PP -> SD LoRA (post-finetune)
        if not args.skip_sd_lora:
            lw = args.lora_weights
            if not Path(lw).exists():
                print(f"\n[!] LoRA weights not found: {lw}")
            else:
                print(f"\n  [B2] SAM3 PP -> SD LoRA (post-finetune)")
                run_sd_on_pp_output(
                    dataset=ds, pp_out_dir=sam3_pp_out, mask_dir=sam3_mask,
                    output_dir=OUT_ROOT / "sam3_pp_sd_lora" / ds,
                    gpu=args.gpu_sd, sd_model=args.sd_model, lora_weights=lw,
                    sd_strength=args.sd_strength, sd_steps=args.sd_steps,
                    max_refine_frames=args.max_refine_frames,
                )
        else:
            print(f"  [skip B2] SD-LoRA")

    print(f"\n{'#'*64}")
    print(f"  Pipeline complete.  Results -> {OUT_ROOT}")
    print(f"{'#'*64}\n")


if __name__ == "__main__":
    main()
