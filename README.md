# Video Object Removal & Inpainting

**AIAA 3201 – Introduction to Computer Vision, Spring 2026, Project 3**

Automated pipeline for removing dynamic objects from videos and filling the background using three progressive approaches.

---

## Project Overview

| Part | Method | Description |
|------|--------|-------------|
| **Part 1** | Baseline (Hand-crafted) | YOLOv8 segmentation + optical flow dynamic judgment + `cv2.inpaint` |
| **Part 2** | SOTA | SAM2 video tracking + ProPainter temporal inpainting |
| **Part 3** | Optimized | SAM3 mask refinement + ProPainter + Stable Diffusion with LoRA fine-tuning |

### Key Results

| Dataset | Pipeline | JM (IoU↑) | PSNR (dB↑) | SSIM↑ |
|---------|----------|-----------|-----------|-------|
| bmx-trees | Part 2 (SAM2→PP) | 0.6474 | 11.22 | 0.9592 |
| bmx-trees | Part 3 A2 (SAM3→PP) | 0.5240 | **12.42** | 0.9588 |
| tennis | Part 2 (SAM2→PP) | 0.7369 | 11.97 | 0.9131 |
| tennis | Part 3 B2 (SAM3+PP→LoRA) | 0.7134 | **12.32** | 0.8756 |

---

## Repository Structure

```
cvproject_submission/
├── part1_baseline.py               # Part 1: hand-crafted pipeline entry point
├── part2_generate_sam2_masks.py    # Part 2: SAM2 video mask generation
├── part3_generate_sam3_masks.py    # Part 3: SAM3 per-frame mask refinement
├── part3_inpainting_pipeline.py    # Part 3: full inpainting pipeline (PP + SD + LoRA)
├── part3_finetune_sd_lora.py       # Part 3: LoRA fine-tuning on DAVIS dataset
├── eval_mask_iou.py                # Mask quality evaluation (J&F metrics)
├── evaluate_metrics.py             # Video quality evaluation (PSNR/SSIM)
├── requirements.txt
├── README.md
└── modules/
    ├── part1/baseline.py           # YOLOv8 segmentor + optical flow classifier
    ├── part2/tracker.py            # SAM2 video tracking wrapper
    ├── part2/preprocessor.py       # Mask dilation & morphological refinement
    ├── part2/inpainter.py          # ProPainter inference wrapper
    ├── part3/sam3_mask_refiner.py  # SAM3 per-frame mask refiner
    ├── part3/generative_fill.py    # Stable Diffusion inpainting wrapper
    ├── ProPainter/                 # ProPainter source (Wan et al., ICCV 2023)
    └── sam2/                       # SAM2 source (Meta Research)
```

---

## Environment Setup

**Requirements**: Python 3.10, CUDA 12.1, 2× NVIDIA GPU (≥16 GB VRAM each recommended)

```bash
# 1. Create conda environment
conda create -n cvproject python=3.10 -y
conda activate cvproject

# 2. Install PyTorch (CUDA 12.1)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. Install SAM2
pip install -e modules/sam2 --no-build-isolation

# 4. Install remaining dependencies
pip install -r requirements.txt
```

---

## Model Weights

Download and place weights as follows:

### SAM2 (Meta Research)
```bash
# https://github.com/facebookresearch/sam2#model-checkpoints
mkdir -p weights/part2/sam2
# Place: weights/part2/sam2/sam2.1_hiera_large.pt
```

### ProPainter (Wan et al.)
```bash
# https://github.com/sczhou/ProPainter/releases
mkdir -p weights/ProPainter
# Place: weights/ProPainter/ProPainter.pth
#        weights/ProPainter/raft-things.pth
#        weights/ProPainter/recurrent_flow_completion.pth
```

### Stable Diffusion Inpainting (Runway ML)
```bash
huggingface-cli download runwayml/stable-diffusion-inpainting
```

### LoRA Fine-tuned Weights (Part 3 only)
After running fine-tuning, weights are saved to `weights/sd_lora_davis/final/`.

### YOLOv8 (Part 1 only)
Downloaded automatically by `ultralytics` on first run.

---

## Usage

### Part 1 – Baseline (Hand-crafted)

```bash
python part1_baseline.py \
    --video data/raw/tennis.mp4 \
    --points "[[350,400]]" \
    --labels "[1]"
```

### Part 2 – SAM2 + ProPainter

```bash
# Step 1: Generate SAM2 masks
PYTHONPATH=modules/sam2 python part2_generate_sam2_masks.py \
    --datasets tennis bmx-trees wild_video

# Step 2: Run ProPainter inpainting
python part3_inpainting_pipeline.py \
    --datasets tennis bmx-trees wild_video \
    --pipelines sam2_propainter \
    --gpu_pp cuda:0
```

### Part 3 – SAM3 + ProPainter + SD + LoRA

```bash
# Step 1: Generate SAM2 coarse masks (same as Part 2 Step 1)

# Step 2: Refine masks with SAM3
PYTHONPATH=modules/sam2 python part3_generate_sam3_masks.py \
    --datasets tennis bmx-trees wild_video

# Step 3 (optional): Fine-tune SD with LoRA
python part3_finetune_sd_lora.py \
    --data_dir data/raw \
    --output_dir weights/sd_lora_davis

# Step 4: Run all four pipelines
python part3_inpainting_pipeline.py \
    --datasets tennis bmx-trees wild_video \
    --pipelines sam2_propainter sam3_propainter sam3_pp_sd_base sam3_pp_sd_lora \
    --gpu_pp cuda:0 \
    --gpu_sd cuda:1
```

Pipelines:
- **A1** `sam2_propainter` – SAM2 masks → ProPainter
- **A2** `sam3_propainter` – SAM3 refined masks → ProPainter
- **B1** `sam3_pp_sd_base` – SAM3 + ProPainter → SD base inpainting
- **B2** `sam3_pp_sd_lora` – SAM3 + ProPainter → SD with DAVIS LoRA

Output: `data/outputs/part3_new/{pipeline}/{dataset}/inpaint_out.mp4`

### Evaluation

```bash
# Mask quality (J & F metrics)
python eval_mask_iou.py --pred_dir data/mask/unified/sam2 --gt_dir data/Annotations

# Video quality (PSNR / SSIM)
python evaluate_metrics.py --output_root data/outputs/part3_new --gt_root data/raw
```

---

## Notes

- **PYTHONPATH**: SAM2 requires `modules/sam2` on the Python path.
- **Proxy / Offline**: Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` if HuggingFace is blocked after downloading weights.
- **Tennis shadow fix**: `part2_generate_sam2_masks.py` includes a geometric post-processing step (`tennis_shadow_fill`) that extends the mask to cover the player's shadow, raising tennis IoU from 0.584 → 0.737.
- **SAM2 non-determinism**: SAM2 mask generation is non-deterministic due to CUDA random ops. Use saved masks from `data/mask/unified/sam2/tennis_v5_backup/` for reproducibility.
