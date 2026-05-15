#!/usr/bin/env python3
"""
LoRA Fine-tuning of SD 1.5 Inpainting on DAVIS Dataset
=======================================================
Trains a lightweight LoRA adapter on the UNet of the SD 1.5 inpainting model
using DAVIS video frames and their object segmentation masks.

Usage:
  python finetune_sd_lora.py \
      --davis_root data/davis/DAVIS \
      --output_dir weights/sd_lora_davis \
      --max_steps 1500 \
      --batch_size 2 \
      --gpu cuda:3
"""

import os, sys, math, random, argparse
import numpy as np
import cv2
from pathlib import Path

# path fix: ensure conda env torch 2.5.1 is used, not user-local 2.4.1
_conda_sp = "/public/software/anaconda3/envs/cvproject/lib/python3.10/site-packages"
if _conda_sp in sys.path:
    sys.path.remove(_conda_sp)
sys.path.insert(0, _conda_sp)

# os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
# os.environ.setdefault("HTTP_PROXY",  "http://127.0.0.1:7890")

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import DDPMScheduler, AutoencoderKL
from diffusers.models import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model

PROJECT = Path(__file__).resolve().parent
TRAIN_RESOLUTION = 512

# ── Dataset ──────────────────────────────────────────────────────────────────

class DAVISInpaintDataset(Dataset):
    """
    Training pairs from DAVIS: (masked_frame, mask, original_frame).
    The model learns to inpaint the foreground-object region (background reconstruction).
    """
    def __init__(self, davis_root, split="train", resolution=TRAIN_RESOLUTION,
                 dilation_range=(0, 20), augment=True, max_frames_per_seq=8,
                 excluded_seqs=None):
        self.resolution = resolution
        self.dilation_range = dilation_range
        self.augment = augment
        davis_root = Path(davis_root)

        img_dir  = davis_root / "JPEGImages" / "480p"
        ann_dir  = davis_root / "Annotations" / "480p"
        set_file = davis_root / "ImageSets" / "2017" / f"{split}.txt"

        if set_file.exists():
            seqs = [l.strip() for l in set_file.read_text().splitlines() if l.strip()]
        else:
            seqs = sorted(d.name for d in img_dir.iterdir() if d.is_dir())

        if excluded_seqs:
            seqs = [s for s in seqs if s not in excluded_seqs]

        self.pairs = []
        for seq in seqs:
            frames = sorted((img_dir / seq).glob("*.jpg"))
            stride = max(1, len(frames) // max_frames_per_seq)
            for frame in frames[::stride][:max_frames_per_seq]:
                ann = ann_dir / seq / frame.with_suffix(".png").name
                if ann.exists():
                    self.pairs.append((str(frame), str(ann)))

        print(f"[DAVIS] {split}: {len(self.pairs)} pairs from {len(seqs)} seqs")

        self.img_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, ann_path = self.pairs[idx]
        img_bgr = cv2.imread(img_path)
        ann_raw = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)

        if img_bgr is None or ann_raw is None:
            d = np.zeros((self.resolution, self.resolution, 3), np.uint8)
            m = np.zeros((self.resolution, self.resolution), np.uint8)
            return self._make_tensors(d, m)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask    = (ann_raw > 0).astype(np.uint8)

        img_rgb = cv2.resize(img_rgb, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
        mask    = cv2.resize(mask, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if random.random() < 0.5:
                img_rgb = cv2.flip(img_rgb, 1)
                mask    = cv2.flip(mask, 1)
            if random.random() < 0.4:
                a = random.uniform(0.8, 1.2); b = random.randint(-15, 15)
                img_rgb = np.clip(img_rgb.astype(np.float32) * a + b, 0, 255).astype(np.uint8)

        dil = random.randint(*self.dilation_range)
        if dil > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil*2+1, dil*2+1))
            mask = cv2.dilate(mask, k)

        return self._make_tensors(img_rgb, mask)

    def _make_tensors(self, img_rgb, mask):
        masked = img_rgb.copy()
        masked[mask > 0] = 0
        return {
            "original":     self.img_tf(img_rgb),
            "masked_image": self.img_tf(masked),
            "mask":         torch.from_numpy(mask).unsqueeze(0).float(),
            "prompt":       "",
        }


# ── Training ─────────────────────────────────────────────────────────────────

def train(args):
    device = args.gpu
    out    = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[LoRA] device={device} steps={args.max_steps} bs={args.batch_size} rank={args.lora_rank}")

    model_id = args.sd_model
    print(f"[LoRA] Loading '{model_id}' ...")
    tokenizer    = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16", local_files_only=True)
    vae          = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16, variant="fp16", local_files_only=True)
    unet         = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16, variant="fp16", local_files_only=True)
    noise_sched  = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler", local_files_only=True)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_q","to_k","to_v","to_out.0","proj_in","proj_out"],
    )
    unet = get_peft_model(unet, lora_cfg)
    unet.print_trainable_parameters()
    # Cast LoRA parameters to float32 for stable gradient computation
    for name, param in unet.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    vae.to(device)  # stays fp16 (frozen)
    text_encoder.to(device)  # stays fp16 (frozen)
    unet.to(device, dtype=torch.float32)  # LoRA params are fp32

    dataset = DAVISInpaintDataset(
        davis_root=args.davis_root, split="train",
        augment=True, max_frames_per_seq=args.max_frames_per_seq,
        excluded_seqs=["bmx-trees", "tennis"],
    )
    if len(dataset) == 0:
        print("[LoRA] ERROR: empty dataset"); return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

    def cycle(dl):
        while True:
            for b in dl: yield b
    inf_loader = cycle(loader)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate,
                                   betas=(0.9,0.999), weight_decay=1e-2, eps=1e-8)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=1e-6)

    unet.train()
    losses = []
    print(f"\n[LoRA] Training {args.max_steps} steps ...")

    for step in range(1, args.max_steps + 1):
        batch = next(inf_loader)
        originals     = batch["original"].to(device)
        masked_images = batch["masked_image"].to(device)
        masks         = batch["mask"].to(device)
        prompts       = batch["prompt"]

        with torch.no_grad():
            tgt_lat  = vae.encode(originals.to(torch.float16)).latent_dist.sample().to(torch.float32) * vae.config.scaling_factor
            msk_lat  = vae.encode(masked_images.to(torch.float16)).latent_dist.sample().to(torch.float32) * vae.config.scaling_factor

        h_lat, w_lat = tgt_lat.shape[-2:]
        mask_lat = F.interpolate(masks, size=(h_lat, w_lat), mode="nearest")

        noise     = torch.randn_like(tgt_lat)
        bsz       = tgt_lat.shape[0]
        timesteps = torch.randint(0, noise_sched.config.num_train_timesteps, (bsz,), device=device).long()
        noisy     = noise_sched.add_noise(tgt_lat, noise, timesteps)

        unet_in = torch.cat([noisy, mask_lat, msk_lat], dim=1)

        tok = tokenizer(prompts, padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")
        with torch.no_grad():
            emb = text_encoder(tok.input_ids.to(device))[0].to(torch.float32)

        noise_pred = unet(unet_in, timesteps, encoder_hidden_states=emb).sample

        loss_full   = F.mse_loss(noise_pred, noise, reduction="none")
        mask_weight = 1.0 + mask_lat * 2.0
        loss = (loss_full * mask_weight).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()
        sched.step()

        losses.append(loss.item())
        if step % args.log_interval == 0:
            avg = sum(losses[-args.log_interval:]) / args.log_interval
            print(f"  step {step:5d}/{args.max_steps}  loss={avg:.4f}  lr={sched.get_last_lr()[0]:.2e}")

        if step % args.save_interval == 0 or step == args.max_steps:
            ckpt = out / f"checkpoint-{step}"
            unet.save_pretrained(str(ckpt))
            print(f"  saved → {ckpt}")

    final = out / "final"
    unet.save_pretrained(str(final))
    import json
    with open(out / "loss_log.json", "w") as f:
        json.dump({"losses": losses}, f)
    print(f"\n[LoRA] Done. Adapter → {final}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--davis_root",  default="data/davis/DAVIS")
    ap.add_argument("--output_dir",  default="weights/sd_lora_davis")
    ap.add_argument("--sd_model",    default="runwayml/stable-diffusion-inpainting")
    ap.add_argument("--gpu",         default="cuda:3")
    ap.add_argument("--lora_rank",   type=int,   default=8)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--batch_size",  type=int,   default=2)
    ap.add_argument("--max_steps",   type=int,   default=1500)
    ap.add_argument("--max_frames_per_seq", type=int, default=8)
    ap.add_argument("--log_interval",  type=int, default=50)
    ap.add_argument("--save_interval", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    train(args)
