"""
Part 3 – Direction A: SAM3 Mask Refinement
============================================
Refines coarse SAM2 video masks using SAM3 (Segment Anything with Concepts).
Falls back to SAM (facebook/sam-vit-base, already cached) if SAM3 weights are
unavailable.

Key improvements over v1:
  - Floor/court/road mask detection: when Part-2 returns a large wrong mask,
    uses point prompts above the mask area to find the actual foreground object.
  - Tracking extension: for datasets where Part-2 tracking stops early,
    continues SAM detection into empty-mask frames using the last known bbox.
  - Conservative union is skipped when coarse was a floor mask and refined
    found a smaller correct object.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ── helpers ──────────────────────────────────────────────────────────────────

def bbox_from_mask(binary_mask: np.ndarray, pad: int = 12):
    """Return [x1, y1, x2, y2] bounding box of non-zero pixels, with padding."""
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return None
    h, w = binary_mask.shape
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(w, int(xs.max()) + pad)
    y2 = min(h, int(ys.max()) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


# ── SAM3MaskRefiner ───────────────────────────────────────────────────────────

class Sam3MaskRefiner:
    """
    Wraps SAM3 (or SAM vit-base fallback) to refine coarse video masks.

    New in v2:
      * Floor/court mask detection via _is_floor_mask() — when coarse mask is
        a large bottom-heavy region, uses point prompts above it to find the
        real foreground object (player, car, cyclist).
      * Point-prompted refinement via _refine_with_points().
    """

    _use_sam3: bool = False   # class-level flag

    def __init__(self, device: str = "cuda:1", model_id: str = "facebook/sam3"):
        self.device = device
        self.model_id = model_id
        self.processor = None
        self.model = None
        self._load()

    # ── model loading ─────────────────────────────────────────────────────────

    def _load(self):
        os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
        os.environ.setdefault("HTTP_PROXY",  "http://127.0.0.1:7890")
        token = self._get_hf_token()

        # Try SAM3 first
        try:
            from transformers import Sam3Model, Sam3Processor
            print(f"[Sam3Refiner] Loading SAM3 '{self.model_id}' …")
            kw = dict(token=token) if token else {}
            self.processor = Sam3Processor.from_pretrained(self.model_id, **kw)
            self.model = Sam3Model.from_pretrained(
                self.model_id, torch_dtype=torch.float16, **kw
            ).to(self.device).eval()
            Sam3MaskRefiner._use_sam3 = True
            print(f"[Sam3Refiner] SAM3 loaded on {self.device}")
            return
        except ImportError:
            print(f"[Sam3Refiner] SAM3 not in local cache (ImportError)")
        except Exception as e:
            print(f"[Sam3Refiner] SAM3 network download failed ({type(e).__name__}: {str(e)[:80]})")

        self._load_sam_fallback()

    def _load_sam_fallback(self):
        from transformers import SamModel, SamProcessor
        fb = "facebook/sam-vit-base"
        print(f"[Sam3Refiner] Falling back to {fb} (local cache) ...")
        self.processor = SamProcessor.from_pretrained(fb, local_files_only=True)
        self.model = SamModel.from_pretrained(fb, local_files_only=True).to(self.device).eval()
        Sam3MaskRefiner._use_sam3 = False
        print(f"[Sam3Refiner] SAM (vit-base) loaded from cache on {self.device}")

    @staticmethod
    def _get_hf_token() -> str:
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if token:
            return token
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            t = token_path.read_text().strip()
            if t:
                return t
        return ""

    # ── floor-mask detection ─────────────────────────────────────────────────

    def _is_floor_mask(self, coarse_mask: np.ndarray) -> bool:
        """
        Return True if the mask looks like a large floor/court/road region
        rather than the actual foreground object.

        Criteria: area > 12%, centroid-y > 58% from top, spans > 70% of width.
        """
        area_ratio = (coarse_mask > 0).sum() / coarse_mask.size
        if area_ratio < 0.12:
            return False
        ys, xs = np.where(coarse_mask > 0)
        if len(ys) == 0:
            return False
        h, w = coarse_mask.shape
        cy = ys.mean() / h
        x_coverage = (xs.max() - xs.min()) / w
        return area_ratio > 0.12 and cy > 0.58 and x_coverage > 0.70

    # ── point-prompted refinement ─────────────────────────────────────────────

    @torch.no_grad()
    def _refine_with_points(self, image, h: int, w: int, probe_ys: list):
        """
        Try SAM with point prompts at given y/x positions;
        return the best small-object mask found.
        """
        best_mask = None
        best_score = 0.0
        for y in probe_ys:
            for x in [w // 4, w // 2, 3 * w // 4]:
                if not (0 < y < h) or not (0 < x < w):
                    continue
                try:
                    inputs = self.processor(
                        images=image,
                        input_points=[[[x, y]]],
                        input_labels=[[1]],
                        return_tensors="pt",
                    ).to(self.device)
                    outputs = self.model(**inputs)
                    m = self._extract_best_mask(outputs, inputs)
                    if m is None:
                        continue
                    area_ratio = (m > 0).sum() / m.size
                    if 0.005 < area_ratio < 0.35 and area_ratio > best_score:
                        best_mask = m
                        best_score = area_ratio
                except Exception:
                    pass
        return best_mask

    # ── inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def refine(
        self,
        frame_bgr: np.ndarray,
        coarse_mask: np.ndarray,
        concept: str = "dynamic object",
    ) -> np.ndarray:
        """
        Refine one frame's mask.

        Args:
            frame_bgr:   uint8 (H, W, 3) BGR frame from OpenCV.
            coarse_mask: uint8 (H, W) binary mask (0 or 255) from Part 2.
            concept:     text concept for SAM3 prompting.

        Returns:
            uint8 (H, W) refined binary mask (0 or 255).
        """
        if coarse_mask is None or coarse_mask.sum() == 0:
            return coarse_mask if coarse_mask is not None else np.zeros(
                frame_bgr.shape[:2], np.uint8
            )

        bbox = bbox_from_mask(coarse_mask, pad=12)
        if bbox is None:
            return coarse_mask

        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        h_img, w_img = coarse_mask.shape

        try:
            if Sam3MaskRefiner._use_sam3:
                refined = self._infer_sam3(image, bbox, concept)
            else:
                # Check if coarse mask is a large floor/court region
                if self._is_floor_mask(coarse_mask):
                    ys_c = np.where(coarse_mask > 0)[0]
                    top_y = int(ys_c.min()) if len(ys_c) > 0 else h_img // 2
                    # Probe ABOVE and INSIDE the top of the floor region
                    # (player body may be partially above or at the top of the mask)
                    fracs = [-0.28, -0.18, -0.10, -0.04, 0.03, 0.10, 0.18, 0.28]
                    probe_ys = sorted({max(3, min(h_img - 3, int(top_y + f * h_img)))
                                       for f in fracs})
                    player_mask = self._refine_with_points(image, h_img, w_img, probe_ys)
                    if player_mask is not None and player_mask.sum() > 0:
                        return player_mask
                    # Player not detectable — return empty to skip floor inpainting
                    return np.zeros_like(coarse_mask)
                refined = self._infer_sam1(image, bbox)

            if refined is not None and refined.sum() > 0:
                # Conservative union: never shrink the Part-2 mask
                return ((refined > 0) | (coarse_mask > 0)).astype(np.uint8) * 255
        except Exception as e:
            print(f"[Sam3Refiner] Inference error: {e} – keeping coarse mask")

        return coarse_mask

    def _infer_sam3(self, image, bbox: list, concept: str) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        base_kw = dict(images=image, input_boxes=[[[x1, y1, x2, y2]]], return_tensors="pt")
        try:
            inputs = self.processor(text=concept, **base_kw).to(self.device)
        except TypeError:
            inputs = self.processor(**base_kw).to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = self.model(**inputs)
        return self._extract_best_mask(outputs, inputs)

    def _infer_sam1(self, image, bbox: list) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        inputs = self.processor(
            images=image,
            input_boxes=[[[x1, y1, x2, y2]]],
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)
        return self._extract_best_mask(outputs, inputs)

    def _extract_best_mask(self, outputs, inputs) -> np.ndarray:
        """Post-process SAM/SAM3 outputs into the best binary mask."""
        try:
            masks = self.processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )
            pred = masks[0]
        except Exception:
            pred = outputs.pred_masks[0].float().cpu()

        if pred.ndim == 4:
            pred = pred.squeeze(0)

        if hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
            scores = outputs.iou_scores.cpu().squeeze()
            idx = int(scores.argmax()) if scores.ndim > 0 else 0
        else:
            idx = 0

        m = pred[idx].float().numpy() if pred.ndim == 3 else pred.float().numpy()
        return (m > 0.5).astype(np.uint8) * 255

    # ── cleanup ───────────────────────────────────────────────────────────────

    def free(self):
        del self.model, self.processor
        self.model = self.processor = None
        torch.cuda.empty_cache()
        print("[Sam3Refiner] GPU memory freed")


# ── dataset-level helper ──────────────────────────────────────────────────────

def refine_dataset_masks(
    raw_dir: str,
    mask_dir: str,
    output_dir: str,
    device: str = "cuda:1",
    model_id: str = "facebook/sam3",
    concept: str = "dynamic object",
    stride: int = 3,
    extend_tracking: bool = False,
) -> None:
    """
    Refine all masks in `mask_dir` using SAM and save to `output_dir`.

    Args:
        raw_dir:          Directory of original video frames (jpg/png).
        mask_dir:         Directory of Part-2 binary masks (png).
        output_dir:       Where to save refined masks.
        device:           CUDA device string.
        model_id:         HuggingFace model id for SAM3.
        concept:          Text concept for SAM3 prompting.
        stride:           Run SAM on every `stride`-th frame.
        extend_tracking:  When True, continue tracking into empty-mask segments
                          using the last known bounding box (helps datasets where
                          Part-2 tracking stops mid-video).
    """
    os.makedirs(output_dir, exist_ok=True)

    frame_files = sorted(
        f for f in os.listdir(raw_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    mask_files = sorted(f for f in os.listdir(mask_dir) if f.endswith(".png"))

    n = min(len(frame_files), len(mask_files))
    if n == 0:
        print(f"[Sam3Refiner] No frames/masks found in {raw_dir} / {mask_dir}")
        return

    print(f"[Sam3Refiner] Refining {n} masks  concept='{concept}'  stride={stride}")
    refiner = Sam3MaskRefiner(device=device, model_id=model_id)

    refined: dict = {}

    keyframe_idxs = list(range(0, n, stride))
    for i in keyframe_idxs:
        frame = cv2.imread(os.path.join(raw_dir, frame_files[i]))
        mask  = cv2.imread(os.path.join(mask_dir,  mask_files[i]), cv2.IMREAD_GRAYSCALE)
        if frame is None or mask is None:
            continue
        refined[i] = refiner.refine(frame, mask, concept=concept)
        if i % 30 == 0:
            print(f"  ... frame {i}/{n}")

    sorted_kf = sorted(refined)

    # ── Extend tracking into empty-mask zones ────────────────────────────────
    if extend_tracking:
        valid_kf = sorted(k for k in refined if refined[k].sum() > 0)
        if valid_kf:
            last_valid = valid_kf[-1]
            last_bbox = bbox_from_mask(refined[last_valid], pad=40)
            if last_bbox is not None and last_valid < n - 1:
                print(f"[Sam3Refiner] Extending tracking from frame {last_valid} ...")
                ext_stride = max(stride, 3)
                for i in range(last_valid + ext_stride, n, ext_stride):
                    frm = cv2.imread(os.path.join(raw_dir, frame_files[i]))
                    if frm is None:
                        break
                    img_ext = Image.fromarray(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
                    x1, y1, x2, y2 = last_bbox
                    try:
                        inp = refiner.processor(
                            images=img_ext,
                            input_boxes=[[[x1, y1, x2, y2]]],
                            return_tensors="pt",
                        ).to(refiner.device)
                        outp = refiner.model(**inp)
                        m_ext = refiner._extract_best_mask(outp, inp)
                    except Exception:
                        break
                    if m_ext is None or m_ext.sum() == 0:
                        break
                    if (m_ext > 0).sum() / m_ext.size < 0.003:
                        break  # object likely gone from frame
                    refined[i] = m_ext
                    new_bbox = bbox_from_mask(m_ext, pad=40)
                    if new_bbox:
                        last_bbox = new_bbox
                    if i % 30 == 0:
                        print(f"  ... extended to frame {i}/{n}")
                last_ext = max((k for k in refined if refined[k].sum() > 0), default=last_valid)
                if last_ext > last_valid:
                    print(f"[Sam3Refiner] Tracking extended to frame {last_ext}")

    # ── Save all frames ───────────────────────────────────────────────────────
    for i in range(n):
        coarse_path = os.path.join(mask_dir, mask_files[i])
        coarse = cv2.imread(coarse_path, cv2.IMREAD_GRAYSCALE)
        if coarse is None:
            coarse = np.zeros((480, 854), np.uint8)

        if i in refined:
            out = refined[i]
        else:
            # Linear interpolation between nearest refined keyframes
            prev_kf = max((k for k in sorted_kf if k <= i), default=None)
            next_kf = min((k for k in sorted_kf if k >  i), default=None)
            if prev_kf is not None and next_kf is not None and prev_kf != next_kf:
                alpha = (i - prev_kf) / (next_kf - prev_kf)
                blended = (1 - alpha) * refined[prev_kf].astype(float) + \
                           alpha      * refined[next_kf].astype(float)
                out = (blended > 127).astype(np.uint8) * 255
            elif prev_kf is not None:
                out = refined[prev_kf]
            elif next_kf is not None:
                out = refined[next_kf]
            else:
                out = coarse

        # Conservative union — skip if coarse was a floor mask and refined
        # found a smaller correct object (to avoid adding back the wrong mask).
        coarse_area = (coarse > 127).sum() / coarse.size if coarse.size > 0 else 0.0
        _cys = np.where(coarse > 127)
        coarse_cy = (_cys[0].mean() / coarse.shape[0]) if len(_cys[0]) > 0 else 0.5
        coarse_xc = ((_cys[1].max() - _cys[1].min()) / coarse.shape[1]) if len(_cys[1]) > 0 else 0.0
        coarse_is_floor = coarse_area > 0.12 and coarse_cy > 0.58 and coarse_xc > 0.70
        out_area = (out > 0).sum() / out.size if out.size > 0 else 0.0
        if coarse_is_floor:
            pass  # Never union with wrong floor/court/road mask
        else:
            out = ((out > 0) | (coarse > 0)).astype(np.uint8) * 255

        cv2.imwrite(os.path.join(output_dir, mask_files[i]), out)

    refiner.free()
    print(f"[Sam3Refiner] Done -> {output_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir",  required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--concept",  default="dynamic object")
    ap.add_argument("--device",   default="cuda:1")
    ap.add_argument("--model_id", default="facebook/sam3")
    ap.add_argument("--stride",   type=int, default=3)
    ap.add_argument("--extend_tracking", action="store_true")
    args = ap.parse_args()
    refine_dataset_masks(args.raw_dir, args.mask_dir, args.out_dir,
                         args.device, args.model_id, args.concept, args.stride,
                         args.extend_tracking)
