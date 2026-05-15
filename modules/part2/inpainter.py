"""
ProPainter Inpainting Module
Wraps the ProPainter inference script for video inpainting.
Supports generative keyframe blending for improved results.
"""

import os
import sys
import subprocess
import shutil
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROPAINTER_DIR = PROJECT_ROOT / "modules" / "ProPainter"


class ProPainterEngine:
    """Wrapper around ProPainter inference with generative keyframe blending."""

    def __init__(self, device: str = "cuda:1"):
        self.device = device
        self.propainter_dir = PROPAINTER_DIR
        self.weights_dir = PROJECT_ROOT / "weights" / "part2" / "propainter"

        # Verify weights exist
        required = ["ProPainter.pth", "recurrent_flow_completion.pth", "raft-things.pth"]
        for w in required:
            if not (self.weights_dir / w).exists():
                raise FileNotFoundError(f"Missing weight: {self.weights_dir / w}")

        # Symlink weights into ProPainter/weights/ so inference script can find them
        pp_weights = self.propainter_dir / "weights"
        pp_weights.mkdir(exist_ok=True)
        for w in required:
            dst = pp_weights / w
            src = self.weights_dir / w
            if dst.is_symlink():
                dst.unlink()  # remove stale symlink
            if not dst.exists():
                os.symlink(str(src), str(dst))

    def _blend_generative_keyframes(
        self, video_path: str, mask_dir: str, gen_keyframe_dir: str, blended_dir: str
    ) -> str:
        """
        Blend SDXL-inpainted keyframes into the input video frames so that
        ProPainter can use them as clean reference frames.

        For keyframes with generative fills, we composite the generated content
        into the masked region of the original frame. This gives ProPainter
        much better reference data for temporal propagation.
        """
        os.makedirs(blended_dir, exist_ok=True)

        # Get list of generative keyframes
        gen_files = {}
        if os.path.isdir(gen_keyframe_dir):
            for f in os.listdir(gen_keyframe_dir):
                if f.endswith(".png"):
                    frame_idx = int(Path(f).stem)
                    gen_files[frame_idx] = os.path.join(gen_keyframe_dir, f)

        if not gen_files:
            return video_path  # No keyframes to blend

        print(f"[Inpainter] Blending {len(gen_files)} generative keyframes into source")

        # If video_path is a directory (frame dir), copy frames and overlay
        if os.path.isdir(video_path):
            frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))])
            for fname in frame_files:
                frame_idx = int(Path(fname).stem)
                src_frame = cv2.imread(os.path.join(video_path, fname))

                if frame_idx in gen_files:
                    # Load mask and generative result
                    mask_path = os.path.join(mask_dir, f"{frame_idx:05d}.png")
                    if os.path.exists(mask_path):
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        gen_frame = cv2.imread(gen_files[frame_idx])
                        if gen_frame is not None and mask is not None:
                            # Resize gen_frame to match src if needed
                            if gen_frame.shape[:2] != src_frame.shape[:2]:
                                gen_frame = cv2.resize(gen_frame, (src_frame.shape[1], src_frame.shape[0]))
                            if mask.shape[:2] != src_frame.shape[:2]:
                                mask = cv2.resize(mask, (src_frame.shape[1], src_frame.shape[0]))
                            # Feathered blending at mask boundary (0.7 weight for SD to avoid over-reliance)
                            mask_f = cv2.GaussianBlur(mask, (21, 21), 0).astype(np.float32) / 255.0
                            blend_w = 0.7  # 70% SD content, 30% original (helps ProPainter with complex textures)
                            mask_3c = np.stack([mask_f * blend_w] * 3, axis=-1)
                            src_frame = (gen_frame * mask_3c + src_frame * (1 - mask_3c)).astype(np.uint8)

                cv2.imwrite(os.path.join(blended_dir, fname), src_frame)
            return blended_dir
        else:
            # Video file: extract frames, blend, and return blended frame dir
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return video_path
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx in gen_files:
                    mask_path = os.path.join(mask_dir, f"{idx:05d}.png")
                    if os.path.exists(mask_path):
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        gen_frame = cv2.imread(gen_files[idx])
                        if gen_frame is not None and mask is not None:
                            if gen_frame.shape[:2] != frame.shape[:2]:
                                gen_frame = cv2.resize(gen_frame, (frame.shape[1], frame.shape[0]))
                            if mask.shape[:2] != frame.shape[:2]:
                                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                            mask_f = cv2.GaussianBlur(mask, (21, 21), 0).astype(np.float32) / 255.0
                            mask_3c = np.stack([mask_f] * 3, axis=-1)
                            frame = (gen_frame * mask_3c + frame * (1 - mask_3c)).astype(np.uint8)
                cv2.imwrite(os.path.join(blended_dir, f"{idx:05d}.jpg"), frame)
                idx += 1
            cap.release()
            return blended_dir

    def restore(
        self,
        video: str,
        mask: str,
        output: str = None,
        neighbor_length: int = 10,
        subvideo_length: int = 80,
        ref_stride: int = 10,
        mask_dilation: int = 8,
        save_fps: int = 24,
        fp16: bool = True,
        gen_keyframe_dir: str = None,
    ) -> str:
        """
        Run ProPainter video inpainting.

        Args:
            video: Path to input video or frame directory.
            mask: Path to mask directory (binary PNGs).
            output: Output directory.
            neighbor_length: Local neighboring frames length.
            subvideo_length: Sub-video length for long video inference.
            ref_stride: Stride of global reference frames.
            mask_dilation: Dilation for flow masking inside ProPainter.
            save_fps: Output video FPS.
            fp16: Use half precision (A40 has excellent fp16 performance).
            gen_keyframe_dir: Directory with SDXL-inpainted keyframe images.
                              If provided, these are blended into source frames
                              before ProPainter runs, giving it clean reference data.

        Returns:
            Output directory path.
        """
        if output is None:
            output = str(PROJECT_ROOT / "data" / "outputs" / "part2")
        os.makedirs(output, exist_ok=True)

        # Resolve paths to absolute (subprocess runs in ProPainter cwd)
        video = str(Path(video).resolve())
        mask = str(Path(mask).resolve())
        output = str(Path(output).resolve())

        # Blend generative keyframes if available
        actual_video = video
        blended_dir = None
        if gen_keyframe_dir and os.path.isdir(gen_keyframe_dir) and os.listdir(gen_keyframe_dir):
            video_name = Path(video).stem
            blended_dir = str(PROJECT_ROOT / "data" / "temp_blended" / video_name)
            actual_video = self._blend_generative_keyframes(video, mask, gen_keyframe_dir, blended_dir)
            print(f"[Inpainter] Using blended frames from {actual_video}")

        cmd = [
            sys.executable,
            str(self.propainter_dir / "inference_propainter.py"),
            "--video", str(actual_video),
            "--mask", str(mask),
            "--output", str(output),
            "--neighbor_length", str(neighbor_length),
            "--subvideo_length", str(subvideo_length),
            "--ref_stride", str(ref_stride),
            "--mask_dilation", str(mask_dilation),
            "--save_fps", str(save_fps),
        ]
        if fp16:
            cmd.append("--fp16")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.device.replace("cuda:", "")

        print(f"[Inpainter] Running ProPainter on {self.device}")
        print(f"[Inpainter] Config: neighbor_length={neighbor_length}, subvideo_length={subvideo_length}, "
              f"ref_stride={ref_stride}, mask_dilation={mask_dilation}, fp16={fp16}")
        print(f"[Inpainter] Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=str(self.propainter_dir),
            env=env,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"[Inpainter] STDERR:\n{result.stderr}")
            raise RuntimeError(f"ProPainter failed with code {result.returncode}")

        print(f"[Inpainter] STDOUT:\n{result.stdout}")

        # Cleanup blended temp dir
        if blended_dir and os.path.isdir(blended_dir):
            shutil.rmtree(blended_dir, ignore_errors=True)

        print(f"[Inpainter] Done. Output saved to {output}")
        return output
