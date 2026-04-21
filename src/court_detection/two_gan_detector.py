"""two_gan_detector.py — High-level wrapper for the two-GAN field detection model.

Wraps the vendored pytorch-two-GAN generators (Chen & Little 2019) with a
clean API for use in the rest of the pipeline. Handles:
    - Loading the two generators (field segmentation + line detection)
    - Preprocessing frames (BGR → RGB, resize to 256×256, normalise to [-1, 1])
    - Running inference
    - Postprocessing (denormalise, resize back to frame resolution)

Usage:
    detector = TwoGANFieldDetector(
        seg_weights="src/weights/gan_weights/seg_latest_net_G.pth",
        det_weights="src/weights/gan_weights/detec_latest_net_G.pth",
    )

    # Individual outputs
    grass_mask = detector.segment(frame)       # field segmentation, same HxW as frame
    line_mask  = detector.detect_lines(frame)  # pitch line detection, same HxW as frame

    # Or both in one call (one preprocessing step, two forward passes)
    seg, lines = detector.detect_field(frame)

Reference:
    Chen, J. & Little, J.J. (2019)
    "Sports Camera Calibration via Synthetic Data"
    CVPR Workshops. arXiv:1810.10658
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from .two_gan import define_G


class TwoGANFieldDetector:
    """High-level wrapper for the two-GAN field detection model."""

    # Architecture constants — must match training configuration
    _INPUT_NC = 3
    _OUTPUT_NC = 1
    _NGF = 64
    _NETG = "unet_256"
    _NORM = "batch"
    _USE_DROPOUT = False
    _INIT_TYPE = "normal"
    _INPUT_SIZE = 256  # GANs were trained on 256x256

    def __init__(
        self,
        seg_weights: str,
        det_weights: str,
        device: str = "cpu",
    ):
        """Load both generators and move to the requested device.

        Args:
            seg_weights: path to seg_latest_net_G.pth (field segmentation)
            det_weights: path to detec_latest_net_G.pth (line detection)
            device: 'cpu', 'cuda', or 'cuda:N'
        """
        seg_path = Path(seg_weights)
        det_path = Path(det_weights)

        if not seg_path.is_file():
            raise FileNotFoundError(
                f"Segmentation weights not found at {seg_path}. "
                f"Run `python src/weights/download_weights.py` to fetch them."
            )
        if not det_path.is_file():
            raise FileNotFoundError(
                f"Detection weights not found at {det_path}. "
                f"Run `python src/weights/download_weights.py` to fetch them."
            )

        self.device = torch.device(device)

        self.seg_G = self._build_and_load(str(seg_path))
        self.det_G = self._build_and_load(str(det_path))


    def _build_and_load(self, weights_path: str) -> torch.nn.Module:
        """Build a generator with the training architecture and load weights."""
        generator = define_G(
            self._INPUT_NC,
            self._OUTPUT_NC,
            self._NGF,
            self._NETG,
            self._NORM,
            self._USE_DROPOUT,
            self._INIT_TYPE,
        )
        state = torch.load(weights_path, map_location=self.device)
        generator.load_state_dict(state)
        generator.to(self.device)
        generator.eval()
        return generator

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """BGR frame (any size) → (1, 3, 256, 256) tensor in [-1, 1] on device."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Expected BGR frame with shape (H, W, 3), got {frame.shape}"
            )

        resized = cv2.resize(frame, (self._INPUT_SIZE, self._INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = (tensor - 0.5) / 0.5  # [0, 1] → [-1, 1]
        return tensor.to(self.device)

    def _postprocess(
        self,
        output: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Generator output in [-1, 1] → uint8 grayscale mask in [0, 255].

        Args:
            output: (1, 1, 256, 256) tensor from generator
            target_size: (height, width) to resize to, or None for native 256x256

        Returns:
            np.ndarray: uint8 grayscale image
        """
        # Bring to CPU, strip batch and channel dims → (256, 256)
        arr = output[0, 0].detach().cpu().numpy()

        # Denormalise from [-1, 1] to [0, 255]
        arr = ((arr + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)

        if target_size is not None:
            h, w = target_size
            arr = cv2.resize(arr, (w, h))

        return arr

    @torch.no_grad()
    def segment(
        self,
        frame: np.ndarray,
        resize_to_input: bool = True,
    ) -> np.ndarray:
        """Run field segmentation on a frame.

        Args:
            frame: BGR image of any size
            resize_to_input: if True, output is resized back to frame's HxW

        Returns:
            np.ndarray: grayscale mask, uint8, 0..255 (high = grass/field)
        """
        tensor = self._preprocess(frame)
        out = self.seg_G(tensor)
        target = frame.shape[:2] if resize_to_input else None
        return self._postprocess(out, target_size=target)

    @torch.no_grad()
    def detect_lines(
        self,
        frame: np.ndarray,
        resize_to_input: bool = True,
    ) -> np.ndarray:
        """Run line detection on a frame.

        Args:
            frame: BGR image of any size
            resize_to_input: if True, output is resized back to frame's HxW

        Returns:
            np.ndarray: grayscale mask, uint8, 0..255 (high = pitch line)
        """
        tensor = self._preprocess(frame)
        out = self.det_G(tensor)
        target = frame.shape[:2] if resize_to_input else None
        return self._postprocess(out, target_size=target)

    @torch.no_grad()
    def detect_field(
        self,
        frame: np.ndarray,
        resize_to_input: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run both generators on a single preprocessed input.

        More efficient than calling segment() and detect_lines() separately
        because the input is preprocessed only once.

        Args:
            frame: BGR image of any size
            resize_to_input: if True, both outputs are resized back to frame's HxW

        Returns:
            (segmentation, lines) — both grayscale uint8 0..255
        """
        tensor = self._preprocess(frame)
        seg_out = self.seg_G(tensor)
        det_out = self.det_G(tensor)
        target = frame.shape[:2] if resize_to_input else None
        return (
            self._postprocess(seg_out, target_size=target),
            self._postprocess(det_out, target_size=target),
        )


if __name__ == "__main__":
    # Smoke test: load both generators and run on a synthetic green frame
    detector = TwoGANFieldDetector(
        seg_weights="src/weights/gan_weights/seg_latest_net_G.pth",
        det_weights="src/weights/gan_weights/detec_latest_net_G.pth",
        device="cpu",
    )

    # Fake frame: 1280x720, all grass-coloured
    fake_frame = np.full((720, 1280, 3), [34, 139, 34], dtype=np.uint8)

    seg, lines = detector.detect_field(fake_frame)

    print(f"Input frame:  {fake_frame.shape} (HxWxC)")
    print(f"Segmentation: {seg.shape}, dtype={seg.dtype}, "
          f"range=[{seg.min()}, {seg.max()}]")
    print(f"Lines:        {lines.shape}, dtype={lines.dtype}, "
          f"range=[{lines.min()}, {lines.max()}]")
    print("TwoGANFieldDetector test OK")