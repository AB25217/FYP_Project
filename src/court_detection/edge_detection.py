"""
edge_detector.py — Edge detection on broadcast frames.

Applies Gaussian blur followed by Canny edge detection to extract
edges from frames. Used as input for pitch line detection and
as a comparison baseline for the U-Net field marking detector.

The edge detector can be applied globally or within a mask
(e.g. only within the grass area) to reduce noise from crowds,
advertising boards, and other non-pitch elements.

"""

import numpy as np
import cv2
from typing import Optional, Tuple


class EdgeDetector:
    """
    Canny edge detection with preprocessing.

    Usage:
        detector = EdgeDetector()
        edges = detector.detect(frame)

        # With grass mask to only detect edges on the pitch
        edges = detector.detect(frame, mask=grass_mask)
    """

    def __init__(
        self,
        gaussian_sigma: float = 1.5,
        canny_low: int = 50,
        canny_high: int = 150,
        use_bilateral: bool = False,
        bilateral_d: int = 9,
        bilateral_sigma_colour: float = 75,
        bilateral_sigma_space: float = 75,
    ):
        """
        Args:
            gaussian_sigma: sigma for Gaussian blur
            canny_low: Canny low threshold
            canny_high: Canny high threshold
            use_bilateral: use bilateral filter instead of Gaussian
                          (preserves edges better but slower)
            bilateral_d: bilateral filter diameter
            bilateral_sigma_colour: bilateral colour sigma
            bilateral_sigma_space: bilateral spatial sigma
        """
        self.gaussian_sigma = gaussian_sigma
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.use_bilateral = use_bilateral
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_colour = bilateral_sigma_colour
        self.bilateral_sigma_space = bilateral_sigma_space

    def detect(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Detect edges in a frame.

        Args:
            frame: BGR image (H, W, 3) or grayscale (H, W), uint8
            mask: optional binary mask (255=process, 0=ignore)

        Returns:
            np.ndarray: binary edge image (H, W), uint8
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Preprocessing: blur to reduce noise
        if self.use_bilateral:
            blurred = cv2.bilateralFilter(
                gray, self.bilateral_d,
                self.bilateral_sigma_colour,
                self.bilateral_sigma_space
            )
        else:
            ksize = int(self.gaussian_sigma * 6) | 1
            blurred = cv2.GaussianBlur(gray, (ksize, ksize), self.gaussian_sigma)

        # Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Apply mask
        if mask is not None:
            edges = cv2.bitwise_and(edges, mask)

        return edges

    def detect_adaptive(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Edge detection with automatic threshold selection based on
        median pixel intensity. Adapts to different lighting conditions.

        Uses Otsu's method to find optimal thresholds.
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        ksize = int(self.gaussian_sigma * 6) | 1
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), self.gaussian_sigma)

        # Compute adaptive thresholds from image statistics
        median = np.median(blurred)
        low = int(max(0, 0.67 * median))
        high = int(min(255, 1.33 * median))

        edges = cv2.Canny(blurred, low, high)

        if mask is not None:
            edges = cv2.bitwise_and(edges, mask)

        return edges


if __name__ == "__main__":
    print("=== Edge Detector Test ===\n")

    detector = EdgeDetector()

    # Create test frame with lines
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :] = [34, 139, 34]
    cv2.line(frame, (0, 360), (1280, 360), (255, 255, 255), 3)
    cv2.line(frame, (640, 0), (640, 720), (255, 255, 255), 3)
    cv2.circle(frame, (640, 360), 100, (255, 255, 255), 3)

    edges = detector.detect(frame)
    print(f"Standard edges: {np.sum(edges > 0)} edge pixels")

    edges_adaptive = detector.detect_adaptive(frame)
    print(f"Adaptive edges: {np.sum(edges_adaptive > 0)} edge pixels")

    # With mask
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask[100:600, 100:1100] = 255
    edges_masked = detector.detect(frame, mask=mask)
    print(f"Masked edges: {np.sum(edges_masked > 0)} edge pixels")

    print("\n=== Tests complete ===")