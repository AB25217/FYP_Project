"""
semicircle_detector.py — Shadow-aware ball detection for natural light.

When matches are played in sunlight, the ball appears as a bright
semicircle rather than a full circle due to self-shadowing. The
shadow angle depends on the sun's direction.

"""

import numpy as np
import cv2
from typing import Optional, Tuple
from .circle_hough import BallDetection, CircleHoughDetector


class SemicircleDetector(CircleHoughDetector):
    """
    Semicircle-aware ball detector for natural light conditions.

    Extends CircleHoughDetector with a modified kernel that only
    responds to semicircles with a specific shadow angle.

    Usage:
        # Detect with shadow angle of -10 degrees
        detector = SemicircleDetector(shadow_angle=-10.0)
        result = detector.detect(frame)

        # Auto-detect best shadow angle
        result = detector.detect_best_angle(frame)
    """

    def __init__(
        self,
        shadow_angle: float = 0.0,
        r_min: int = 5,
        r_max: int = 30,
        gaussian_sigma: float = 1.5,
        canny_low: int = 50,
        canny_high: int = 150,
        confidence_threshold: float = 0.3,
        max_edge_points: int = 3000,
        use_colour_filter: bool = True,
    ):
        """
        Args:
            shadow_angle: angle of the semicircle diameter in degrees.
                         0 = shadow on bottom half, ball lit from above.
                         Negative = shadow rotated clockwise.
                         D'Orazio uses -10 degrees in their examples.
            Other args: same as CircleHoughDetector
        """
        super().__init__(
            r_min=r_min, r_max=r_max,
            gaussian_sigma=gaussian_sigma,
            canny_low=canny_low, canny_high=canny_high,
            confidence_threshold=confidence_threshold,
            max_edge_points=max_edge_points,
            use_colour_filter=use_colour_filter,
        )
        self.shadow_angle = shadow_angle

    def detect(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> BallDetection:
        """
        Detect ball using semicircle kernel.
        Same interface as CircleHoughDetector.detect().
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Build search mask
        search_mask = np.ones((h, w), dtype=np.uint8) * 255

        if self.use_colour_filter:
            search_mask = cv2.bitwise_and(
                search_mask, self._white_colour_filter(frame)
            )

        if mask is not None:
            search_mask = cv2.bitwise_and(search_mask, mask)

        # Edge detection
        ksize = int(self.gaussian_sigma * 6) | 1
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), self.gaussian_sigma)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        edges = cv2.bitwise_and(edges, search_mask)

        # Gradient
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

        # Semicircle accumulation
        result = self._accumulate_semicircle(edges, grad_x, grad_y)

        # Temporal filter
        result = self._temporal_filter(result, w, h)
        self._prev_detection = result
        return result

    def _accumulate_semicircle(
        self,
        edges: np.ndarray,
        grad_x: np.ndarray,
        grad_y: np.ndarray,
    ) -> BallDetection:
        """
        Modified CHT accumulation for semicircles.

        Only edge pixels on the lit side of the ball (determined by
        shadow_angle) contribute votes. The lit side boundary is
        defined by the line y >= tan(alpha) * x through each
        candidate centre.

        Implements D'Orazio equation for O_alpha kernel.
        """
        h, w = edges.shape
        alpha_rad = np.deg2rad(self.shadow_angle)

        # Gradient versor
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-8
        e_x = grad_x / grad_mag
        e_y = grad_y / grad_mag

        # Edge pixels
        edge_ys, edge_xs = np.where(edges > 0)
        if len(edge_ys) < 10:
            return BallDetection(0, 0, 0, 0.0, False)

        # Subsample
        if len(edge_ys) > self.max_edge_points:
            indices = np.random.choice(
                len(edge_ys), self.max_edge_points, replace=False
            )
            edge_ys = edge_ys[indices]
            edge_xs = edge_xs[indices]

        accumulator = np.zeros((h, w), dtype=np.float64)
        tan_alpha = np.tan(alpha_rad)
        half_r = (self.r_max + self.r_min) / 2

        for idx in range(len(edge_ys)):
            ey, ex = edge_ys[idx], edge_xs[idx]
            gx = e_x[ey, ex]
            gy = e_y[ey, ex]

            for r in range(self.r_min, self.r_max + 1):
                for sign in [1, -1]:
                    dx = sign * gx * r
                    dy = sign * gy * r

                    # Only accumulate if on the lit side of the semicircle
                    # Boundary: y >= tan(alpha) * x - tolerance
                    if dy >= tan_alpha * dx - half_r:
                        cx = int(round(ex + dx))
                        cy = int(round(ey + dy))

                        if 0 <= cx < w and 0 <= cy < h:
                            accumulator[cy, cx] += 1.0 / r

        # Normalise (half circle = pi instead of 2*pi)
        normaliser = np.pi * (self.r_max - self.r_min + 1)
        if normaliser > 0:
            accumulator /= normaliser

        return self._find_peak(accumulator, edges)

    def detect_best_angle(
        self,
        frame: np.ndarray,
        angles: list = None,
        mask: Optional[np.ndarray] = None,
    ) -> BallDetection:
        """
        Try multiple shadow angles and return the best detection.

        Useful when the shadow direction is unknown or changes
        during the match. Tests several angles and picks the
        detection with highest confidence.

        Args:
            frame: BGR image
            angles: list of angles to try (degrees).
                   Default: [-20, -10, 0, 10, 20]
            mask: optional search mask

        Returns:
            BallDetection: best detection across all angles
        """
        if angles is None:
            angles = [-20.0, -10.0, 0.0, 10.0, 20.0]

        best_result = BallDetection(0, 0, 0, 0.0, False)
        best_confidence = 0.0
        original_angle = self.shadow_angle

        for angle in angles:
            self.shadow_angle = angle
            result = self.detect(frame, mask=mask)

            if result.detected and result.confidence > best_confidence:
                best_confidence = result.confidence
                best_result = result

        self.shadow_angle = original_angle
        return best_result


if __name__ == "__main__":
    print("=== Semicircle Detector Test ===\n")

    # Create frame with a semicircle (simulating shadowed ball)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :] = [34, 139, 34]

    # Draw a semicircle — top half only (shadow on bottom)
    cv2.ellipse(frame, (640, 360), (15, 15), 0, -180, 0,
                (255, 255, 255), -1)

    # Test single angle
    detector = SemicircleDetector(shadow_angle=0.0, r_min=8, r_max=25)
    result = detector.detect(frame)
    print(f"Single angle (0°): detected={result.detected}")

    # Test best angle
    result_best = detector.detect_best_angle(frame)
    print(f"Best angle: detected={result_best.detected}")
    if result_best.detected:
        print(f"  Position: ({result_best.x}, {result_best.y})")
        print(f"  Confidence: {result_best.confidence:.3f}")

    print("\n=== Tests complete ===")