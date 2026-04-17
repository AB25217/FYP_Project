"""
circle_hough.py — Modified Circle Hough Transform for ball detection.

Detects the football by finding circular shapes in edge-detected images.
Uses gradient versor (normalised gradient direction) instead of gradient
magnitude, so the detector finds the most COMPLETE circle rather than
the most CONTRASTED one.

This prevents false detections on high-contrast non-ball objects like
the "O" in "SONY" on advertising boards (D'Orazio Figure 2).

Key equations from D'Orazio et al. (2002):
    Gradient versor (eq. 3):
        e_vec(x,y) = [Ex/|E|, Ey/|E|]

    Kernel vector normalised by distance (eq. 4):
        O_vec(x,y) = [cos(atan(y/x))/sqrt(x²+y²), sin(atan(y/x))/sqrt(x²+y²)]

    Detection operator (eq. 1):
        u(x,y) = integral of e_vec . O_vec / (2*pi*(R_max - R_min))

Reference: D'Orazio, Ancona, Cicirelli, Nitti (2002) IEEE ICPR
"""

import numpy as np
import cv2
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class BallDetection:
    """A single ball detection result."""
    x: int              # Centre x coordinate (pixels)
    y: int              # Centre y coordinate (pixels)
    radius: int         # Detected radius (pixels)
    confidence: float   # Detection confidence (0-1)
    detected: bool      # Whether a ball was found

    @property
    def centre(self) -> Tuple[int, int]:
        return (self.x, self.y)

    @property
    def foot_position(self) -> Tuple[int, int]:
        """Bottom of ball — useful for pitch projection."""
        return (self.x, self.y + self.radius)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Bounding box (x, y, w, h)."""
        return (self.x - self.radius, self.y - self.radius,
                self.radius * 2, self.radius * 2)


class CircleHoughDetector:
    """
    Modified Circle Hough Transform ball detector.

    Usage:
        detector = CircleHoughDetector(r_min=5, r_max=25)
        result = detector.detect(frame)
        if result.detected:
            print(f"Ball at ({result.x}, {result.y})")

        # With masks to reduce false positives
        result = detector.detect(frame, mask=grass_mask)
    """

    def __init__(
        self,
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
            r_min: minimum expected ball radius (pixels)
            r_max: maximum expected ball radius (pixels)
            gaussian_sigma: Gaussian blur sigma before edge detection
            canny_low: Canny low threshold
            canny_high: Canny high threshold
            confidence_threshold: minimum confidence to accept detection
            max_edge_points: subsample edge pixels if more than this (speed)
            use_colour_filter: filter for white/light ball regions
        """
        self.r_min = r_min
        self.r_max = r_max
        self.gaussian_sigma = gaussian_sigma
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.confidence_threshold = confidence_threshold
        self.max_edge_points = max_edge_points
        self.use_colour_filter = use_colour_filter

        # Previous detection for temporal consistency
        self._prev_detection = None

    def detect(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> BallDetection:
        """
        Detect the football in a single frame.

        Args:
            frame: BGR image (H, W, 3), uint8
            mask: optional binary mask (255=search here, 0=ignore).
                  Combine grass_mask and background_subtraction mask
                  before passing.

        Returns:
            BallDetection: detection result
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

        # Compute gradient versor (normalised direction)
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

        # Run modified CHT
        result = self._accumulate_votes(edges, grad_x, grad_y)

        # Temporal consistency check
        result = self._temporal_filter(result, w, h)

        self._prev_detection = result
        return result

    def _accumulate_votes(
        self,
        edges: np.ndarray,
        grad_x: np.ndarray,
        grad_y: np.ndarray,
    ) -> BallDetection:
        """
        Core modified CHT accumulation.

        Each edge pixel votes for possible circle centres along its
        gradient direction. Votes are normalised by:
            1. Gradient magnitude (versor) — direction only
            2. Distance from centre (1/r) — equal contribution from all radii

        This implements D'Orazio equations (1)-(4).
        """
        h, w = edges.shape

        # Gradient versor: normalised direction, independent of magnitude
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-8
        e_x = grad_x / grad_mag
        e_y = grad_y / grad_mag

        # Get edge pixel locations
        edge_ys, edge_xs = np.where(edges > 0)
        if len(edge_ys) < 10:
            return BallDetection(0, 0, 0, 0.0, False)

        # Subsample if too many edge pixels
        if len(edge_ys) > self.max_edge_points:
            indices = np.random.choice(
                len(edge_ys), self.max_edge_points, replace=False
            )
            edge_ys = edge_ys[indices]
            edge_xs = edge_xs[indices]

        # Accumulator space
        accumulator = np.zeros((h, w), dtype=np.float64)

        # Vote for each edge pixel
        for idx in range(len(edge_ys)):
            ey, ex = edge_ys[idx], edge_xs[idx]
            gx = e_x[ey, ex]
            gy = e_y[ey, ex]

            # Vote along gradient direction for each candidate radius
            for r in range(self.r_min, self.r_max + 1):
                # Two possible centres: along and against gradient
                for sign in [1, -1]:
                    cx = int(round(ex + sign * gx * r))
                    cy = int(round(ey + sign * gy * r))

                    if 0 <= cx < w and 0 <= cy < h:
                        # Normalised vote: 1/r so all radii contribute equally
                        accumulator[cy, cx] += 1.0 / r

        # Normalise by the annulus area
        normaliser = 2.0 * np.pi * (self.r_max - self.r_min + 1)
        if normaliser > 0:
            accumulator /= normaliser

        # Find peak and estimate radius
        return self._find_peak(accumulator, edges)

    def _find_peak(
        self,
        accumulator: np.ndarray,
        edges: np.ndarray,
    ) -> BallDetection:
        """
        Find the peak in the accumulator and estimate the ball radius
        by counting edge pixels on candidate circles.
        """
        h, w = accumulator.shape

        # Smooth accumulator to avoid noisy peaks
        acc_smooth = cv2.GaussianBlur(accumulator, (5, 5), 1.0)
        max_val = np.max(acc_smooth)

        if max_val < self.confidence_threshold:
            return BallDetection(0, 0, 0, 0.0, False)

        # Peak location
        max_loc = np.unravel_index(np.argmax(acc_smooth), acc_smooth.shape)
        cy, cx = max_loc

        # Estimate radius: find which radius has the most edge pixels
        best_radius = self.r_min
        best_score = 0

        for r in range(self.r_min, self.r_max + 1):
            count = 0
            num_samples = max(20, int(2 * np.pi * r))

            for k in range(num_samples):
                angle = 2 * np.pi * k / num_samples
                px = int(round(cx + r * np.cos(angle)))
                py = int(round(cy + r * np.sin(angle)))

                if 0 <= px < w and 0 <= py < h and edges[py, px] > 0:
                    count += 1

            score = count / num_samples
            if score > best_score:
                best_score = score
                best_radius = r

        confidence = min(best_score * 2.0, 1.0)

        if confidence < self.confidence_threshold:
            return BallDetection(0, 0, 0, 0.0, False)

        return BallDetection(
            x=int(cx), y=int(cy),
            radius=best_radius,
            confidence=float(confidence),
            detected=True
        )

    def _white_colour_filter(self, frame: np.ndarray) -> np.ndarray:
        """
        Mask white/light regions where the ball is likely to be.
        The standard football is predominantly white.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 0, 150])
        upper = np.array([180, 120, 255])
        white_mask = cv2.inRange(hsv, lower, upper)

        # Dilate to include nearby pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        white_mask = cv2.dilate(white_mask, kernel, iterations=2)

        return white_mask

    def _temporal_filter(
        self,
        result: BallDetection,
        img_width: int,
        img_height: int,
    ) -> BallDetection:
        """
        Reject detections that are too far from the previous frame's
        ball position. The ball can't teleport across the pitch.
        """
        if not result.detected:
            return result

        if self._prev_detection is not None and self._prev_detection.detected:
            dist = np.sqrt(
                (result.x - self._prev_detection.x) ** 2 +
                (result.y - self._prev_detection.y) ** 2
            )
            max_displacement = max(img_width, img_height) * 0.3
            if dist > max_displacement:
                return BallDetection(0, 0, 0, 0.0, False)

        return result

    def detect_opencv_fallback(self, frame: np.ndarray) -> BallDetection:
        """
        Fallback using OpenCV's built-in HoughCircles.
        Useful for comparison in your evaluation section.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
            param1=self.canny_high, param2=30,
            minRadius=self.r_min, maxRadius=self.r_max,
        )

        if circles is not None:
            best = np.round(circles[0, 0]).astype(int)
            return BallDetection(
                x=int(best[0]), y=int(best[1]),
                radius=int(best[2]), confidence=0.8, detected=True
            )

        return BallDetection(0, 0, 0, 0.0, False)

    def reset_tracking(self):
        """Reset temporal state for new video."""
        self._prev_detection = None


def draw_ball_detection(
    frame: np.ndarray,
    detection: BallDetection,
    colour: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw ball detection circle and label on frame."""
    if detection.detected:
        cv2.circle(frame, detection.centre, detection.radius, colour, thickness)
        cv2.circle(frame, detection.centre, 2, colour, -1)
        label = f"Ball ({detection.confidence:.2f})"
        cv2.putText(frame, label,
                    (detection.x - 30, detection.y - detection.radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    return frame


if __name__ == "__main__":
    print("=== Circle Hough Detector Test ===\n")

    # Synthetic test frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :] = [34, 139, 34]
    cv2.circle(frame, (640, 360), 15, (255, 255, 255), -1)

    detector = CircleHoughDetector(r_min=8, r_max=25)

    # Custom CHT
    result = detector.detect(frame)
    print(f"Custom CHT: detected={result.detected}")

    # OpenCV fallback
    result_cv = detector.detect_opencv_fallback(frame)
    print(f"OpenCV CHT: detected={result_cv.detected}, "
          f"pos=({result_cv.x},{result_cv.y}), r={result_cv.radius}")

    # Empty frame
    empty = np.zeros((720, 1280, 3), dtype=np.uint8)
    empty[:, :] = [34, 139, 34]
    result_empty = detector.detect(empty)
    print(f"Empty frame: detected={result_empty.detected}")

    print("\n=== Tests complete ===")