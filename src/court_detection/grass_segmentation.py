"""
grass_segmentation.py — Segment the grass area from broadcast frames.

Converts the frame to HSV colour space and thresholds the green
channel to extract the playing field. The resulting mask is used:
    1. By camera calibration — restrict field line detection to the pitch
    2. By player detection — only search for players on the pitch
    3. By ball detection — reduce false positives from off-pitch objects
    4. By team clustering — mask out grass from player bounding boxes

The grass mask also defines the court boundary, which helps
determine which detected objects are players vs spectators.


"""

import numpy as np
import cv2
from typing import Tuple, Optional


class GrassSegmenter:
    """
    Segment the grass playing field from broadcast frames.

    Usage:
        segmenter = GrassSegmenter()
        mask = segmenter.segment(frame)

        # Use mask for other modules
        edges = edge_detector.detect(frame, mask=mask)
        detections = player_detector.detect(frame, mask=mask)
    """

    def __init__(
        self,
        hue_low: int = 30,
        hue_high: int = 85,
        sat_low: int = 30,
        sat_high: int = 255,
        val_low: int = 30,
        val_high: int = 255,
        morph_kernel_size: int = 7,
        min_area_fraction: float = 0.05,
    ):
        """
        Args:
            hue_low/high: Hue range for green grass (OpenCV: 0-180)
            sat_low/high: Saturation range (excludes very pale greens)
            val_low/high: Value range (excludes very dark regions)
            morph_kernel_size: kernel for morphological cleanup
            min_area_fraction: minimum contour area as fraction of image
                              to be considered part of the pitch
        """
        self.hue_low = hue_low
        self.hue_high = hue_high
        self.sat_low = sat_low
        self.sat_high = sat_high
        self.val_low = val_low
        self.val_high = val_high
        self.morph_kernel_size = morph_kernel_size
        self.min_area_fraction = min_area_fraction

    def segment(self, frame: np.ndarray) -> np.ndarray:
        """
        Segment the grass area from a BGR frame.

        Args:
            frame: BGR image (H, W, 3), uint8

        Returns:
            np.ndarray: binary mask (H, W), uint8 (255=grass, 0=not grass)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold green range
        lower = np.array([self.hue_low, self.sat_low, self.val_low])
        upper = np.array([self.hue_high, self.sat_high, self.val_high])
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size)
        )

        # Close: fill small holes in the grass (players, lines)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Open: remove small noise blobs
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Remove small contours (not part of the main pitch)
        mask = self._remove_small_regions(mask, frame.shape[:2])

        return mask

    def segment_with_fill(self, frame: np.ndarray) -> np.ndarray:
        """
        Segment grass and fill the convex hull of the largest region.

        This produces a cleaner mask that covers the entire pitch
        including areas where the grass colour is inconsistent
        (e.g. worn patches, shadows).
        """
        mask = self.segment(frame)

        # Find the largest contour (the pitch)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return mask

        largest = max(contours, key=cv2.contourArea)

        # Fill the convex hull
        hull = cv2.convexHull(largest)
        filled = np.zeros_like(mask)
        cv2.fillConvexPoly(filled, hull, 255)

        return filled

    def get_pitch_boundary(
        self,
        frame: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Get the boundary contour of the pitch.

        Returns:
            np.ndarray or None: contour points of the pitch boundary
        """
        mask = self.segment(frame)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        return max(contours, key=cv2.contourArea)

    def is_on_pitch(
        self,
        mask: np.ndarray,
        x: int,
        y: int,
    ) -> bool:
        """Check if a pixel position is on the grass."""
        h, w = mask.shape
        if 0 <= x < w and 0 <= y < h:
            return mask[y, x] > 0
        return False

    def get_grass_colour_stats(
        self,
        frame: np.ndarray,
    ) -> dict:
        """
        Compute grass colour statistics for the frame.

        Useful for adapting thresholds to different stadiums
        or lighting conditions.

        Returns:
            dict: mean and std of H, S, V within the grass area
        """
        mask = self.segment(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        grass_pixels = hsv[mask > 0]

        if len(grass_pixels) == 0:
            return {"h_mean": 0, "s_mean": 0, "v_mean": 0}

        return {
            "h_mean": float(np.mean(grass_pixels[:, 0])),
            "h_std": float(np.std(grass_pixels[:, 0])),
            "s_mean": float(np.mean(grass_pixels[:, 1])),
            "s_std": float(np.std(grass_pixels[:, 1])),
            "v_mean": float(np.mean(grass_pixels[:, 2])),
            "v_std": float(np.std(grass_pixels[:, 2])),
            "grass_fraction": float(np.sum(mask > 0) / mask.size),
        }

    def auto_calibrate(self, frame: np.ndarray) -> None:
        """
        Auto-calibrate HSV thresholds from a frame.

        Samples the dominant green colour from the frame and adjusts
        thresholds accordingly. Call on the first frame of a new video.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Initial conservative threshold
        lower = np.array([25, 20, 20])
        upper = np.array([90, 255, 255])
        initial_mask = cv2.inRange(hsv, lower, upper)

        grass_pixels = hsv[initial_mask > 0]

        if len(grass_pixels) < 100:
            return

        h_mean = np.mean(grass_pixels[:, 0])
        h_std = np.std(grass_pixels[:, 0])
        s_mean = np.mean(grass_pixels[:, 1])
        v_mean = np.mean(grass_pixels[:, 2])

        # Set thresholds around measured values
        self.hue_low = int(max(0, h_mean - 2.5 * h_std))
        self.hue_high = int(min(180, h_mean + 2.5 * h_std))
        self.sat_low = int(max(0, s_mean * 0.3))
        self.val_low = int(max(0, v_mean * 0.3))

    def _remove_small_regions(
        self,
        mask: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Remove contours smaller than min_area_fraction of the image."""
        total_area = image_shape[0] * image_shape[1]
        min_area = total_area * self.min_area_fraction

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        cleaned = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                cv2.drawContours(cleaned, [contour], -1, 255, -1)

        return cleaned


if __name__ == "__main__":
    print("=== Grass Segmenter Test ===\n")

    segmenter = GrassSegmenter()

    # Create test frame: green pitch with non-green elements
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :] = [34, 139, 34]  # Green grass

    # Add non-grass: crowd area at top
    frame[0:100, :] = [50, 50, 80]
    # Advertising board at bottom
    frame[650:720, :] = [200, 200, 200]
    # White pitch lines
    cv2.line(frame, (640, 100), (640, 650), (255, 255, 255), 3)

    mask = segmenter.segment(frame)
    print(f"1. Basic segmentation:")
    print(f"   Grass pixels: {np.sum(mask > 0)}")
    print(f"   Grass fraction: {np.sum(mask > 0) / mask.size:.2%}")

    # Filled version
    filled = segmenter.segment_with_fill(frame)
    print(f"\n2. Filled segmentation:")
    print(f"   Filled pixels: {np.sum(filled > 0)}")

    # Colour stats
    stats = segmenter.get_grass_colour_stats(frame)
    print(f"\n3. Grass colour stats:")
    print(f"   H mean: {stats['h_mean']:.1f}")
    print(f"   S mean: {stats['s_mean']:.1f}")
    print(f"   V mean: {stats['v_mean']:.1f}")
    print(f"   Fraction: {stats['grass_fraction']:.2%}")

    # Point check
    print(f"\n4. Point checks:")
    print(f"   (640, 300) on pitch: {segmenter.is_on_pitch(mask, 640, 300)}")
    print(f"   (640, 50) on pitch: {segmenter.is_on_pitch(mask, 640, 50)}")

    # Auto calibrate
    print(f"\n5. Auto calibration:")
    print(f"   Before: hue=[{segmenter.hue_low}, {segmenter.hue_high}]")
    segmenter.auto_calibrate(frame)
    print(f"   After:  hue=[{segmenter.hue_low}, {segmenter.hue_high}]")

    print("\n=== Tests complete ===")