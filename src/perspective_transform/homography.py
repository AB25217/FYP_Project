"""
homography.py — Compute and manage homography matrices.

The homography H is a 3x3 matrix that maps between image pixel
coordinates and pitch world coordinates (metres).

Two directions:
    Forward:  pitch → image   (H maps pitch points to pixels)
    Inverse:  image → pitch   (H_inv maps pixels to pitch points)

For player projection, we need the INVERSE homography:
    Given a player's foot position in pixels, where are they on the pitch?

The homography comes from the camera calibration module (camera pose
engine or pose tracker). This module wraps it for convenient use.

Since the pitch is flat (z=0), the homography is derived from the
3x4 projection matrix by dropping the third column:
    P = [p1 p2 p3 p4]  →  H = [p1 p2 p4]

Reference: 
    Mavrogiannis & Maglogiannis (2022) Section 3.2, equation (2)
    Chen & Little (2019) — homography from camera pose
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List


class HomographyManager:
    """
    Manage homography matrices for perspective transformation.

    Usage:
        # From camera calibration pipeline
        hm = HomographyManager()
        hm.set_homography(H)

        # Project a pixel point to pitch coordinates
        pitch_x, pitch_y = hm.pixel_to_pitch(640, 500)

        # Or from manual correspondences (fallback)
        hm.compute_from_correspondences(
            pixel_points=[(100, 400), (1100, 400), (600, 200), (600, 600)],
            pitch_points=[(0, 0), (105, 0), (52.5, 0), (52.5, 68)]
        )
    """

    def __init__(self):
        self._H = None          # Forward: pitch → image
        self._H_inv = None      # Inverse: image → pitch
        self._is_valid = False

    def set_homography(self, H: np.ndarray) -> None:
        """
        Set the homography matrix (pitch → image direction).

        This is what the camera calibration module produces.
        The inverse is automatically computed for pixel → pitch conversion.

        Args:
            H: 3x3 homography matrix
        """
        self._H = H.astype(np.float64)

        try:
            self._H_inv = np.linalg.inv(self._H)
            self._is_valid = True
        except np.linalg.LinAlgError:
            self._H_inv = None
            self._is_valid = False
            print("Warning: homography is singular, cannot compute inverse")

    def set_from_projection_matrix(self, P: np.ndarray) -> None:
        """
        Derive homography from a 3x4 projection matrix.

        Since the pitch is at z=0, the homography is obtained
        by dropping the 3rd column of P:
            P = [p1 p2 p3 p4]  →  H = [p1 p2 p4]

        Args:
            P: 3x4 projection matrix from camera model
        """
        H = P[:, [0, 1, 3]]
        self.set_homography(H)

    def compute_from_correspondences(
        self,
        pixel_points: List[Tuple[float, float]],
        pitch_points: List[Tuple[float, float]],
    ) -> bool:
        """
        Compute homography from manual point correspondences.

        Fallback method when camera calibration is unavailable.
        Needs at least 4 point pairs.

        Args:
            pixel_points: list of (x, y) in image pixels
            pitch_points: list of (x, y) in pitch metres

        Returns:
            bool: True if homography was computed successfully
        """
        if len(pixel_points) < 4 or len(pitch_points) < 4:
            print("Need at least 4 point correspondences")
            return False

        src = np.array(pitch_points, dtype=np.float64)
        dst = np.array(pixel_points, dtype=np.float64)

        # H maps pitch → image
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        if H is not None:
            self.set_homography(H)
            return True
        else:
            print("Failed to compute homography from correspondences")
            return False

    def pixel_to_pitch(
        self,
        pixel_x: float,
        pixel_y: float,
    ) -> Optional[Tuple[float, float]]:
        """
        Convert a single pixel position to pitch coordinates.

        Args:
            pixel_x: x coordinate in image pixels
            pixel_y: y coordinate in image pixels

        Returns:
            tuple: (pitch_x, pitch_y) in metres, or None if invalid
        """
        if not self._is_valid:
            return None

        # Homogeneous pixel coordinates
        pixel_h = np.array([pixel_x, pixel_y, 1.0])

        # Apply inverse homography
        pitch_h = self._H_inv @ pixel_h

        # Convert from homogeneous
        if abs(pitch_h[2]) < 1e-8:
            return None

        pitch_x = pitch_h[0] / pitch_h[2]
        pitch_y = pitch_h[1] / pitch_h[2]

        return (float(pitch_x), float(pitch_y))

    def pixel_to_pitch_batch(
        self,
        pixel_points: np.ndarray,
    ) -> np.ndarray:
        """
        Convert multiple pixel positions to pitch coordinates.

        Args:
            pixel_points: (N, 2) array of pixel coordinates

        Returns:
            np.ndarray: (N, 2) array of pitch coordinates in metres.
                       Invalid points are set to (NaN, NaN).
        """
        if not self._is_valid or len(pixel_points) == 0:
            return np.full((len(pixel_points), 2), np.nan)

        N = len(pixel_points)

        # Add homogeneous coordinate
        ones = np.ones((N, 1))
        pixel_h = np.hstack([pixel_points, ones])  # (N, 3)

        # Apply inverse homography
        pitch_h = (self._H_inv @ pixel_h.T).T  # (N, 3)

        # Convert from homogeneous
        w = pitch_h[:, 2]
        valid = np.abs(w) > 1e-8

        result = np.full((N, 2), np.nan)
        result[valid, 0] = pitch_h[valid, 0] / w[valid]
        result[valid, 1] = pitch_h[valid, 1] / w[valid]

        return result

    def pitch_to_pixel(
        self,
        pitch_x: float,
        pitch_y: float,
    ) -> Optional[Tuple[float, float]]:
        """
        Convert pitch coordinates to pixel position.
        Useful for drawing pitch markings on the image.

        Args:
            pitch_x: x in metres
            pitch_y: y in metres

        Returns:
            tuple: (pixel_x, pixel_y), or None if invalid
        """
        if not self._is_valid:
            return None

        pitch_h = np.array([pitch_x, pitch_y, 1.0])
        pixel_h = self._H @ pitch_h

        if abs(pixel_h[2]) < 1e-8:
            return None

        px = pixel_h[0] / pixel_h[2]
        py = pixel_h[1] / pixel_h[2]

        return (float(px), float(py))

    def is_valid_pitch_position(
        self,
        pitch_x: float,
        pitch_y: float,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        margin: float = 5.0,
    ) -> bool:
        """
        Check if a projected pitch position is within bounds.

        Args:
            pitch_x, pitch_y: position in metres
            pitch_length: pitch length (default 105m)
            pitch_width: pitch width (default 68m)
            margin: allow positions slightly outside pitch

        Returns:
            bool: True if position is within bounds
        """
        return (
            -margin <= pitch_x <= pitch_length + margin and
            -margin <= pitch_y <= pitch_width + margin
        )

    @property
    def is_valid(self) -> bool:
        return self._is_valid

    @property
    def homography(self) -> Optional[np.ndarray]:
        return self._H.copy() if self._H is not None else None

    @property
    def inverse_homography(self) -> Optional[np.ndarray]:
        return self._H_inv.copy() if self._H_inv is not None else None


if __name__ == "__main__":
    print("=== Homography Manager Test ===\n")

    hm = HomographyManager()

    # Test with a known homography (simple scaling + translation)
    print("1. Testing with synthetic homography...")
    H = np.array([
        [10.0,  0.0,  100.0],
        [ 0.0,  8.0,   50.0],
        [ 0.0,  0.0,    1.0],
    ])
    hm.set_homography(H)
    print(f"   Valid: {hm.is_valid}")

    # Pitch origin (0,0) should map to pixel (100, 50)
    px = hm.pitch_to_pixel(0, 0)
    print(f"   Pitch (0,0) -> pixel {px}")

    # Pixel (100, 50) should map back to pitch (0,0)
    pt = hm.pixel_to_pitch(100, 50)
    print(f"   Pixel (100,50) -> pitch {pt}")

    # Test batch
    print("\n2. Testing batch conversion...")
    pixels = np.array([[100, 50], [200, 130], [150, 90]], dtype=np.float64)
    pitch_pts = hm.pixel_to_pitch_batch(pixels)
    print(f"   Pixels: {pixels.tolist()}")
    print(f"   Pitch:  {np.round(pitch_pts, 2).tolist()}")

    # Test bounds check
    print("\n3. Testing bounds validation...")
    print(f"   (52.5, 34) valid: {hm.is_valid_pitch_position(52.5, 34)}")
    print(f"   (200, 34) valid: {hm.is_valid_pitch_position(200, 34)}")
    print(f"   (-3, 34) valid: {hm.is_valid_pitch_position(-3, 34)}")

    # Test from correspondences
    print("\n4. Testing from correspondences...")
    hm2 = HomographyManager()
    pixel_pts = [(0, 0), (1000, 0), (1000, 600), (0, 600)]
    pitch_pts_corr = [(0, 0), (105, 0), (105, 68), (0, 68)]
    success = hm2.compute_from_correspondences(pixel_pts, pitch_pts_corr)
    print(f"   Computed: {success}")
    if success:
        result = hm2.pixel_to_pitch(500, 300)
        print(f"   Pixel (500,300) -> pitch ({result[0]:.1f}, {result[1]:.1f})")

    print("\n=== Tests complete ===")