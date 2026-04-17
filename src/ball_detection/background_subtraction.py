"""
background_subtraction.py — Remove static objects for ball detection.

Uses Gaussian Mixture Model (MOG2) to learn the static background
from video frames. Only moving regions are passed to the ball
detector, eliminating false positives from:
    - Circular advertising logos (the "O" in "SONY" problem)
    - Pitch markings (centre circle, corner arcs)
    - Fixed stadium structures

"""

import numpy as np
import cv2
from typing import Optional, Tuple


class BackgroundSubtractor:
    """
    Background subtraction for filtering static objects in ball detection.

    Wraps OpenCV's MOG2 (Mixture of Gaussians) with football-specific
    preprocessing: morphological cleanup, minimum foreground area
    filtering, and optional grass masking.

    The first ~30 frames are used to learn the background model.
    After that, the model adapts gradually to handle lighting changes.
    """

    def __init__(
        self,
        history: int = 200,
        var_threshold: float = 50.0,
        detect_shadows: bool = False,
        learning_rate: float = -1,
        min_area: int = 50,
        morph_kernel_size: int = 5,
    ):
        """
        Args:
            history: number of frames to build the background model
            var_threshold: variance threshold for MOG2 (higher = less sensitive)
            detect_shadows: whether to detect and mark shadows
            learning_rate: background learning rate (-1 = auto)
            min_area: minimum foreground region area (pixels) to keep
            morph_kernel_size: kernel size for morphological cleanup
        """
        self.learning_rate = learning_rate
        self.min_area = min_area

        # Create MOG2 background subtractor
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )

        # Morphological kernels for cleanup
        self.open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        self.close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size * 2 + 1, morph_kernel_size * 2 + 1)
        )

        self.frame_count = 0

    def apply(
        self,
        frame: np.ndarray,
        grass_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply background subtraction to a frame.

        Args:
            frame: BGR image (H, W, 3), uint8
            grass_mask: optional grass area mask. If provided, only
                       foreground within the grass area is returned.

        Returns:
            np.ndarray: binary foreground mask (H, W), uint8
                       255 = moving object, 0 = background
        """
        # Apply MOG2
        if self.learning_rate >= 0:
            fg_mask = self.mog2.apply(frame, learningRate=self.learning_rate)
        else:
            fg_mask = self.mog2.apply(frame)

        # Threshold to binary (MOG2 with shadows uses 127 for shadows)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological cleanup
        # Opening removes small noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.open_kernel)
        # Closing fills small holes in foreground objects
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.close_kernel)

        # Remove small regions (noise that survived morphological ops)
        fg_mask = self._remove_small_regions(fg_mask)

        # Apply grass mask if provided
        if grass_mask is not None:
            fg_mask = cv2.bitwise_and(fg_mask, grass_mask)

        self.frame_count += 1
        return fg_mask

    def _remove_small_regions(self, mask: np.ndarray) -> np.ndarray:
        """Remove foreground regions smaller than min_area pixels."""
        if self.min_area <= 0:
            return mask

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        cleaned = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_area:
                cv2.drawContours(cleaned, [contour], -1, 255, -1)

        return cleaned

    def get_background(self) -> Optional[np.ndarray]:
        """
        Get the current learned background image.
        Useful for debugging and visualisation.

        Returns:
            np.ndarray or None: background image if available
        """
        bg = self.mog2.getBackgroundImage()
        return bg

    def is_ready(self) -> bool:
        """
        Check if the background model has been sufficiently trained.
        Needs at least ~30 frames to build a stable model.
        """
        return self.frame_count >= 30

    def reset(self) -> None:
        """Reset the background model (e.g. for a new video)."""
        history = self.mog2.getHistory()
        var_thresh = self.mog2.getVarThreshold()
        detect_shadows = self.mog2.getDetectShadows()

        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_thresh,
            detectShadows=detect_shadows,
        )
        self.frame_count = 0


if __name__ == "__main__":
    print("=== Background Subtractor Test ===\n")

    bg_sub = BackgroundSubtractor()

    # Simulate a video: static green background with a moving white ball
    print("1. Simulating 50-frame video sequence...")
    for i in range(50):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:, :] = [34, 139, 34]  # Green grass

        # Static advertising board (should be learned as background)
        cv2.rectangle(frame, (100, 50), (300, 100), (255, 255, 255), -1)
        cv2.putText(frame, "SONY", (130, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        # Moving ball (changes position each frame)
        ball_x = 300 + i * 15
        cv2.circle(frame, (ball_x, 400), 12, (255, 255, 255), -1)

        # Moving player
        player_x = 200 + i * 10
        cv2.rectangle(frame, (player_x, 300), (player_x + 30, 400),
                      (0, 0, 200), -1)

        fg_mask = bg_sub.apply(frame)

        if i == 49:
            fg_pixels = np.sum(fg_mask > 0)
            print(f"   Frame {i}: foreground pixels = {fg_pixels}")
            print(f"   Background model ready: {bg_sub.is_ready()}")

    # Test 2: Check that static objects are removed
    print("\n2. Checking static object removal...")
    # After 50 frames, the SONY logo should be part of the background
    # The ball and player should still be foreground
    bg = bg_sub.get_background()
    if bg is not None:
        print(f"   Background image shape: {bg.shape}")
    else:
        print("   Background image not available")

    # Test 3: Reset
    print("\n3. Testing reset...")
    bg_sub.reset()
    print(f"   Frame count after reset: {bg_sub.frame_count}")
    print(f"   Ready after reset: {bg_sub.is_ready()}")

    print("\n=== Tests complete ===")