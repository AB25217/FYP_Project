"""
pose_tracker.py — Track camera pose across video frames using ECC.

Between full camera pose estimations (which are expensive), this module
tracks how the camera pose changes frame-to-frame using the Enhanced
Correlation Coefficient (ECC) algorithm.


How it works:
    1. Frame 0: full estimation → get homography H_0 and edge map E_0
    2. Frame 1: detect edges E_1, use ECC to find warp from E_0 to E_1
       → H_1 = warp @ H_0
    3. Frame 2: detect edges E_2, use ECC from E_1 to E_2
       → H_2 = warp @ H_1
    ... continue until frame 200
    4. Frame 200: full re-estimation → reset H and E

The tracker also monitors alignment quality and triggers early
re-estimation if tracking drifts (correlation drops below threshold).

"""

import numpy as np
import cv2
from typing import Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class TrackingState:
    """Current state of the camera pose tracker."""
    homography: np.ndarray              # Current estimated homography
    edge_image: np.ndarray              # Edge image from current frame
    frame_index: int                    # Current frame number
    frames_since_estimation: int        # Frames since last full estimation
    is_estimated: bool                  # True if this frame used full estimation
    correlation: float                  # ECC correlation (quality measure)
    tracking_valid: bool                # Whether tracking is currently reliable


class CameraPoseTracker:
    """
    Tracks camera pose across video frames.

    Alternates between full estimation (expensive, accurate) and
    tracking (cheap, accumulates drift). Re-estimates when either
    the tracking period expires or tracking quality drops.

    Usage:
        tracker = CameraPoseTracker(estimation_interval=200)

        for frame_idx, frame in enumerate(video_frames):
            # Get edge image from field marking detector
            edge_image = unet.predict(frame)

            if tracker.needs_estimation():
                # Run full pipeline: siamese → database → refinement
                homography = full_estimation_pipeline(edge_image)
                state = tracker.set_estimation(homography, edge_image, frame_idx)
            else:
                # Track from previous frame
                state = tracker.track(edge_image, frame_idx)

            # Use state.homography for player position projection
            if state.tracking_valid:
                project_players(detections, state.homography)
    """

    def __init__(
        self,
        estimation_interval: int = 200,
        min_correlation: float = 0.7,
        max_iterations: int = 50,
        epsilon: float = 1e-5,
        warp_mode: str = "homography",
    ):
        """
        Args:
            estimation_interval: run full estimation every N frames
            min_correlation: minimum ECC correlation before triggering re-estimation
            max_iterations: max iterations for ECC per frame
            epsilon: convergence threshold for ECC
            warp_mode: 'homography' or 'affine'
        """
        self.estimation_interval = estimation_interval
        self.min_correlation = min_correlation
        self.max_iterations = max_iterations
        self.epsilon = epsilon

        if warp_mode == "homography":
            self.warp_mode = cv2.MOTION_HOMOGRAPHY
        elif warp_mode == "affine":
            self.warp_mode = cv2.MOTION_AFFINE
        else:
            raise ValueError(f"Unknown warp_mode: {warp_mode}")

        self.criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.max_iterations,
            self.epsilon,
        )

        # Internal state
        self._current_homography = None
        self._previous_edge = None
        self._frames_since_estimation = 0
        self._frame_index = 0
        self._initialised = False
        self._last_correlation = 1.0

        # History for analysis
        self._correlation_history = []
        self._estimation_frames = []

    def needs_estimation(self) -> bool:
        """
        Check whether a full pose estimation is needed.

        Returns True if:
            - Tracker has not been initialised yet
            - Enough frames have passed since last estimation
            - Tracking quality has dropped below threshold
        """
        if not self._initialised:
            return True

        if self._frames_since_estimation >= self.estimation_interval:
            return True

        if self._last_correlation < self.min_correlation:
            return True

        return False

    def set_estimation(
        self,
        homography: np.ndarray,
        edge_image: np.ndarray,
        frame_index: int,
    ) -> TrackingState:
        """
        Set a new keyframe from full pose estimation.
        Resets the tracking state.

        Args:
            homography: 3x3 homography from full estimation pipeline
            edge_image: edge image for this frame (for subsequent tracking)
            frame_index: current frame number

        Returns:
            TrackingState: state for this frame
        """
        self._current_homography = homography.copy()
        self._previous_edge = self._prepare_edge(edge_image)
        self._frames_since_estimation = 0
        self._frame_index = frame_index
        self._initialised = True
        self._last_correlation = 1.0

        self._estimation_frames.append(frame_index)
        self._correlation_history.append(1.0)

        return TrackingState(
            homography=self._current_homography.copy(),
            edge_image=edge_image,
            frame_index=frame_index,
            frames_since_estimation=0,
            is_estimated=True,
            correlation=1.0,
            tracking_valid=True,
        )

    def track(
        self,
        edge_image: np.ndarray,
        frame_index: int,
    ) -> TrackingState:
        """
        Track camera pose from previous frame to current frame
        using ECC alignment on edge images.

        Args:
            edge_image: edge image for current frame
            frame_index: current frame number

        Returns:
            TrackingState: state for this frame
        """
        if not self._initialised:
            raise RuntimeError("Tracker not initialised. Call set_estimation() first.")

        current_edge = self._prepare_edge(edge_image)

        # Check both images have enough content
        if (np.sum(self._previous_edge > 0.1) < 30 or
            np.sum(current_edge > 0.1) < 30):
            # Not enough edge content — keep previous homography
            self._frames_since_estimation += 1
            self._frame_index = frame_index
            self._last_correlation = 0.0
            self._correlation_history.append(0.0)

            return TrackingState(
                homography=self._current_homography.copy(),
                edge_image=edge_image,
                frame_index=frame_index,
                frames_since_estimation=self._frames_since_estimation,
                is_estimated=False,
                correlation=0.0,
                tracking_valid=False,
            )

        # Initialise warp as identity
        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Run ECC alignment: find warp from previous edge to current edge
        try:
            correlation, warp_matrix = cv2.findTransformECC(
                self._previous_edge,
                current_edge,
                warp_matrix,
                self.warp_mode,
                self.criteria,
            )
            tracking_valid = correlation >= self.min_correlation

        except cv2.error:
            correlation = 0.0
            tracking_valid = False
            warp_matrix = np.eye(3, dtype=np.float32)

        # Update homography: H_current = warp @ H_previous
        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            self._current_homography = warp_matrix.astype(np.float64) @ self._current_homography
        else:
            warp_3x3 = np.eye(3, dtype=np.float64)
            warp_3x3[:2, :] = warp_matrix.astype(np.float64)
            self._current_homography = warp_3x3 @ self._current_homography

        # Update state
        self._previous_edge = current_edge
        self._frames_since_estimation += 1
        self._frame_index = frame_index
        self._last_correlation = correlation
        self._correlation_history.append(correlation)

        return TrackingState(
            homography=self._current_homography.copy(),
            edge_image=edge_image,
            frame_index=frame_index,
            frames_since_estimation=self._frames_since_estimation,
            is_estimated=False,
            correlation=correlation,
            tracking_valid=tracking_valid,
        )

    def _prepare_edge(self, edge_image: np.ndarray) -> np.ndarray:
        """
        Prepare an edge image for ECC alignment.
        Converts to float32 and normalises.
        """
        if edge_image.dtype == np.uint8:
            prepared = edge_image.astype(np.float32) / 255.0
        else:
            prepared = edge_image.astype(np.float32)

        # Ensure consistent size
        if prepared.shape != (180, 320):
            prepared = cv2.resize(prepared, (320, 180))

        return prepared

    def reset(self) -> None:
        """Reset the tracker to uninitialised state."""
        self._current_homography = None
        self._previous_edge = None
        self._frames_since_estimation = 0
        self._frame_index = 0
        self._initialised = False
        self._last_correlation = 1.0
        self._correlation_history = []
        self._estimation_frames = []

    def get_stats(self) -> dict:
        """
        Return tracking statistics for analysis and reporting.

        Useful for your evaluation section — shows how often
        re-estimation was triggered and tracking quality over time.
        """
        if not self._correlation_history:
            return {"status": "not started"}

        correlations = np.array(self._correlation_history)

        return {
            "total_frames": len(correlations),
            "estimation_frames": len(self._estimation_frames),
            "estimation_indices": self._estimation_frames.copy(),
            "tracking_ratio": 1.0 - len(self._estimation_frames) / max(len(correlations), 1),
            "mean_correlation": float(np.mean(correlations)),
            "min_correlation": float(np.min(correlations)),
            "frames_below_threshold": int(np.sum(correlations < self.min_correlation)),
        }


if __name__ == "__main__":
    print("=== Camera Pose Tracker Test ===\n")

    # Test 1: Basic lifecycle
    print("1. Testing tracker lifecycle...")
    tracker = CameraPoseTracker(estimation_interval=5)

    assert tracker.needs_estimation(), "Should need estimation when uninitialised"

    # Simulate first estimation
    H_init = np.eye(3, dtype=np.float64)
    edge_init = np.zeros((180, 320), dtype=np.uint8)
    cv2.line(edge_init, (160, 0), (160, 180), 255, 2)
    cv2.line(edge_init, (0, 90), (320, 90), 255, 2)

    state = tracker.set_estimation(H_init, edge_init, frame_index=0)
    print(f"   Frame 0: estimated, correlation={state.correlation:.2f}")
    assert state.is_estimated
    assert not tracker.needs_estimation()

    # Simulate tracking for 4 frames with slightly shifted edges
    for i in range(1, 5):
        edge_shifted = np.zeros((180, 320), dtype=np.uint8)
        cv2.line(edge_shifted, (160 + i, 0), (160 + i, 180), 255, 2)
        cv2.line(edge_shifted, (0, 90), (320, 90), 255, 2)

        state = tracker.track(edge_shifted, frame_index=i)
        print(f"   Frame {i}: tracked, correlation={state.correlation:.2f}, "
              f"valid={state.tracking_valid}")

    # Track one more frame to hit the interval
    edge_shifted = np.zeros((180, 320), dtype=np.uint8)
    cv2.line(edge_shifted, (165, 0), (165, 180), 255, 2)
    cv2.line(edge_shifted, (0, 90), (320, 90), 255, 2)
    state = tracker.track(edge_shifted, frame_index=5)
    print(f"   Frame 5: tracked, correlation={state.correlation:.2f}")

    # After estimation_interval frames tracked, should need re-estimation
    assert tracker.needs_estimation(), "Should need estimation after interval"
    print(f"   Frame 6: needs re-estimation = {tracker.needs_estimation()}")

    # Test 2: Stats
    print("\n2. Testing statistics...")
    stats = tracker.get_stats()
    print(f"   Total frames: {stats['total_frames']}")
    print(f"   Estimation frames: {stats['estimation_frames']}")
    print(f"   Tracking ratio: {stats['tracking_ratio']:.1%}")
    print(f"   Mean correlation: {stats['mean_correlation']:.3f}")

    # Test 3: Reset
    print("\n3. Testing reset...")
    tracker.reset()
    assert tracker.needs_estimation()
    print("   Reset successful, needs estimation again")

    # Test 4: Early re-estimation on quality drop
    print("\n4. Testing early re-estimation trigger...")
    tracker2 = CameraPoseTracker(
        estimation_interval=100,
        min_correlation=0.8,
    )

    # Set initial estimation
    tracker2.set_estimation(H_init, edge_init, 0)

    # Track with a very different edge image (should drop correlation)
    bad_edge = np.zeros((180, 320), dtype=np.uint8)
    cv2.circle(bad_edge, (100, 100), 50, 255, 2)  # Completely different content

    state = tracker2.track(bad_edge, 1)
    print(f"   Correlation after bad frame: {state.correlation:.3f}")
    print(f"   Needs re-estimation: {tracker2.needs_estimation()}")

    print("\n=== All tests passed ===")
    print("\nThe camera pose tracker sits in the pipeline like this:")
    print("  for each frame:")
    print("    if tracker.needs_estimation():")
    print("      H = full_pipeline(frame)  # siamese → database → refine")
    print("      tracker.set_estimation(H, edges, frame_idx)")
    print("    else:")
    print("      state = tracker.track(edges, frame_idx)")
    print("      H = state.homography")