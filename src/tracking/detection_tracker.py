"""
detection_tracker.py — Manage detection-tracking alternation.

Alternates between expensive detection (HOG+SVM / CHT) and cheap
tracking (Lucas-Kanade) across video frames. Detection runs every
N frames; tracking fills the gaps.

Mavrogiannis uses a 2-frame tracking period: detect, track, detect,
track, ... This keeps bounding boxes accurate (detection refreshes
them) while halving the computation cost.

This module also handles:
    - Matching new detections to existing tracks (Hungarian algorithm)
    - Creating new tracks for unmatched detections
    - Removing tracks that have been lost too long
    - Maintaining consistent player IDs across frames
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass

from .lucas_kanade import LucasKanadeTracker, TrackedObject


@dataclass
class FrameResult:
    """Result of processing a single frame."""
    frame_index: int
    is_detection_frame: bool
    tracked_objects: List[TrackedObject]
    num_detected: int
    num_tracked: int
    num_lost: int
    num_new: int


class DetectionTracker:
    """
    Manages detection/tracking alternation for video processing.

    Usage:
        dt = DetectionTracker(detect_interval=2)

        for frame_idx, frame in enumerate(video):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if dt.is_detection_frame(frame_idx):
                # Run expensive detection
                player_dets = player_detector.detect(frame)
                ball_det = ball_detector.detect(frame)

                bboxes = [d.bbox for d in player_dets]
                if ball_det.detected:
                    bboxes.append(ball_det.bbox)

                result = dt.process_detection(gray, bboxes, frame_idx)
            else:
                result = dt.process_tracking(gray, frame_idx)

            # Use result.tracked_objects for downstream processing
    """

    def __init__(
        self,
        detect_interval: int = 2,
        iou_match_threshold: float = 0.3,
        max_lost_frames: int = 5,
        lk_win_size: Tuple[int, int] = (15, 15),
        lk_max_level: int = 3,
    ):
        """
        Args:
            detect_interval: run detection every N frames
            iou_match_threshold: minimum IoU to match detection to track
            max_lost_frames: remove track after this many lost frames
            lk_win_size: Lucas-Kanade window size
            lk_max_level: LK pyramid levels
        """
        self.detect_interval = detect_interval
        self.iou_match_threshold = iou_match_threshold

        self.tracker = LucasKanadeTracker(
            win_size=lk_win_size,
            max_level=lk_max_level,
            max_lost_frames=max_lost_frames,
        )

        self._frame_count = 0
        self._initialised = False

    def is_detection_frame(self, frame_index: int) -> bool:
        """Check if this frame should run detection."""
        if not self._initialised:
            return True
        return frame_index % self.detect_interval == 0

    def process_detection(
        self,
        gray_frame: np.ndarray,
        detections: List[Tuple[int, int, int, int]],
        frame_index: int,
        labels: Optional[List[str]] = None,
    ) -> FrameResult:
        """
        Process a detection frame: match new detections to existing
        tracks, create new tracks, update positions.

        Args:
            gray_frame: grayscale image
            detections: list of (x, y, w, h) bounding boxes
            frame_index: current frame number
            labels: optional labels per detection

        Returns:
            FrameResult: processing result
        """
        num_new = 0

        if not self._initialised:
            # First frame: initialise all tracks
            self.tracker.initialise(gray_frame, detections, labels)
            self._initialised = True
            num_new = len(detections)

            return FrameResult(
                frame_index=frame_index,
                is_detection_frame=True,
                tracked_objects=self.tracker.get_active_tracks(),
                num_detected=len(detections),
                num_tracked=0,
                num_lost=0,
                num_new=num_new,
            )

        # First, track existing objects to current frame
        tracked = self.tracker.update(gray_frame)

        # Match detections to existing tracks using IoU
        matched_det_indices = set()
        matched_track_indices = set()

        if len(tracked) > 0 and len(detections) > 0:
            # Compute IoU matrix between detections and tracks
            iou_matrix = np.zeros((len(detections), len(tracked)))

            for d_idx, (dx, dy, dw, dh) in enumerate(detections):
                for t_idx, track in enumerate(tracked):
                    tx, ty, tw, th = track.bbox
                    iou_matrix[d_idx, t_idx] = self._compute_iou(
                        (dx, dy, dw, dh), (tx, ty, tw, th)
                    )

            # Greedy matching: best IoU first
            while True:
                if iou_matrix.size == 0:
                    break

                max_iou = np.max(iou_matrix)
                if max_iou < self.iou_match_threshold:
                    break

                d_idx, t_idx = np.unravel_index(
                    np.argmax(iou_matrix), iou_matrix.shape
                )

                # Update the matched track with the new detection position
                dx, dy, dw, dh = detections[d_idx]
                tracked[t_idx].x = dx + dw / 2.0
                tracked[t_idx].y = dy + dh / 2.0
                tracked[t_idx].w = dw
                tracked[t_idx].h = dh
                tracked[t_idx].lost_frames = 0

                matched_det_indices.add(d_idx)
                matched_track_indices.add(t_idx)

                # Remove matched row and column
                iou_matrix[d_idx, :] = -1
                iou_matrix[:, t_idx] = -1

        # Create new tracks for unmatched detections
        for d_idx, (dx, dy, dw, dh) in enumerate(detections):
            if d_idx not in matched_det_indices:
                label = labels[d_idx] if labels and d_idx < len(labels) else "player"
                self.tracker.add_detection(dx, dy, dw, dh, label)
                num_new += 1

        self._frame_count += 1

        active = self.tracker.get_active_tracks()
        lost = self.tracker.get_lost_tracks()

        return FrameResult(
            frame_index=frame_index,
            is_detection_frame=True,
            tracked_objects=active,
            num_detected=len(detections),
            num_tracked=len(matched_track_indices),
            num_lost=len(lost),
            num_new=num_new,
        )

    def process_tracking(
        self,
        gray_frame: np.ndarray,
        frame_index: int,
    ) -> FrameResult:
        """
        Process a tracking-only frame (no detection).

        Args:
            gray_frame: grayscale image
            frame_index: current frame number

        Returns:
            FrameResult: processing result
        """
        if not self._initialised:
            return FrameResult(
                frame_index=frame_index,
                is_detection_frame=False,
                tracked_objects=[],
                num_detected=0, num_tracked=0, num_lost=0, num_new=0,
            )

        tracked = self.tracker.update(gray_frame)
        self._frame_count += 1

        active = self.tracker.get_active_tracks()
        lost = self.tracker.get_lost_tracks()

        return FrameResult(
            frame_index=frame_index,
            is_detection_frame=False,
            tracked_objects=active,
            num_detected=0,
            num_tracked=len(active),
            num_lost=len(lost),
            num_new=0,
        )

    def get_player_tracks(self) -> List[TrackedObject]:
        """Return only player tracks (exclude ball)."""
        return [t for t in self.tracker.get_active_tracks()
                if t.label == "player"]

    def get_ball_track(self) -> Optional[TrackedObject]:
        """Return the ball track if it exists."""
        balls = [t for t in self.tracker.get_active_tracks()
                 if t.label == "ball"]
        return balls[0] if balls else None

    def reset(self) -> None:
        """Reset all state."""
        self.tracker.reset()
        self._frame_count = 0
        self._initialised = False

    def _compute_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
    ) -> float:
        """Compute IoU between two (x, y, w, h) boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi = max(x1, x2)
        yi = max(y1, y2)
        xf = min(x1 + w1, x2 + w2)
        yf = min(y1 + h1, y2 + h2)

        if xi >= xf or yi >= yf:
            return 0.0

        intersection = (xf - xi) * (yf - yi)
        union = w1 * h1 + w2 * h2 - intersection

        return intersection / max(union, 1e-8)

    def get_stats(self) -> dict:
        """Return tracking statistics."""
        active = self.tracker.get_active_tracks()
        lost = self.tracker.get_lost_tracks()

        return {
            "frames_processed": self._frame_count,
            "active_tracks": len(active),
            "lost_tracks": len(lost),
            "total_ids_assigned": self.tracker._next_id,
            "players": len([t for t in active if t.label == "player"]),
            "balls": len([t for t in active if t.label == "ball"]),
        }


if __name__ == "__main__":
    print("=== Detection Tracker Test ===\n")
    import cv2

    dt = DetectionTracker(detect_interval=3)

    print("1. Simulating 12-frame video (detect every 3 frames)...")

    for i in range(12):
        # Create frame with moving objects
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:, :] = [34, 139, 34]

        # Player moves right
        px = 100 + i * 8
        cv2.rectangle(frame, (px, 150), (px + 30, 250), (0, 0, 200), -1)

        # Second player moves left
        px2 = 500 - i * 5
        cv2.rectangle(frame, (px2, 180), (px2 + 30, 280), (200, 0, 0), -1)

        # Ball
        bx = 300 + i * 3
        cv2.circle(frame, (bx, 200), 8, (255, 255, 255), -1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if dt.is_detection_frame(i):
            detections = [
                (px, 150, 30, 100),
                (px2, 180, 30, 100),
                (bx - 8, 192, 16, 16),
            ]
            labels = ["player", "player", "ball"]
            result = dt.process_detection(gray, detections, i, labels)
            tag = "DET"
        else:
            result = dt.process_tracking(gray, i)
            tag = "trk"

        print(f"   Frame {i:2d} [{tag}]: "
              f"active={len(result.tracked_objects)} "
              f"new={result.num_new} lost={result.num_lost}")

    # Stats
    print(f"\n2. Statistics:")
    stats = dt.get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")

    # Player/ball separation
    print(f"\n3. Track types:")
    print(f"   Players: {len(dt.get_player_tracks())}")
    ball = dt.get_ball_track()
    print(f"   Ball: {'at ({:.0f}, {:.0f})'.format(ball.x, ball.y) if ball else 'not tracked'}")

    print("\n=== Tests complete ===")