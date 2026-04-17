"""
lucas_kanade.py — Pyramidal Lucas-Kanade optical flow tracker.

Tracks player bounding box centres and ball position between
detection frames using OpenCV's pyramidal Lucas-Kanade implementation.

Mavrogiannis tracks the centroid of each bounding box because
"object detection produces pretty accurate bounding boxes; thus,
it is almost certain that the centroid belongs to a player's outfit
or the center of the ball" (Section 3.5).

The pyramidal version handles large motions by tracking from
coarse to fine resolution levels, catching movements that would
escape a single-scale local window.

"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TrackedObject:
    """A single tracked object (player or ball)."""
    object_id: int
    x: float                # Current centre x
    y: float                # Current centre y
    w: int                  # Bounding box width (from last detection)
    h: int                  # Bounding box height (from last detection)
    lost_frames: int = 0    # Consecutive frames without successful tracking
    label: str = "player"   # "player" or "ball"

    @property
    def centre(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def foot_position(self) -> Tuple[float, float]:
        return (self.x, self.y + self.h / 2)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (int(self.x - self.w / 2), int(self.y - self.h / 2),
                self.w, self.h)


class LucasKanadeTracker:
    """
    Pyramidal Lucas-Kanade optical flow tracker.

    Tracks point locations between consecutive frames. Each tracked
    object is represented by its bounding box centroid.

    Usage:
        tracker = LucasKanadeTracker()

        # Initialise with detections from first frame
        tracker.initialise(prev_gray, detections)

        # Track to next frame
        tracked = tracker.update(curr_gray)
    """

    def __init__(
        self,
        win_size: Tuple[int, int] = (15, 15),
        max_level: int = 3,
        max_lost_frames: int = 5,
        min_displacement: float = 0.5,
        max_displacement: float = 100.0,
    ):
        """
        Args:
            win_size: LK window size (larger = handles faster motion)
            max_level: pyramid levels (higher = handles larger motion)
            max_lost_frames: remove track after this many failed frames
            min_displacement: ignore movements smaller than this (noise)
            max_displacement: reject movements larger than this (error)
        """
        self.win_size = win_size
        self.max_level = max_level
        self.max_lost_frames = max_lost_frames
        self.min_displacement = min_displacement
        self.max_displacement = max_displacement

        # LK parameters for OpenCV
        self.lk_params = dict(
            winSize=win_size,
            maxLevel=max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30, 0.01
            ),
        )

        # State
        self._prev_gray = None
        self._tracked_objects: List[TrackedObject] = []
        self._next_id = 0

    def initialise(
        self,
        gray_frame: np.ndarray,
        detections: List[Tuple[int, int, int, int]],
        labels: Optional[List[str]] = None,
    ) -> List[TrackedObject]:
        """
        Initialise tracker with detections from a detection frame.

        Args:
            gray_frame: grayscale image
            detections: list of (x, y, w, h) bounding boxes
            labels: optional list of labels ("player" or "ball")

        Returns:
            list: TrackedObject for each detection
        """
        self._prev_gray = gray_frame.copy()
        self._tracked_objects = []

        for i, (x, y, w, h) in enumerate(detections):
            cx = x + w / 2.0
            cy = y + h / 2.0
            label = labels[i] if labels else "player"

            obj = TrackedObject(
                object_id=self._next_id,
                x=cx, y=cy, w=w, h=h,
                lost_frames=0,
                label=label,
            )
            self._tracked_objects.append(obj)
            self._next_id += 1

        return self._tracked_objects.copy()

    def update(self, gray_frame: np.ndarray) -> List[TrackedObject]:
        """
        Track all objects from previous frame to current frame.

        Uses cv2.calcOpticalFlowPyrLK to find where each tracked
        point moved to in the new frame.

        Args:
            gray_frame: current grayscale frame

        Returns:
            list: updated TrackedObject list (lost tracks removed)
        """
        if self._prev_gray is None or len(self._tracked_objects) == 0:
            self._prev_gray = gray_frame.copy()
            return []

        # Build points array from tracked objects
        prev_points = np.array(
            [[obj.x, obj.y] for obj in self._tracked_objects],
            dtype=np.float32
        ).reshape(-1, 1, 2)

        # Run Lucas-Kanade
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray_frame,
            prev_points, None,
            **self.lk_params
        )

        # Update tracked objects
        updated = []
        for i, obj in enumerate(self._tracked_objects):
            if status[i][0] == 1:
                # Successfully tracked
                new_x = float(next_points[i][0][0])
                new_y = float(next_points[i][0][1])

                # Check displacement
                dx = new_x - obj.x
                dy = new_y - obj.y
                displacement = np.sqrt(dx**2 + dy**2)

                if displacement > self.max_displacement:
                    # Too large — likely an error
                    obj.lost_frames += 1
                elif displacement < self.min_displacement:
                    # Too small — treat as stationary (noise)
                    obj.lost_frames = 0
                else:
                    # Valid movement
                    obj.x = new_x
                    obj.y = new_y
                    obj.lost_frames = 0

                # Check bounds
                h, w = gray_frame.shape[:2]
                if 0 <= obj.x < w and 0 <= obj.y < h:
                    if obj.lost_frames <= self.max_lost_frames:
                        updated.append(obj)
                else:
                    # Tracked off screen
                    pass
            else:
                # Tracking failed
                obj.lost_frames += 1
                if obj.lost_frames <= self.max_lost_frames:
                    updated.append(obj)

        self._tracked_objects = updated
        self._prev_gray = gray_frame.copy()

        return self._tracked_objects.copy()

    def add_detection(
        self,
        x: int, y: int, w: int, h: int,
        label: str = "player",
    ) -> TrackedObject:
        """Add a new detection to tracking (e.g. when a new player appears)."""
        obj = TrackedObject(
            object_id=self._next_id,
            x=x + w / 2.0, y=y + h / 2.0,
            w=w, h=h,
            lost_frames=0, label=label,
        )
        self._tracked_objects.append(obj)
        self._next_id += 1
        return obj

    def get_active_tracks(self) -> List[TrackedObject]:
        """Return currently active tracks."""
        return [obj for obj in self._tracked_objects if obj.lost_frames == 0]

    def get_lost_tracks(self) -> List[TrackedObject]:
        """Return tracks that have been lost but not yet removed."""
        return [obj for obj in self._tracked_objects if obj.lost_frames > 0]

    def reset(self) -> None:
        """Reset all tracking state."""
        self._prev_gray = None
        self._tracked_objects = []
        self._next_id = 0


if __name__ == "__main__":
    print("=== Lucas-Kanade Tracker Test ===\n")
    import cv2

    tracker = LucasKanadeTracker()

    # Create synthetic frames with a moving object
    print("1. Simulating 10-frame tracking sequence...")

    for i in range(10):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:, :] = [34, 139, 34]

        # Moving player (shifts right by 5px per frame)
        px = 100 + i * 5
        cv2.rectangle(frame, (px, 150), (px + 30, 250), (0, 0, 200), -1)

        # Stationary player
        cv2.rectangle(frame, (400, 180), (430, 280), (200, 0, 0), -1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if i == 0:
            detections = [(px, 150, 30, 100), (400, 180, 30, 100)]
            tracker.initialise(gray, detections)
            print(f"   Frame {i}: initialised {len(detections)} tracks")
        else:
            tracked = tracker.update(gray)
            active = tracker.get_active_tracks()
            print(f"   Frame {i}: {len(active)} active tracks, "
                  f"positions: {[(int(t.x), int(t.y)) for t in active]}")

    # Test stats
    print(f"\n2. Final state:")
    print(f"   Active: {len(tracker.get_active_tracks())}")
    print(f"   Lost: {len(tracker.get_lost_tracks())}")

    # Test reset
    print("\n3. Testing reset...")
    tracker.reset()
    print(f"   Tracks after reset: {len(tracker.get_active_tracks())}")

    print("\n=== Tests complete ===")