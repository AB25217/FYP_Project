"""
temporal_smoother.py — Average player positions over a time window.

Formations are not instantaneous — a team holds a shape over minutes,
not frames. Individual frames capture momentary movements (sprinting
for a ball, pressing, etc.) that don't reflect the underlying formation.

This module:
    1. Accumulates player positions over a rolling window (e.g. 30 seconds)
    2. Computes mean positions per player/team within the window
    3. Detects when the formation changes significantly

The smoothed positions are what gets passed to the formation classifier,
not raw per-frame positions.
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class SmoothedPositions:
    """Temporally smoothed player positions."""
    positions: np.ndarray       # (N, 2) mean positions
    player_ids: List[int]       # Player IDs corresponding to positions
    num_frames: int             # Number of frames in the window
    variance: float             # Mean positional variance (movement indicator)


class TemporalSmoother:
    """
    Smooth player positions over a rolling time window.

    Usage:
        smoother = TemporalSmoother(window_size=750)  # 30s at 25fps

        for frame_positions, frame_ids in video_output:
            smoother.add_frame(frame_positions, frame_ids)

            if smoother.is_ready():
                smoothed = smoother.get_smoothed()
                # Use smoothed.positions for formation detection
    """

    def __init__(
        self,
        window_size: int = 750,
        min_appearances: float = 0.3,
    ):
        """
        Args:
            window_size: number of frames in the rolling window
                        (e.g. 750 = 30 seconds at 25fps)
            min_appearances: minimum fraction of frames a player must
                           appear in to be included in smoothed output
        """
        self.window_size = window_size
        self.min_appearances = min_appearances

        # Store per-frame data: list of (positions, ids) tuples
        self._buffer: deque = deque(maxlen=window_size)

    def add_frame(
        self,
        positions: np.ndarray,
        player_ids: List[int],
    ) -> None:
        """
        Add a frame's player positions to the buffer.

        Args:
            positions: (N, 2) pitch coordinates for this frame
            player_ids: list of player IDs for this frame
        """
        self._buffer.append((positions.copy(), list(player_ids)))

    def is_ready(self) -> bool:
        """Check if enough frames have been accumulated."""
        return len(self._buffer) >= self.window_size * 0.1

    def get_smoothed(
        self,
        team_ids: Optional[List[int]] = None,
    ) -> SmoothedPositions:
        """
        Compute smoothed positions from the buffer.

        Args:
            team_ids: if provided, only include these player IDs

        Returns:
            SmoothedPositions: averaged positions
        """
        if len(self._buffer) == 0:
            return SmoothedPositions(
                positions=np.array([]).reshape(0, 2),
                player_ids=[], num_frames=0, variance=0.0
            )

        # Collect all positions per player ID
        player_positions: Dict[int, List[np.ndarray]] = {}

        for positions, ids in self._buffer:
            for i, pid in enumerate(ids):
                if team_ids is not None and pid not in team_ids:
                    continue
                if i < len(positions):
                    if pid not in player_positions:
                        player_positions[pid] = []
                    player_positions[pid].append(positions[i])

        # Filter by minimum appearances
        min_count = max(1, int(len(self._buffer) * self.min_appearances))

        valid_ids = []
        mean_positions = []
        variances = []

        for pid, pos_list in player_positions.items():
            if len(pos_list) >= min_count:
                pos_array = np.array(pos_list)
                mean_pos = np.mean(pos_array, axis=0)
                var = np.mean(np.var(pos_array, axis=0))

                valid_ids.append(pid)
                mean_positions.append(mean_pos)
                variances.append(var)

        if not valid_ids:
            return SmoothedPositions(
                positions=np.array([]).reshape(0, 2),
                player_ids=[], num_frames=len(self._buffer), variance=0.0
            )

        positions_array = np.array(mean_positions)
        mean_variance = float(np.mean(variances))

        return SmoothedPositions(
            positions=positions_array,
            player_ids=valid_ids,
            num_frames=len(self._buffer),
            variance=mean_variance,
        )

    def detect_formation_change(
        self,
        threshold: float = 10.0,
    ) -> bool:
        """
        Detect if the formation has changed significantly.

        Compares the mean positions in the first half of the buffer
        to the second half. If the average displacement exceeds the
        threshold, a formation change is likely.

        Args:
            threshold: displacement threshold in metres

        Returns:
            bool: True if a formation change is detected
        """
        if len(self._buffer) < 20:
            return False

        buffer_list = list(self._buffer)
        mid = len(buffer_list) // 2

        first_half = buffer_list[:mid]
        second_half = buffer_list[mid:]

        # Get mean positions for each half
        first_means = self._compute_means(first_half)
        second_means = self._compute_means(second_half)

        # Find common player IDs
        common_ids = set(first_means.keys()) & set(second_means.keys())

        if not common_ids:
            return False

        # Compute mean displacement
        displacements = []
        for pid in common_ids:
            d = np.sqrt(np.sum((first_means[pid] - second_means[pid]) ** 2))
            displacements.append(d)

        mean_displacement = np.mean(displacements)
        return mean_displacement > threshold

    def _compute_means(
        self,
        frames: list,
    ) -> Dict[int, np.ndarray]:
        """Compute mean position per player across a list of frames."""
        positions_per_id: Dict[int, List[np.ndarray]] = {}

        for positions, ids in frames:
            for i, pid in enumerate(ids):
                if i < len(positions):
                    if pid not in positions_per_id:
                        positions_per_id[pid] = []
                    positions_per_id[pid].append(positions[i])

        means = {}
        for pid, pos_list in positions_per_id.items():
            means[pid] = np.mean(pos_list, axis=0)

        return means

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)


if __name__ == "__main__":
    print("=== Temporal Smoother Test ===\n")

    smoother = TemporalSmoother(window_size=20)

    # Simulate 25 frames of a 4-3-3 with small random movements
    print("1. Accumulating 25 frames...")
    np.random.seed(42)

    base_positions = np.array([
        [25, 10], [25, 27], [25, 41], [25, 58],
        [48, 17], [48, 34], [48, 51],
        [78, 14], [78, 34], [78, 54],
    ], dtype=np.float64)
    player_ids = list(range(10))

    for i in range(25):
        noise = np.random.randn(10, 2) * 2.0
        frame_pos = base_positions + noise
        smoother.add_frame(frame_pos, player_ids)

    print(f"   Buffer size: {smoother.buffer_size}")
    print(f"   Ready: {smoother.is_ready()}")

    # Get smoothed positions
    smoothed = smoother.get_smoothed()
    print(f"\n2. Smoothed positions:")
    print(f"   Players: {len(smoothed.player_ids)}")
    print(f"   Variance: {smoothed.variance:.2f} (low = stable formation)")

    # Compare smoothed vs base
    for i, pid in enumerate(smoothed.player_ids):
        base = base_positions[pid]
        sm = smoothed.positions[i]
        error = np.sqrt(np.sum((base - sm)**2))
        print(f"   Player {pid}: base=({base[0]:.0f},{base[1]:.0f}) "
              f"smoothed=({sm[0]:.1f},{sm[1]:.1f}) error={error:.1f}m")

    # Test formation change detection
    print(f"\n3. Formation change: {smoother.detect_formation_change()}")

    print("\n=== Tests complete ===")