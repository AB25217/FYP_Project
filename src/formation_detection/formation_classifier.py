"""
formation_classifier.py — Main formation detection logic.

Combines all formation detection components into a single interface:
    1. Filter out goalkeepers
    2. Smooth positions over time
    3. Cluster into defensive lines
    4. Match against formation templates
    5. Output: formation label + confidence per team
Usage:
    classifier = FormationClassifier()

    # Feed positions each frame
    for frame_positions, frame_ids, frame_teams in pipeline_output:
        result = classifier.update(frame_positions, frame_ids, frame_teams)

        if result is not None:
            print(f"Team A: {result.team_a_formation}")
            print(f"Team B: {result.team_b_formation}")
"""

import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass

from .goalkeeper_filter import GoalkeeperFilter
from .line_clustering import LineClustering
from .template_matching import TemplateMatcher
from .temporal_smoother import TemporalSmoother


@dataclass
class FormationResult:
    """Formation detection result for both teams."""
    team_a_formation: str           # e.g. "4-3-3"
    team_a_confidence: float        # 0-1
    team_a_line_counts: List[int]   # e.g. [4, 3, 3]
    team_b_formation: str
    team_b_confidence: float
    team_b_line_counts: List[int]
    num_frames_used: int            # Frames in the smoothing window
    formation_changed: bool         # Whether a change was detected


class FormationClassifier:
    """
    End-to-end formation detection from player positions.

    Workflow per update:
        Positions → GK filter → Temporal smooth → Line cluster → Template match
                                                                       ↓
                                                            Formation label + confidence
    """

    def __init__(
        self,
        smoothing_window: int = 750,
        update_interval: int = 125,
        change_threshold: float = 10.0,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ):
        """
        Args:
            smoothing_window: frames for temporal averaging (750 = 30s at 25fps)
            update_interval: re-classify formation every N frames (125 = 5s)
            change_threshold: displacement threshold for formation change
            pitch_length: FIFA pitch length
            pitch_width: FIFA pitch width
        """
        self.update_interval = update_interval
        self.change_threshold = change_threshold

        # Components
        self.gk_filter = GoalkeeperFilter(pitch_length=pitch_length)
        self.line_clusterer = LineClustering()
        self.template_matcher = TemplateMatcher(
            pitch_length=pitch_length, pitch_width=pitch_width
        )
        self.smoother_a = TemporalSmoother(window_size=smoothing_window)
        self.smoother_b = TemporalSmoother(window_size=smoothing_window)

        # State
        self._frame_count = 0
        self._last_result = None

    def update(
        self,
        positions: np.ndarray,
        player_ids: List[int],
        teams: List[str],
    ) -> Optional[FormationResult]:
        """
        Process one frame of player positions.

        Returns a FormationResult every update_interval frames,
        or None on intermediate frames.

        Args:
            positions: (N, 2) pitch coordinates for all players
            player_ids: list of player IDs
            teams: list of team labels ("team_a", "team_b", "referee")

        Returns:
            FormationResult or None
        """
        # Separate by team
        team_a_mask = np.array([t == "team_a" for t in teams])
        team_b_mask = np.array([t == "team_b" for t in teams])

        if np.sum(team_a_mask) > 0:
            self.smoother_a.add_frame(
                positions[team_a_mask],
                [pid for pid, m in zip(player_ids, team_a_mask) if m]
            )

        if np.sum(team_b_mask) > 0:
            self.smoother_b.add_frame(
                positions[team_b_mask],
                [pid for pid, m in zip(player_ids, team_b_mask) if m]
            )

        self._frame_count += 1

        # Only classify every update_interval frames
        if self._frame_count % self.update_interval != 0:
            return self._last_result

        # Check if we have enough data
        if not self.smoother_a.is_ready() and not self.smoother_b.is_ready():
            return None

        # Detect formation change
        changed_a = self.smoother_a.detect_formation_change(self.change_threshold)
        changed_b = self.smoother_b.detect_formation_change(self.change_threshold)

        # Classify each team
        team_a_form, team_a_conf, team_a_lines = self._classify_team(
            self.smoother_a, "team_a"
        )
        team_b_form, team_b_conf, team_b_lines = self._classify_team(
            self.smoother_b, "team_b"
        )

        result = FormationResult(
            team_a_formation=team_a_form,
            team_a_confidence=team_a_conf,
            team_a_line_counts=team_a_lines,
            team_b_formation=team_b_form,
            team_b_confidence=team_b_conf,
            team_b_line_counts=team_b_lines,
            num_frames_used=self.smoother_a.buffer_size,
            formation_changed=changed_a or changed_b,
        )

        self._last_result = result
        return result

    def _classify_team(
        self,
        smoother: TemporalSmoother,
        team: str,
    ) -> tuple:
        """
        Classify a single team's formation.

        Returns:
            tuple: (formation_name, confidence, line_counts)
        """
        smoothed = smoother.get_smoothed()

        if len(smoothed.positions) < 5:
            return ("unknown", 0.0, [])

        # Filter goalkeeper
        filtered = self.gk_filter.filter_team(
            smoothed.positions,
            smoothed.player_ids,
            team
        )

        outfield = filtered.outfield_positions

        if len(outfield) < 5:
            return ("unknown", 0.0, [])

        # Line clustering
        line_result = self.line_clusterer.cluster(outfield, team)

        # Template matching
        match_result = self.template_matcher.match(outfield, team)

        # Use template match as primary, line clustering as backup
        if match_result.confidence > 0.3:
            formation = match_result.best_formation
            confidence = match_result.confidence
        else:
            formation = line_result.formation_string
            confidence = max(0.1, line_result.silhouette)

        return (formation, confidence, line_result.line_counts)

    def classify_once(
        self,
        positions: np.ndarray,
        player_ids: List[int],
        team: str,
    ) -> tuple:
        """
        Classify formation from a single snapshot (no temporal smoothing).

        Useful for testing or when you have ground truth positions.

        Args:
            positions: (N, 2) pitch positions for one team
            player_ids: player IDs
            team: "team_a" or "team_b"

        Returns:
            tuple: (formation_name, confidence, line_counts)
        """
        filtered = self.gk_filter.filter_team(positions, player_ids, team)
        outfield = filtered.outfield_positions

        if len(outfield) < 5:
            return ("unknown", 0.0, [])

        line_result = self.line_clusterer.cluster(outfield, team)
        match_result = self.template_matcher.match(outfield, team)

        if match_result.confidence > 0.3:
            return (match_result.best_formation, match_result.confidence,
                    line_result.line_counts)
        else:
            return (line_result.formation_string, max(0.1, line_result.silhouette),
                    line_result.line_counts)

    def get_last_result(self) -> Optional[FormationResult]:
        """Return the most recent formation result."""
        return self._last_result

    def reset(self) -> None:
        """Reset all state."""
        self.smoother_a.clear()
        self.smoother_b.clear()
        self._frame_count = 0
        self._last_result = None


if __name__ == "__main__":
    print("=== Formation Classifier Test ===\n")

    classifier = FormationClassifier()

    # Test 1: Single snapshot classification
    print("1. Testing single snapshot (4-3-3)...")
    positions_433 = np.array([
        [3, 34],        # GK
        [25, 10], [25, 27], [25, 41], [25, 58],  # Defence
        [48, 17], [48, 34], [48, 51],              # Midfield
        [78, 14], [78, 34], [78, 54],              # Attack
    ], dtype=np.float64)
    ids_433 = list(range(11))

    form, conf, lines = classifier.classify_once(positions_433, ids_433, "team_a")
    print(f"   Formation: {form}")
    print(f"   Confidence: {conf:.2f}")
    print(f"   Lines: {lines}")

    # Test 2: Single snapshot (4-4-2)
    print("\n2. Testing single snapshot (4-4-2)...")
    positions_442 = np.array([
        [3, 34],
        [22, 10], [22, 27], [22, 41], [22, 58],
        [50, 10], [50, 27], [50, 41], [50, 58],
        [75, 25], [75, 43],
    ], dtype=np.float64)

    form2, conf2, lines2 = classifier.classify_once(positions_442, list(range(11)), "team_a")
    print(f"   Formation: {form2}")
    print(f"   Confidence: {conf2:.2f}")
    print(f"   Lines: {lines2}")

    # Test 3: Temporal update
    print("\n3. Testing temporal updates (50 frames)...")
    classifier2 = FormationClassifier(
        smoothing_window=20, update_interval=10
    )

    np.random.seed(42)
    for i in range(50):
        noise = np.random.randn(11, 2) * 2.0
        frame_pos = positions_433 + noise
        frame_ids = list(range(11))
        frame_teams = ["team_a"] * 11

        result = classifier2.update(frame_pos, frame_ids, frame_teams)

        if result is not None:
            print(f"   Frame {i}: Team A = {result.team_a_formation} "
                  f"(conf={result.team_a_confidence:.2f})")

    print("\n=== Tests complete ===")