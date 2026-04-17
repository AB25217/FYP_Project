"""
template_matching.py — Match detected line structure to known formations.

Defines standard football formation templates (4-4-2, 4-3-3, 3-5-2, etc.)
with idealised player positions. Compares the detected player positions
against each template and finds the best match using the Hungarian
algorithm for optimal assignment.

The cost is the total Euclidean distance between detected positions
and template positions after optimal assignment. The template with
the lowest cost is the detected formation.

Templates are defined in normalised coordinates (0-1) relative to
the team's half, then scaled to actual pitch dimensions.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MatchResult:
    """Result of matching against a single formation template."""
    formation_name: str
    total_cost: float           # Sum of distances after optimal assignment
    mean_cost: float            # Average distance per player
    assignment: List[Tuple[int, int]]  # (player_idx, template_idx) pairs


@dataclass
class FormationMatch:
    """Best formation match result."""
    best_formation: str         # e.g. "4-3-3"
    confidence: float           # 0-1 confidence score
    all_matches: List[MatchResult]  # Sorted by cost (best first)


class TemplateMatcher:
    """
    Match detected player positions to known formation templates.

    Usage:
        matcher = TemplateMatcher()
        result = matcher.match(outfield_positions, team="team_a")
        print(result.best_formation)  # "4-3-3"
        print(result.confidence)      # 0.85
    """

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

        # Define formation templates
        # Positions are (x_norm, y_norm) where:
        #   x_norm: 0 = own goal line, 1 = opponent goal line
        #   y_norm: 0 = bottom touchline, 1 = top touchline
        # Each template has 10 outfield players
        self.templates = self._define_templates()

    def _define_templates(self) -> Dict[str, np.ndarray]:
        """Define standard formation templates with normalised positions."""
        templates = {}

        # 4-4-2
        templates["4-4-2"] = np.array([
            # Defence
            [0.25, 0.15], [0.25, 0.38], [0.25, 0.62], [0.25, 0.85],
            # Midfield
            [0.50, 0.15], [0.50, 0.38], [0.50, 0.62], [0.50, 0.85],
            # Attack
            [0.75, 0.35], [0.75, 0.65],
        ])

        # 4-3-3
        templates["4-3-3"] = np.array([
            [0.25, 0.15], [0.25, 0.38], [0.25, 0.62], [0.25, 0.85],
            [0.45, 0.25], [0.45, 0.50], [0.45, 0.75],
            [0.75, 0.20], [0.75, 0.50], [0.75, 0.80],
        ])

        # 4-2-3-1
        templates["4-2-3-1"] = np.array([
            [0.25, 0.15], [0.25, 0.38], [0.25, 0.62], [0.25, 0.85],
            [0.40, 0.35], [0.40, 0.65],
            [0.60, 0.20], [0.60, 0.50], [0.60, 0.80],
            [0.80, 0.50],
        ])

        # 3-5-2
        templates["3-5-2"] = np.array([
            [0.25, 0.25], [0.25, 0.50], [0.25, 0.75],
            [0.45, 0.10], [0.45, 0.30], [0.45, 0.50], [0.45, 0.70], [0.45, 0.90],
            [0.75, 0.35], [0.75, 0.65],
        ])

        # 3-4-3
        templates["3-4-3"] = np.array([
            [0.25, 0.25], [0.25, 0.50], [0.25, 0.75],
            [0.50, 0.15], [0.50, 0.38], [0.50, 0.62], [0.50, 0.85],
            [0.75, 0.20], [0.75, 0.50], [0.75, 0.80],
        ])

        # 4-1-4-1
        templates["4-1-4-1"] = np.array([
            [0.20, 0.15], [0.20, 0.38], [0.20, 0.62], [0.20, 0.85],
            [0.35, 0.50],
            [0.55, 0.15], [0.55, 0.38], [0.55, 0.62], [0.55, 0.85],
            [0.80, 0.50],
        ])

        # 5-3-2
        templates["5-3-2"] = np.array([
            [0.20, 0.10], [0.20, 0.30], [0.20, 0.50], [0.20, 0.70], [0.20, 0.90],
            [0.45, 0.25], [0.45, 0.50], [0.45, 0.75],
            [0.75, 0.35], [0.75, 0.65],
        ])

        # 4-5-1
        templates["4-5-1"] = np.array([
            [0.25, 0.15], [0.25, 0.38], [0.25, 0.62], [0.25, 0.85],
            [0.50, 0.10], [0.50, 0.30], [0.50, 0.50], [0.50, 0.70], [0.50, 0.90],
            [0.80, 0.50],
        ])

        return templates

    def match(
        self,
        positions: np.ndarray,
        team: str = "team_a",
    ) -> FormationMatch:
        """
        Match detected positions against all formation templates.

        Args:
            positions: (N, 2) outfield player positions in pitch coords
            team: "team_a" (defends x=0) or "team_b" (defends x=105)

        Returns:
            FormationMatch: best formation and all match scores
        """
        if len(positions) == 0:
            return FormationMatch(
                best_formation="unknown",
                confidence=0.0,
                all_matches=[],
            )

        # Normalise positions to [0,1] relative to team's attacking direction
        norm_positions = self._normalise_positions(positions, team)

        all_matches = []

        for name, template in self.templates.items():
            # Only compare if player count matches (or is close)
            if abs(len(positions) - len(template)) > 2:
                continue

            result = self._match_template(norm_positions, template, name)
            all_matches.append(result)

        if not all_matches:
            return FormationMatch("unknown", 0.0, [])

        # Sort by total cost (best match first)
        all_matches.sort(key=lambda r: r.mean_cost)

        # Compute confidence: ratio between best and second-best
        best = all_matches[0]
        if len(all_matches) >= 2:
            second = all_matches[1]
            if second.mean_cost > 0:
                ratio = 1.0 - (best.mean_cost / second.mean_cost)
                confidence = max(0.0, min(ratio * 2, 1.0))
            else:
                confidence = 0.5
        else:
            confidence = 0.5

        return FormationMatch(
            best_formation=best.formation_name,
            confidence=confidence,
            all_matches=all_matches,
        )

    def _normalise_positions(
        self,
        positions: np.ndarray,
        team: str,
    ) -> np.ndarray:
        """
        Normalise pitch positions to [0,1] range.

        For team_a (defends x=0): x_norm = x / half_pitch
        For team_b (defends x=105): x_norm = (105-x) / half_pitch

        y_norm = y / pitch_width for both teams
        """
        norm = np.zeros_like(positions)

        half_pitch = self.pitch_length / 2.0

        if team == "team_a":
            norm[:, 0] = positions[:, 0] / self.pitch_length
        else:
            norm[:, 0] = (self.pitch_length - positions[:, 0]) / self.pitch_length

        norm[:, 1] = positions[:, 1] / self.pitch_width

        # Clip to [0, 1]
        norm = np.clip(norm, 0.0, 1.0)

        return norm

    def _match_template(
        self,
        positions: np.ndarray,
        template: np.ndarray,
        name: str,
    ) -> MatchResult:
        """
        Match positions to a template using the Hungarian algorithm.

        Handles cases where the number of detected players differs
        from the template size by padding with high-cost dummy entries.
        """
        n_pos = len(positions)
        n_temp = len(template)

        # Build cost matrix (Euclidean distance)
        max_size = max(n_pos, n_temp)
        cost_matrix = np.full((max_size, max_size), fill_value=10.0)

        for i in range(n_pos):
            for j in range(n_temp):
                dist = np.sqrt(
                    (positions[i, 0] - template[j, 0]) ** 2 +
                    (positions[i, 1] - template[j, 1]) ** 2
                )
                cost_matrix[i, j] = dist

        # Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Compute total cost (only for real pairs)
        total_cost = 0.0
        assignment = []
        real_pairs = 0

        for r, c in zip(row_ind, col_ind):
            if r < n_pos and c < n_temp:
                total_cost += cost_matrix[r, c]
                assignment.append((r, c))
                real_pairs += 1

        mean_cost = total_cost / max(real_pairs, 1)

        # Penalty for mismatched player count
        count_diff = abs(n_pos - n_temp)
        mean_cost += count_diff * 0.1

        return MatchResult(
            formation_name=name,
            total_cost=total_cost,
            mean_cost=mean_cost,
            assignment=assignment,
        )

    def get_template_names(self) -> List[str]:
        """Return all available formation template names."""
        return list(self.templates.keys())

    def add_template(self, name: str, positions: np.ndarray) -> None:
        """Add a custom formation template."""
        if len(positions) != 10:
            print(f"Warning: template '{name}' has {len(positions)} players, expected 10")
        self.templates[name] = positions.copy()


if __name__ == "__main__":
    print("=== Template Matcher Test ===\n")

    matcher = TemplateMatcher()
    print(f"Available formations: {matcher.get_template_names()}")

    # Test with a clear 4-3-3 formation
    positions_433 = np.array([
        [25, 10], [25, 27], [25, 41], [25, 58],
        [48, 17], [48, 34], [48, 51],
        [78, 14], [78, 34], [78, 54],
    ], dtype=np.float64)

    result = matcher.match(positions_433, team="team_a")
    print(f"\n4-3-3 input:")
    print(f"  Best match: {result.best_formation}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Top 3:")
    for m in result.all_matches[:3]:
        print(f"    {m.formation_name}: mean_cost={m.mean_cost:.3f}")

    # Test with a 4-4-2
    positions_442 = np.array([
        [22, 10], [22, 27], [22, 41], [22, 58],
        [50, 10], [50, 27], [50, 41], [50, 58],
        [75, 25], [75, 43],
    ], dtype=np.float64)

    result_442 = matcher.match(positions_442, team="team_a")
    print(f"\n4-4-2 input:")
    print(f"  Best match: {result_442.best_formation}")
    print(f"  Confidence: {result_442.confidence:.2f}")

    print("\n=== Tests complete ===")