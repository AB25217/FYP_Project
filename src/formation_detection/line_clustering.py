"""
line_clustering.py — Group outfield players into defensive lines.

A formation like 4-3-3 has three lines:
    - Defence (4 players)
    - Midfield (3 players)
    - Attack (3 players)

This module clusters outfield player positions along the x-axis
(distance from their goal) to identify these lines. K-means is
used with k=3 or k=4, and the best k is selected using the
silhouette score.

For team_a (defending left, x=0):
    - Low x values = defenders
    - Mid x values = midfielders
    - High x values = attackers

For team_b (defending right, x=105):
    - High x values = defenders
    - Mid x values = midfielders
    - Low x values = attackers

The number of players in each line gives the formation string:
    [4 defenders, 3 midfielders, 3 attackers] → "4-3-3"
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LineClusterResult:
    """Result of line clustering."""
    line_counts: List[int]       # Players per line [defence, mid, ..., attack]
    line_labels: np.ndarray      # Line index per player (0=defence, ...)
    line_centres: np.ndarray     # Mean x-position of each line
    num_lines: int               # Number of lines found
    silhouette: float            # Silhouette score (quality measure)
    formation_string: str        # e.g. "4-3-3"


class LineClustering:
    """
    Cluster outfield players into defensive lines.

    Usage:
        lc = LineClustering()
        result = lc.cluster(positions, team="team_a")
        print(result.formation_string)  # "4-3-3"
    """

    def __init__(
        self,
        min_k: int = 2,
        max_k: int = 4,
        random_state: int = 42,
    ):
        """
        Args:
            min_k: minimum number of lines to try
            max_k: maximum number of lines to try
            random_state: for reproducibility
        """
        self.min_k = min_k
        self.max_k = max_k
        self.random_state = random_state

    def cluster(
        self,
        positions: np.ndarray,
        team: str = "team_a",
    ) -> LineClusterResult:
        """
        Cluster outfield players into lines.

        Args:
            positions: (N, 2) outfield player positions (pitch coords)
            team: "team_a" (defends x=0) or "team_b" (defends x=105)

        Returns:
            LineClusterResult: line assignment and formation string
        """
        if len(positions) < 3:
            return LineClusterResult(
                line_counts=[len(positions)],
                line_labels=np.zeros(len(positions), dtype=np.int32),
                line_centres=np.array([np.mean(positions[:, 0])]) if len(positions) > 0 else np.array([]),
                num_lines=1,
                silhouette=0.0,
                formation_string=str(len(positions)),
            )

        # Use x-coordinate (distance along pitch) for clustering
        x_positions = positions[:, 0].reshape(-1, 1)

        # For team_b, flip x so defenders have low values
        if team == "team_b":
            x_positions = 105.0 - x_positions

        # Try different k values and pick the best silhouette
        best_result = None
        best_silhouette = -1.0

        for k in range(self.min_k, min(self.max_k + 1, len(positions))):
            kmeans = KMeans(
                n_clusters=k, n_init=10,
                random_state=self.random_state
            )
            labels = kmeans.fit_predict(x_positions)

            if k >= 2 and len(set(labels)) >= 2:
                sil = silhouette_score(x_positions, labels)
            else:
                sil = 0.0

            if sil > best_silhouette:
                best_silhouette = sil
                best_labels = labels
                best_k = k
                best_centres = kmeans.cluster_centers_.flatten()

        # Sort lines from defence to attack (low x = defence)
        sorted_order = np.argsort(best_centres)

        # Re-map labels so 0 = defence, 1 = midfield, etc.
        label_map = {old: new for new, old in enumerate(sorted_order)}
        remapped_labels = np.array([label_map[l] for l in best_labels])
        sorted_centres = best_centres[sorted_order]

        # Count players per line
        line_counts = []
        for line_idx in range(best_k):
            count = int(np.sum(remapped_labels == line_idx))
            line_counts.append(count)

        # Build formation string
        formation_string = "-".join(str(c) for c in line_counts)

        return LineClusterResult(
            line_counts=line_counts,
            line_labels=remapped_labels,
            line_centres=sorted_centres,
            num_lines=best_k,
            silhouette=float(best_silhouette),
            formation_string=formation_string,
        )

    def cluster_with_fixed_k(
        self,
        positions: np.ndarray,
        k: int,
        team: str = "team_a",
    ) -> LineClusterResult:
        """
        Cluster with a specific number of lines (no auto-selection).

        Useful when you know the expected formation structure
        (e.g. k=3 for standard 4-4-2, k=4 for 4-2-3-1).
        """
        if len(positions) < k:
            return self.cluster(positions, team)

        x_positions = positions[:, 0].reshape(-1, 1)
        if team == "team_b":
            x_positions = 105.0 - x_positions

        kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
        labels = kmeans.fit_predict(x_positions)
        centres = kmeans.cluster_centers_.flatten()

        if k >= 2 and len(set(labels)) >= 2:
            sil = silhouette_score(x_positions, labels)
        else:
            sil = 0.0

        sorted_order = np.argsort(centres)
        label_map = {old: new for new, old in enumerate(sorted_order)}
        remapped_labels = np.array([label_map[l] for l in labels])
        sorted_centres = centres[sorted_order]

        line_counts = [int(np.sum(remapped_labels == i)) for i in range(k)]
        formation_string = "-".join(str(c) for c in line_counts)

        return LineClusterResult(
            line_counts=line_counts,
            line_labels=remapped_labels,
            line_centres=sorted_centres,
            num_lines=k,
            silhouette=float(sil),
            formation_string=formation_string,
        )


if __name__ == "__main__":
    print("=== Line Clustering Test ===\n")

    lc = LineClustering()

    # Simulate a 4-3-3 formation (team_a, defending x=0)
    positions_433 = np.array([
        # Defence (x ~ 25)
        [25, 10], [25, 27], [25, 41], [25, 58],
        # Midfield (x ~ 50)
        [50, 15], [50, 34], [50, 53],
        # Attack (x ~ 75)
        [75, 15], [75, 34], [75, 53],
    ], dtype=np.float64)

    result = lc.cluster(positions_433, team="team_a")
    print(f"4-3-3 test:")
    print(f"  Formation: {result.formation_string}")
    print(f"  Line counts: {result.line_counts}")
    print(f"  Silhouette: {result.silhouette:.3f}")
    print(f"  Line centres: {np.round(result.line_centres, 1)}")

    # Simulate a 4-4-2
    positions_442 = np.array([
        [20, 10], [20, 27], [20, 41], [20, 58],
        [45, 10], [45, 27], [45, 41], [45, 58],
        [70, 25], [70, 43],
    ], dtype=np.float64)

    result_442 = lc.cluster(positions_442, team="team_a")
    print(f"\n4-4-2 test:")
    print(f"  Formation: {result_442.formation_string}")
    print(f"  Line counts: {result_442.line_counts}")

    # Simulate team_b (defending x=105)
    positions_b = np.array([
        [80, 10], [80, 27], [80, 41], [80, 58],
        [55, 20], [55, 34], [55, 48],
        [30, 20], [30, 34], [30, 48],
    ], dtype=np.float64)

    result_b = lc.cluster(positions_b, team="team_b")
    print(f"\nTeam B 4-3-3 test:")
    print(f"  Formation: {result_b.formation_string}")

    print("\n=== Tests complete ===")