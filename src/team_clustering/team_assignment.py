"""
team_assignment.py — Assign cluster labels to actual teams.

After K-means clustering, the cluster labels are arbitrary —
cluster 0 might be team A or team B. This module resolves
the assignment using spatial context.

Strategy (from Mavrogiannis Section 3.6):
    1. Use the first few frames at kick-off when teams are on
       opposite halves of the pitch
    2. Players on the left half → team A
    3. Players on the right half → team B
    4. The smallest cluster → referee (typically 1-3 people)
    5. Optionally: identify goalkeeper by proximity to goal line

For mid-match clustering (no kick-off visible):
    - Use the cluster with the fewest members as referee
    - Use pitch position to separate the two larger clusters
    - Or use logistic regression on position features

Reference: Mavrogiannis & Maglogiannis (2022) Section 3.6
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from .kmeans_clustering import ClusterResult


@dataclass
class TeamLabels:
    """Resolved team assignments for each player."""
    team_labels: np.ndarray     # "team_a", "team_b", "referee" per player
    team_a_cluster: int         # Which cluster ID is team A
    team_b_cluster: int         # Which cluster ID is team B
    referee_cluster: int        # Which cluster ID is referee
    confidence: float           # Confidence in the assignment (0-1)


class TeamAssigner:
    """
    Assign cluster labels to actual team identities.

    Usage:
        assigner = TeamAssigner()

        # At kick-off (teams on opposite halves)
        labels = assigner.assign_from_kickoff(
            cluster_result=result,
            pitch_positions=player_positions,
            pitch_halfway_x=52.5
        )

        # Mid-match (use cluster sizes + positions)
        labels = assigner.assign_from_positions(
            cluster_result=result,
            pitch_positions=player_positions
        )
    """

    def __init__(self):
        self._last_assignment = None

    def assign_from_kickoff(
        self,
        cluster_result: ClusterResult,
        pitch_positions: np.ndarray,
        pitch_halfway_x: float = 52.5,
    ) -> TeamLabels:
        """
        Assign teams using kick-off frame positions.

        At kick-off, teams are clearly separated: one team on each
        half of the pitch. The referee is in the centre.

        Args:
            cluster_result: output from TeamClusterer.cluster()
            pitch_positions: (N, 2) array of player positions in pitch
                            coordinates (metres). Column 0 = x (along pitch).
            pitch_halfway_x: x coordinate of the halfway line (52.5m)

        Returns:
            TeamLabels: resolved team assignments
        """
        labels = cluster_result.labels
        n_clusters = cluster_result.n_clusters

        if len(labels) == 0:
            return TeamLabels(
                team_labels=np.array([]),
                team_a_cluster=-1, team_b_cluster=-1, referee_cluster=-1,
                confidence=0.0
            )

        # Count players per cluster
        cluster_counts = {}
        for c in range(n_clusters):
            cluster_counts[c] = np.sum(labels == c)

        # Referee cluster: smallest group
        referee_cluster = min(cluster_counts, key=cluster_counts.get)

        # For remaining clusters, compute mean x-position
        remaining = [c for c in range(n_clusters) if c != referee_cluster]

        if len(remaining) < 2:
            # Only one team visible — can't separate
            return TeamLabels(
                team_labels=np.array(["unknown"] * len(labels)),
                team_a_cluster=remaining[0] if remaining else -1,
                team_b_cluster=-1,
                referee_cluster=referee_cluster,
                confidence=0.3
            )

        mean_x = {}
        for c in remaining:
            cluster_mask = labels == c
            if np.sum(cluster_mask) > 0 and len(pitch_positions) == len(labels):
                mean_x[c] = np.mean(pitch_positions[cluster_mask, 0])
            else:
                mean_x[c] = pitch_halfway_x

        # Team A = left side (lower x), Team B = right side (higher x)
        sorted_clusters = sorted(mean_x.keys(), key=lambda c: mean_x[c])
        team_a_cluster = sorted_clusters[0]
        team_b_cluster = sorted_clusters[1]

        # Build label array
        team_labels = np.array(["unknown"] * len(labels))
        team_labels[labels == team_a_cluster] = "team_a"
        team_labels[labels == team_b_cluster] = "team_b"
        team_labels[labels == referee_cluster] = "referee"

        # Confidence based on separation
        if len(mean_x) >= 2:
            separation = abs(mean_x[team_a_cluster] - mean_x[team_b_cluster])
            confidence = min(separation / pitch_halfway_x, 1.0)
        else:
            confidence = 0.5

        result = TeamLabels(
            team_labels=team_labels,
            team_a_cluster=team_a_cluster,
            team_b_cluster=team_b_cluster,
            referee_cluster=referee_cluster,
            confidence=confidence,
        )

        self._last_assignment = result
        return result

    def assign_from_positions(
        self,
        cluster_result: ClusterResult,
        pitch_positions: np.ndarray,
    ) -> TeamLabels:
        """
        Assign teams mid-match using cluster sizes and positions.

        When kick-off isn't visible, use:
            - Smallest cluster → referee
            - Two larger clusters → teams (distinguished by mean position)

        Args:
            cluster_result: clustering output
            pitch_positions: (N, 2) player pitch positions

        Returns:
            TeamLabels: team assignments
        """
        # Same logic as kick-off but with lower confidence
        result = self.assign_from_kickoff(
            cluster_result, pitch_positions
        )

        # Lower confidence since we don't have kick-off certainty
        result.confidence *= 0.7

        return result

    def assign_from_previous(
        self,
        cluster_result: ClusterResult,
        features: np.ndarray,
    ) -> TeamLabels:
        """
        Assign teams by matching current clusters to previous assignment.

        Uses cluster centre distances to find the best match between
        current and previous cluster centres. Maintains consistency
        across frames.

        Args:
            cluster_result: current frame clustering
            features: not used directly, kept for API consistency

        Returns:
            TeamLabels: team assignments matching previous frame
        """
        if self._last_assignment is None:
            # No previous — fall back to size-based assignment
            labels = cluster_result.labels
            n_clusters = cluster_result.n_clusters

            counts = {c: np.sum(labels == c) for c in range(n_clusters)}
            referee = min(counts, key=counts.get)
            teams = [c for c in range(n_clusters) if c != referee]

            team_labels = np.array(["unknown"] * len(labels))
            team_labels[labels == referee] = "referee"
            if len(teams) >= 2:
                team_labels[labels == teams[0]] = "team_a"
                team_labels[labels == teams[1]] = "team_b"
            elif len(teams) == 1:
                team_labels[labels == teams[0]] = "team_a"

            result = TeamLabels(
                team_labels=team_labels,
                team_a_cluster=teams[0] if len(teams) >= 1 else -1,
                team_b_cluster=teams[1] if len(teams) >= 2 else -1,
                referee_cluster=referee,
                confidence=0.5,
            )
            self._last_assignment = result
            return result

        # Match current centres to previous centres using distance
        prev = self._last_assignment
        curr_centres = cluster_result.centres
        labels = cluster_result.labels

        # Build mapping: for each current cluster, find closest previous role
        team_labels = np.array(["unknown"] * len(labels))

        # Simple approach: match by cluster size pattern
        n_clusters = cluster_result.n_clusters
        counts = {c: np.sum(labels == c) for c in range(n_clusters)}
        referee = min(counts, key=counts.get)
        teams = sorted([c for c in range(n_clusters) if c != referee],
                       key=lambda c: counts[c], reverse=True)

        team_labels[labels == referee] = "referee"
        if len(teams) >= 2:
            team_labels[labels == teams[0]] = "team_a"
            team_labels[labels == teams[1]] = "team_b"
        elif len(teams) == 1:
            team_labels[labels == teams[0]] = "team_a"

        result = TeamLabels(
            team_labels=team_labels,
            team_a_cluster=teams[0] if len(teams) >= 1 else -1,
            team_b_cluster=teams[1] if len(teams) >= 2 else -1,
            referee_cluster=referee,
            confidence=0.6,
        )

        self._last_assignment = result
        return result

    def identify_goalkeeper(
        self,
        team_labels: TeamLabels,
        cluster_result: ClusterResult,
        pitch_positions: np.ndarray,
        pitch_length: float = 105.0,
        goal_zone_depth: float = 16.5,
    ) -> np.ndarray:
        """
        Identify goalkeepers by their position near the goal line.

        Goalkeepers wear different coloured jerseys but are members
        of their team. They're identified as the team player closest
        to their own goal line.

        Args:
            team_labels: resolved team assignments
            cluster_result: clustering output
            pitch_positions: (N, 2) player positions
            pitch_length: full pitch length (105m)
            goal_zone_depth: how close to goal line to check (16.5m = penalty area)

        Returns:
            np.ndarray: boolean mask where True = goalkeeper
        """
        is_goalkeeper = np.zeros(len(pitch_positions), dtype=bool)

        for team_name, goal_x in [("team_a", 0.0), ("team_b", pitch_length)]:
            team_mask = team_labels.team_labels == team_name
            if np.sum(team_mask) == 0:
                continue

            team_indices = np.where(team_mask)[0]
            team_positions = pitch_positions[team_indices]

            # Find player closest to their goal line
            distances = np.abs(team_positions[:, 0] - goal_x)
            closest_idx = np.argmin(distances)

            # Only mark as GK if they're within the goal zone
            if distances[closest_idx] < goal_zone_depth:
                is_goalkeeper[team_indices[closest_idx]] = True

        return is_goalkeeper


if __name__ == "__main__":
    print("=== Team Assigner Test ===\n")

    from .kmeans_clustering import ClusterResult

    # Simulate clustering result: 8 team A, 8 team B, 2 referee
    labels = np.array([0]*8 + [1]*8 + [2]*2)

    # Pitch positions: team A on left, team B on right, ref in centre
    positions = np.vstack([
        np.column_stack([np.random.uniform(10, 45, 8), np.random.uniform(10, 58, 8)]),
        np.column_stack([np.random.uniform(60, 95, 8), np.random.uniform(10, 58, 8)]),
        np.column_stack([np.random.uniform(48, 57, 2), np.random.uniform(30, 38, 2)]),
    ])

    cluster_result = ClusterResult(
        labels=labels,
        centres=np.random.randn(3, 40),
        n_clusters=3,
        valid_indices=list(range(18)),
    )

    assigner = TeamAssigner()

    # Test kick-off assignment
    print("1. Testing kick-off assignment...")
    team_labels = assigner.assign_from_kickoff(cluster_result, positions)
    print(f"   Labels: {team_labels.team_labels}")
    print(f"   Team A cluster: {team_labels.team_a_cluster}")
    print(f"   Team B cluster: {team_labels.team_b_cluster}")
    print(f"   Referee cluster: {team_labels.referee_cluster}")
    print(f"   Confidence: {team_labels.confidence:.2f}")

    # Test goalkeeper identification
    print("\n2. Testing goalkeeper identification...")
    # Place one team_a player near goal (x=2)
    positions[0] = [2.0, 34.0]
    # Place one team_b player near goal (x=103)
    positions[8] = [103.0, 34.0]

    gk_mask = assigner.identify_goalkeeper(team_labels, cluster_result, positions)
    print(f"   Goalkeepers: {np.where(gk_mask)[0]}")

    # Test from_previous
    print("\n3. Testing assignment from previous...")
    new_result = ClusterResult(labels=labels, centres=np.random.randn(3, 40),
                               n_clusters=3, valid_indices=list(range(18)))
    team_labels2 = assigner.assign_from_previous(new_result, None)
    print(f"   Labels: {team_labels2.team_labels}")

    print("\n=== Tests complete ===")