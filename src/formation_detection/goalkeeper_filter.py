"""
goalkeeper_filter.py — Identify and exclude goalkeepers.

Formations describe outfield players only (e.g. 4-4-2 = 10 outfield
players). Goalkeepers must be identified and excluded before
formation analysis.

Identification methods:
    1. Position-based: player closest to their own goal line
       within the penalty area depth (16.5m)
    2. Cluster-based: if team clustering produced a separate
       goalkeeper cluster (different kit colour), use that

Method 1 is primary since it works even when the goalkeeper's
jersey colour is similar to the team's.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FilteredTeam:
    """Team positions with goalkeeper separated."""
    outfield_positions: np.ndarray    # (N, 2) outfield player positions
    outfield_ids: List[int]           # Player IDs of outfield players
    goalkeeper_position: Optional[Tuple[float, float]]  # GK position or None
    goalkeeper_id: Optional[int]      # GK player ID or None
    team: str                         # "team_a" or "team_b"


class GoalkeeperFilter:
    """
    Identify and exclude goalkeepers from player positions.

    Usage:
        gk_filter = GoalkeeperFilter()
        filtered = gk_filter.filter_team(
            positions=team_a_positions,
            player_ids=team_a_ids,
            team="team_a"
        )
        # filtered.outfield_positions → use for formation detection
        # filtered.goalkeeper_position → excluded from formation
    """

    def __init__(
        self,
        pitch_length: float = 105.0,
        goal_zone_depth: float = 16.5,
        min_outfield_players: int = 3,
    ):
        """
        Args:
            pitch_length: FIFA pitch length in metres
            goal_zone_depth: maximum distance from goal line to count as GK
                            (16.5m = penalty area depth)
            min_outfield_players: don't remove GK if fewer outfield remain
        """
        self.pitch_length = pitch_length
        self.goal_zone_depth = goal_zone_depth
        self.min_outfield_players = min_outfield_players

    def filter_team(
        self,
        positions: np.ndarray,
        player_ids: List[int],
        team: str,
        goalkeeper_cluster_id: Optional[int] = None,
    ) -> FilteredTeam:
        """
        Identify the goalkeeper and separate from outfield players.

        Args:
            positions: (N, 2) array of (pitch_x, pitch_y) positions
            player_ids: list of player IDs
            team: "team_a" (defends left goal, x=0) or
                  "team_b" (defends right goal, x=105)
            goalkeeper_cluster_id: if known from clustering, the GK's
                                  player_id. Takes priority over position.

        Returns:
            FilteredTeam: separated outfield and goalkeeper
        """
        if len(positions) == 0:
            return FilteredTeam(
                outfield_positions=np.array([]).reshape(0, 2),
                outfield_ids=[], goalkeeper_position=None,
                goalkeeper_id=None, team=team
            )

        # Method 1: Use cluster-based ID if provided
        if goalkeeper_cluster_id is not None and goalkeeper_cluster_id in player_ids:
            gk_idx = player_ids.index(goalkeeper_cluster_id)
            return self._split_at_index(positions, player_ids, gk_idx, team)

        # Method 2: Position-based — closest to own goal line
        if team == "team_a":
            goal_x = 0.0
        else:
            goal_x = self.pitch_length

        distances = np.abs(positions[:, 0] - goal_x)
        closest_idx = np.argmin(distances)

        # Only classify as GK if within goal zone
        if distances[closest_idx] > self.goal_zone_depth:
            # No clear goalkeeper — return all as outfield
            return FilteredTeam(
                outfield_positions=positions.copy(),
                outfield_ids=player_ids.copy(),
                goalkeeper_position=None,
                goalkeeper_id=None,
                team=team
            )

        # Don't remove GK if too few players remain
        if len(positions) - 1 < self.min_outfield_players:
            return FilteredTeam(
                outfield_positions=positions.copy(),
                outfield_ids=player_ids.copy(),
                goalkeeper_position=None,
                goalkeeper_id=None,
                team=team
            )

        return self._split_at_index(positions, player_ids, closest_idx, team)

    def _split_at_index(
        self,
        positions: np.ndarray,
        player_ids: List[int],
        gk_idx: int,
        team: str,
    ) -> FilteredTeam:
        """Split positions into outfield and goalkeeper at given index."""
        gk_pos = (float(positions[gk_idx, 0]), float(positions[gk_idx, 1]))
        gk_id = player_ids[gk_idx]

        outfield_mask = np.ones(len(positions), dtype=bool)
        outfield_mask[gk_idx] = False

        outfield_pos = positions[outfield_mask]
        outfield_ids = [pid for i, pid in enumerate(player_ids) if i != gk_idx]

        return FilteredTeam(
            outfield_positions=outfield_pos,
            outfield_ids=outfield_ids,
            goalkeeper_position=gk_pos,
            goalkeeper_id=gk_id,
            team=team
        )


if __name__ == "__main__":
    print("=== Goalkeeper Filter Test ===\n")

    gk_filter = GoalkeeperFilter()

    # Team A: defends left goal (x=0), GK should be near x=0
    positions_a = np.array([
        [3.0, 34.0],    # GK — closest to x=0
        [25.0, 15.0],   # Defender
        [25.0, 50.0],   # Defender
        [50.0, 34.0],   # Midfielder
        [80.0, 34.0],   # Forward
    ])
    ids_a = [0, 1, 2, 3, 4]

    result = gk_filter.filter_team(positions_a, ids_a, "team_a")
    print(f"Team A: GK id={result.goalkeeper_id}, pos={result.goalkeeper_position}")
    print(f"  Outfield: {len(result.outfield_positions)} players, ids={result.outfield_ids}")

    # Team B: defends right goal (x=105)
    positions_b = np.array([
        [102.0, 34.0],  # GK
        [80.0, 20.0],   # Defender
        [80.0, 48.0],   # Defender
        [55.0, 34.0],   # Midfielder
    ])
    ids_b = [10, 11, 12, 13]

    result_b = gk_filter.filter_team(positions_b, ids_b, "team_b")
    print(f"\nTeam B: GK id={result_b.goalkeeper_id}, pos={result_b.goalkeeper_position}")
    print(f"  Outfield: {len(result_b.outfield_positions)} players")

    print("\n=== Tests complete ===")