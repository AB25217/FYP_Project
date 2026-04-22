"""tactical_assigner.py — Label k=5 clusters as teams/GKs/referee using
pitch-space logic from Mavrogiannis & Maglogiannis (2022), Section 4.3.

After KMeans produces 5 arbitrary cluster IDs, this module:
    1. Designates the smallest cluster as the referee.
    2. Among the remaining 4, uses mean pitch x-position to decide which
       clusters are team_a (left-defending) and team_b (right-defending).
       Per the paper: "the basic idea to deduce team sides is that players
       of the same team on average are closer to their side than the
       opposite side."
    3. Links each goalkeeper cluster to whichever team's mean x is closer.

The paper uses logistic regression for team-side assignment; we use
the simpler equivalent (argmin of mean-x distance), which is what LR
would converge to given only the x-feature. Result is identical.

Display colours are auto-derived from each cluster's dominant H bin.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ClusterLabels:
    """Mapping from cluster_id (int) to semantic label + display colour."""
    labels: Dict[int, str]
    display_colours: Dict[int, Tuple[int, int, int]]


class TacticalAssigner:
    """Assign k=5 clusters to team_a / team_b / gk_a / gk_b / referee
    using pitch positions (per Mavrogiannis 4.3)."""

    # Pitch geometry
    _PITCH_LENGTH = 105.0  # metres

    def __init__(self):
        self._labels: Dict[int, str] = {}
        self._colours: Dict[int, Tuple[int, int, int]] = {}

    def assign(
        self,
        cluster_labels_per_detection: np.ndarray,
        cluster_centres: np.ndarray,
        pitch_positions_per_detection: Optional[np.ndarray] = None,
        h_bins: int = 36,
    ) -> ClusterLabels:
        """
        Args:
            cluster_labels_per_detection: (N,) array of cluster IDs from
                initialisation frames
            cluster_centres: (K, feature_dim) KMeans centroids
            pitch_positions_per_detection: (N, 2) array of (x, y) pitch
                coordinates in metres, aligned with cluster_labels_per_detection.
                If None, falls back to size-based assignment (team_a = largest).
            h_bins: number of H bins in the feature (first h_bins dims)

        Returns:
            ClusterLabels with semantic labels and BGR display colours.
        """
        n_clusters = cluster_centres.shape[0]
        cluster_sizes = {
            c: int(np.sum(cluster_labels_per_detection == c))
            for c in range(n_clusters)
        }

        semantic_labels: Dict[int, str] = {}

        if n_clusters >= 5 and pitch_positions_per_detection is not None:
            # Paper's approach: referee is smallest cluster,
            # teams assigned by mean pitch x, GKs linked to nearer team.
            semantic_labels = self._assign_by_pitch_position(
                cluster_labels_per_detection,
                cluster_sizes,
                pitch_positions_per_detection,
            )
        elif n_clusters >= 5:
            # Fallback: no pitch positions available — use size-based heuristic
            semantic_labels = self._assign_by_size(cluster_centres, cluster_sizes)
        elif n_clusters >= 3:
            by_size = sorted(cluster_sizes.items(), key=lambda kv: kv[1], reverse=True)
            semantic_labels[by_size[0][0]] = "team_a"
            semantic_labels[by_size[1][0]] = "team_b"
            semantic_labels[by_size[2][0]] = "referee"
        else:
            by_size = sorted(cluster_sizes.items(), key=lambda kv: kv[1], reverse=True)
            for i, (cid, _) in enumerate(by_size):
                semantic_labels[cid] = f"cluster_{i}"

        # Derive display colours from each cluster's dominant H bin
        display_colours = self._derive_colours(cluster_centres, h_bins)

        self._labels = semantic_labels
        self._colours = display_colours
        return ClusterLabels(labels=semantic_labels, display_colours=display_colours)

    # Assignment strategies
    

    def _assign_by_pitch_position(
        self,
        cluster_labels: np.ndarray,
        cluster_sizes: Dict[int, int],
        pitch_positions: np.ndarray,
    ) -> Dict[int, str]:
        """Paper's method: referee = smallest cluster, teams = mean-x sides,
        goalkeepers linked to nearer team by mean x."""
        out: Dict[int, str] = {}

        # Drop detections with invalid pitch positions (NaN)
        valid_mask = ~np.any(np.isnan(pitch_positions), axis=1)
        labels_valid = cluster_labels[valid_mask]
        pitch_valid = pitch_positions[valid_mask]

        # Mean pitch x per cluster, only using valid projections
        mean_x_per_cluster: Dict[int, float] = {}
        for c in cluster_sizes:
            member_mask = labels_valid == c
            if np.sum(member_mask) > 0:
                mean_x_per_cluster[c] = float(np.mean(pitch_valid[member_mask, 0]))
            else:
                # Cluster has no valid pitch projections — treat as centre
                mean_x_per_cluster[c] = self._PITCH_LENGTH / 2.0

        # Step 1: referee = smallest cluster
        referee_id = min(cluster_sizes, key=cluster_sizes.get)
        out[referee_id] = "referee"

        remaining = [c for c in cluster_sizes if c != referee_id]

        # Step 2: of the remaining 4, the two largest are the teams.
        remaining_sorted_by_size = sorted(
            remaining, key=lambda c: cluster_sizes[c], reverse=True
        )
        team_candidates = remaining_sorted_by_size[:2]
        gk_candidates = remaining_sorted_by_size[2:]  # up to 2 GK clusters

        # Of the two team candidates, team_a defends the left (lower mean x),
        # team_b defends the right (higher mean x).
        team_candidates_sorted = sorted(team_candidates, key=lambda c: mean_x_per_cluster[c])
        team_a_id = team_candidates_sorted[0]
        team_b_id = team_candidates_sorted[1] if len(team_candidates_sorted) > 1 else team_a_id
        out[team_a_id] = "team_a"
        if team_b_id != team_a_id:
            out[team_b_id] = "team_b"

        # Step 3: link each GK to the team whose mean x is closer.
        team_a_x = mean_x_per_cluster[team_a_id]
        team_b_x = mean_x_per_cluster[team_b_id]

        gk_a_assigned = False
        gk_b_assigned = False
        # Sort GK candidates by mean x so assignments are deterministic
        for gk_id in sorted(gk_candidates, key=lambda c: mean_x_per_cluster[c]):
            gk_x = mean_x_per_cluster[gk_id]
            # Closer to team_a?
            prefers_a = abs(gk_x - team_a_x) < abs(gk_x - team_b_x)

            if prefers_a and not gk_a_assigned:
                out[gk_id] = "gk_a"
                gk_a_assigned = True
            elif not prefers_a and not gk_b_assigned:
                out[gk_id] = "gk_b"
                gk_b_assigned = True
            elif not gk_a_assigned:
                out[gk_id] = "gk_a"
                gk_a_assigned = True
            elif not gk_b_assigned:
                out[gk_id] = "gk_b"
                gk_b_assigned = True
            else:
                out[gk_id] = f"cluster_{gk_id}"

        return out

    def _assign_by_size(
        self,
        cluster_centres: np.ndarray,
        cluster_sizes: Dict[int, int],
    ) -> Dict[int, str]:
        """Fallback when pitch positions are unavailable. Largest two clusters
        are the teams (arbitrary a/b), smallest is the referee, remaining two
        are goalkeepers linked by HSV centroid proximity."""
        out: Dict[int, str] = {}
        by_size = sorted(cluster_sizes.items(), key=lambda kv: kv[1], reverse=True)

        team_a_id = by_size[0][0]
        team_b_id = by_size[1][0]
        referee_id = by_size[-1][0]
        gk_candidates = [c for c, _ in by_size[2:-1]]

        out[team_a_id] = "team_a"
        out[team_b_id] = "team_b"
        out[referee_id] = "referee"

        team_a_c = cluster_centres[team_a_id]
        team_b_c = cluster_centres[team_b_id]
        gk_a_assigned = False
        gk_b_assigned = False
        for gk_id in gk_candidates:
            gk_c = cluster_centres[gk_id]
            dist_a = np.linalg.norm(gk_c - team_a_c)
            dist_b = np.linalg.norm(gk_c - team_b_c)
            prefers_a = dist_a < dist_b
            if prefers_a and not gk_a_assigned:
                out[gk_id] = "gk_a"; gk_a_assigned = True
            elif not prefers_a and not gk_b_assigned:
                out[gk_id] = "gk_b"; gk_b_assigned = True
            elif not gk_a_assigned:
                out[gk_id] = "gk_a"; gk_a_assigned = True
            elif not gk_b_assigned:
                out[gk_id] = "gk_b"; gk_b_assigned = True
        return out

    # Display-colour derivation


    @staticmethod
    def _derive_colours(
        cluster_centres: np.ndarray,
        h_bins: int,
    ) -> Dict[int, Tuple[int, int, int]]:
        out: Dict[int, Tuple[int, int, int]] = {}
        for cid in range(cluster_centres.shape[0]):
            hist_h = cluster_centres[cid, :h_bins]
            if hist_h.sum() > 0:
                dominant_bin = int(np.argmax(hist_h))
                hue = int(dominant_bin * 180 / h_bins)
                hsv_pixel = np.uint8([[[hue, 200, 230]]])
                bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
                out[cid] = (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))
            else:
                out[cid] = (128, 128, 128)
        return out

    @property
    def labels(self) -> Dict[int, str]:
        return self._labels

    @property
    def display_colours(self) -> Dict[int, Tuple[int, int, int]]:
        return self._colours