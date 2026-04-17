"""
kmeans_clustering.py — Cluster players into teams using K-means.

Takes HSV histogram features from all detected players and groups
them into K clusters. Typically K=3 (team A, team B, referee)
or K=4 (adding a separate goalkeeper cluster).
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ClusterResult:
    """Result of team clustering."""
    labels: np.ndarray          # Cluster label per player
    centres: np.ndarray         # Cluster centres in feature space
    n_clusters: int
    valid_indices: List[int]    # Indices of players that were clustered


class TeamClusterer:
    """
    K-means clustering for team separation.

    Usage:
        clusterer = TeamClusterer(n_clusters=3)
        result = clusterer.cluster(features)

        # Or accumulate over frames for stability
        clusterer.accumulate(features_frame_1)
        clusterer.accumulate(features_frame_2)
        result = clusterer.cluster_accumulated()
    """

    def __init__(
        self,
        n_clusters: int = 3,
        max_iterations: int = 100,
        n_init: int = 10,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.n_init = n_init
        self.random_state = random_state

        self.kmeans = None
        self._accumulated_features = []
        self._accumulated_indices = []

    def cluster(self, features, valid_indices=None):
        if len(features) < self.n_clusters:
            return ClusterResult(
                labels=np.zeros(len(features), dtype=np.int32),
                centres=features[:1] if len(features) > 0 else np.array([]),
                n_clusters=1,
                valid_indices=valid_indices or list(range(len(features))),
            )

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iterations,
            n_init=self.n_init,
            random_state=self.random_state,
        )

        labels = self.kmeans.fit_predict(features)
        return ClusterResult(
            labels=labels,
            centres=self.kmeans.cluster_centers_,
            n_clusters=self.n_clusters,
            valid_indices=valid_indices or list(range(len(features))),
        )

    def predict(self, features):
        if self.kmeans is None:
            raise RuntimeError("No model. Call cluster() first.")
        return self.kmeans.predict(features)

    def accumulate(self, features, valid_indices=None):
        self._accumulated_features.append(features)
        if valid_indices is not None:
            self._accumulated_indices.extend(valid_indices)

    def cluster_accumulated(self):
        if len(self._accumulated_features) == 0:
            return ClusterResult(np.array([]), np.array([]), 0, [])
        all_features = np.vstack(self._accumulated_features)
        return self.cluster(all_features, self._accumulated_indices)

    def clear_accumulated(self):
        self._accumulated_features = []
        self._accumulated_indices = []

    def evaluate(self, predicted_labels, ground_truth_labels):
        v = v_measure_score(ground_truth_labels, predicted_labels)
        h = homogeneity_score(ground_truth_labels, predicted_labels)
        c = completeness_score(ground_truth_labels, predicted_labels)

        print(f"Clustering evaluation:")
        print(f"  V-measure:    {v:.3f}")
        print(f"  Homogeneity:  {h:.3f}")
        print(f"  Completeness: {c:.3f}")

        return {"v_measure": float(v), "homogeneity": float(h), "completeness": float(c)}


if __name__ == "__main__":
    print("=== Team Clusterer Test ===\n")
    np.random.seed(42)

    red = np.random.randn(8, 40).astype(np.float32) + 2
    blue = np.random.randn(8, 40).astype(np.float32) - 2
    ref = np.random.randn(2, 40).astype(np.float32)
    features = np.vstack([red, blue, ref])
    gt = np.array([0]*8 + [1]*8 + [2]*2)

    idx = np.random.permutation(len(features))
    features, gt = features[idx], gt[idx]

    clusterer = TeamClusterer(n_clusters=3)
    result = clusterer.cluster(features)
    print(f"Labels: {result.labels}")
    clusterer.evaluate(result.labels, gt)

    print("\n=== Tests complete ===")