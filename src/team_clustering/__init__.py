"""
team_clustering — Separate detected players into teams using jersey colours.

Uses HSV colour histograms from player bounding boxes with K-means
clustering to group players into teams. Based on the approach from
Mavrogiannis & Maglogiannis (2022).

Modules:
    hsv_histogram.py      — Extract colour features from player crops
    kmeans_clustering.py   — Cluster players into groups by colour
    team_assignment.py     — Assign cluster labels to actual teams
"""

from .hsv_histogram import HSVFeatureExtractor
from .kmeans_clustering import TeamClusterer
from .team_assignment import TeamAssigner