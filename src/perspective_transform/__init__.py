"""
perspective_transform — Project pixel positions to 2D pitch coordinates.

Takes player foot positions in image pixels and the homography matrix
from camera calibration, and converts them to real pitch coordinates
in metres. This is the bridge between detection and formation analysis.

Modules:
    homography.py       — Compute and manage homography matrices
    pitch_projection.py — Project positions and visualise on 2D pitch
"""

from .homography import HomographyManager
from .pitch_projection import PitchProjector