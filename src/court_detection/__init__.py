"""
court_detection — Detect the football court area and pitch lines.

Provides grass segmentation and line detection that feed into
camera calibration and serve as masks for player/ball detection.

Modules:
    edge_detector.py        — Canny edge detection with preprocessing
    grass_segmentation.py   — Segment grass area using HSV thresholding
    line_detector.py        — Detect pitch lines using Hough Transform
"""

from .edge_detector import EdgeDetector
from .grass_segmentation import GrassSegmenter
from .line_detector import LineDetector