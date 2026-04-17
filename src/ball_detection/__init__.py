"""
ball_detection — Football detection using Circle Hough Transform.

Modules:
    circle_hough.py         — Modified CHT with gradient versor normalisation
    semi_circle_detector.py  — Shadow-aware variant for natural light
    background_subtraction.py — Remove static objects to reduce false positives
"""

from .circle_hough import CircleHoughDetector, BallDetection, draw_ball_detection
from .semi_circle_detector import SemicircleDetector
from .background_subtraction import BackgroundSubtractor
