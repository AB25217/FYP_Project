"""
pipeline — End-to-end video processing pipeline.

Orchestrates all modules to process broadcast football video:
    Video → Court Detection → Camera Calibration → Player/Ball Detection
    → Tracking → Team Clustering → Perspective Transform → Formation Detection

Modules:
    video_processor.py  — Main pipeline: video in → analytics out
"""

from .video_processor import VideoProcessor
from .config_loader import get_paths