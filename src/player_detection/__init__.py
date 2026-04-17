"""
player_detection — HOG+SVM player detection.

Detects players in broadcast frames using Histogram of Oriented
Gradients features with a Support Vector Machine classifier.
Trained from scratch on the Roboflow football dataset.

Modules:
    data_preparation.py  — Extract training patches from annotated data
    hog_extractor.py     — Compute HOG features from image patches
    svm_classifier.py    — Train and run SVM classifier
    sliding_window.py    — Multi-scale detection on full frames
"""

from .hog_extractor import HOGExtractor
from .svm_classifier import SVMClassifier
from .sliding_window import SlidingWindowDetector, PlayerDetection, draw_player_detections
from .data_preparation import DataPreparation
