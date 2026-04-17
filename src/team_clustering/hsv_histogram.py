
"""
hsv_histogram.py — Extract HSV colour features from player crops.

Converts each player bounding box to HSV colour space and computes
a normalised histogram. Grass pixels are masked out so only the
player's jersey contributes to the colour profile.

Reference: Mavrogiannis & Maglogiannis (2022) Section 3.6
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


class HSVFeatureExtractor:
    def __init__(
        self,
        h_bins: int = 30,
        s_bins: int = 32,
        v_bins: int = 0,
        use_upper_body: bool = True,
        upper_body_ratio: float = 0.5,
        mask_grass: bool = True,
        grass_h_range: Tuple[int, int] = (30, 85),
        grass_s_range: Tuple[int, int] = (30, 255),
        grass_v_range: Tuple[int, int] = (30, 255),
        min_non_grass_pixels: int = 50,
    ):
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins
        self.use_upper_body = use_upper_body
        self.upper_body_ratio = upper_body_ratio
        self.mask_grass = mask_grass
        self.grass_h_range = grass_h_range
        self.grass_s_range = grass_s_range
        self.grass_v_range = grass_v_range
        self.min_non_grass_pixels = min_non_grass_pixels
        self.feature_dim = h_bins + s_bins + (v_bins if v_bins > 0 else 0)

    def extract(self, player_crop: np.ndarray) -> Optional[np.ndarray]:
        if player_crop.size == 0 or player_crop.shape[0] < 5 or player_crop.shape[1] < 5:
            return None

        if self.use_upper_body:
            h = player_crop.shape[0]
            crop = player_crop[:int(h * self.upper_body_ratio), :]
        else:
            crop = player_crop

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        if self.mask_grass:
            lower = np.array([self.grass_h_range[0], self.grass_s_range[0], self.grass_v_range[0]])
            upper = np.array([self.grass_h_range[1], self.grass_s_range[1], self.grass_v_range[1]])
            grass_mask = cv2.inRange(hsv, lower, upper)
            non_grass_mask = cv2.bitwise_not(grass_mask)
        else:
            non_grass_mask = np.ones(crop.shape[:2], dtype=np.uint8) * 255

        if np.sum(non_grass_mask > 0) < self.min_non_grass_pixels:
            return None

        features = []

        h_hist = cv2.calcHist([hsv], [0], non_grass_mask, [self.h_bins], [0, 180]).flatten()
        features.append(self._normalise(h_hist))

        s_hist = cv2.calcHist([hsv], [1], non_grass_mask, [self.s_bins], [0, 256]).flatten()
        features.append(self._normalise(s_hist))

        if self.v_bins > 0:
            v_hist = cv2.calcHist([hsv], [2], non_grass_mask, [self.v_bins], [0, 256]).flatten()
            features.append(self._normalise(v_hist))

        return np.concatenate(features)

    def extract_batch(self, crops: List[np.ndarray]) -> Tuple[np.ndarray, List[int]]:
        features = []
        valid_indices = []
        for i, crop in enumerate(crops):
            feat = self.extract(crop)
            if feat is not None:
                features.append(feat)
                valid_indices.append(i)
        if len(features) == 0:
            return np.zeros((0, self.feature_dim)), []
        return np.array(features, dtype=np.float32), valid_indices

    def extract_from_frame(self, frame: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, List[int]]:
        h, w = frame.shape[:2]
        crops = []
        for (bx, by, bw, bh) in bboxes:
            bx, by = max(0, bx), max(0, by)
            bw, bh = min(bw, w - bx), min(bh, h - by)
            crops.append(frame[by:by+bh, bx:bx+bw])
        return self.extract_batch(crops)

    def _normalise(self, histogram: np.ndarray) -> np.ndarray:
        total = histogram.sum()
        return histogram / total if total > 0 else histogram