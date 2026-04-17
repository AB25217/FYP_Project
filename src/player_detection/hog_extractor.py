"""
hog_extractor.py — Histogram of Oriented Gradients feature extraction.

Computes HOG features from image patches for player/non-player
classification. HOG captures the distribution of gradient directions
in local regions, making it effective for detecting human figures.

HOG parameters :
    - Cell size: 8x8 pixels
    - Block size: 2x2 cells (16x16 pixels)
    - Block stride: 1 cell (8x8 pixels)
    - Orientation bins: 9 (0-180 degrees, unsigned gradients)
    - Window size: 48x96 pixels (configured for football players)

The feature dimension depends on window size and HOG parameters:
    For 48x96 window: ((48-16)/8 + 1) * ((96-16)/8 + 1) * 4 * 9 = 5 * 11 * 36 = 1980

"""

import numpy as np
import cv2
import os
import glob
from typing import Tuple, List, Optional


class HOGExtractor:
    """
    Extract HOG features from image patches.

    Usage:
        extractor = HOGExtractor(window_size=(48, 96))

        # Single patch
        feature = extractor.extract(patch)

        # Batch from directory
        features, labels = extractor.extract_from_directory(
            positive_dir="data/processed/hog_training/positive",
            negative_dir="data/processed/hog_training/negative"
        )
    """

    def __init__(
        self,
        window_size: Tuple[int, int] = (48, 96),
        cell_size: Tuple[int, int] = (8, 8),
        block_size: Tuple[int, int] = (16, 16),
        block_stride: Tuple[int, int] = (8, 8),
        nbins: int = 9,
    ):
        """
        Args:
            window_size: (width, height) of input patches
            cell_size: (width, height) of HOG cells
            block_size: (width, height) of HOG blocks (in pixels)
            block_stride: (width, height) of block stride
            nbins: number of orientation histogram bins
        """
        self.window_size = window_size
        self.cell_size = cell_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.nbins = nbins

        # Create OpenCV HOG descriptor
        self.hog = cv2.HOGDescriptor(
            _winSize=window_size,
            _blockSize=block_size,
            _blockStride=block_stride,
            _cellSize=cell_size,
            _nbins=nbins,
        )

        # Calculate feature dimension
        blocks_x = (window_size[0] - block_size[0]) // block_stride[0] + 1
        blocks_y = (window_size[1] - block_size[1]) // block_stride[1] + 1
        cells_per_block = (block_size[0] // cell_size[0]) * (block_size[1] // cell_size[1])
        self.feature_dim = blocks_x * blocks_y * cells_per_block * nbins

    def extract(self, patch: np.ndarray) -> np.ndarray:
        """
        Extract HOG features from a single image patch.

        Args:
            patch: image patch, will be resized to window_size if needed.
                  Can be BGR or grayscale.

        Returns:
            np.ndarray: 1D feature vector of shape (feature_dim,)
        """
        # Convert to grayscale if needed
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = patch

        # Resize to window size
        if gray.shape[:2] != (self.window_size[1], self.window_size[0]):
            gray = cv2.resize(gray, self.window_size)

        # Compute HOG
        features = self.hog.compute(gray)

        return features.flatten()

    def extract_batch(self, patches: List[np.ndarray]) -> np.ndarray:
        """
        Extract HOG features from multiple patches.

        Args:
            patches: list of image patches

        Returns:
            np.ndarray: feature matrix of shape (N, feature_dim)
        """
        features = []
        for patch in patches:
            feat = self.extract(patch)
            features.append(feat)

        return np.array(features)

    def extract_from_directory(
        self,
        positive_dir: str,
        negative_dir: str,
        max_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract HOG features from positive and negative patch directories.

        Args:
            positive_dir: directory of player patches
            negative_dir: directory of background patches
            max_samples: limit samples per class (for testing)

        Returns:
            tuple: (features, labels)
                features: (N, feature_dim) float array
                labels: (N,) int array (1=player, 0=background)
        """
        features = []
        labels = []

        # Load positive patches
        pos_paths = sorted(
            glob.glob(os.path.join(positive_dir, "*.jpg")) +
            glob.glob(os.path.join(positive_dir, "*.png"))
        )
        if max_samples is not None:
            pos_paths = pos_paths[:max_samples]

        print(f"Extracting HOG from {len(pos_paths)} positive patches...")
        for path in pos_paths:
            patch = cv2.imread(path)
            if patch is not None:
                feat = self.extract(patch)
                features.append(feat)
                labels.append(1)

        # Load negative patches
        neg_paths = sorted(
            glob.glob(os.path.join(negative_dir, "*.jpg")) +
            glob.glob(os.path.join(negative_dir, "*.png"))
        )
        if max_samples is not None:
            neg_paths = neg_paths[:max_samples]

        # Also include hard negatives if they exist
        hard_neg_dir = os.path.join(os.path.dirname(negative_dir), "hard_negatives")
        if os.path.exists(hard_neg_dir):
            hard_paths = sorted(
                glob.glob(os.path.join(hard_neg_dir, "*.jpg")) +
                glob.glob(os.path.join(hard_neg_dir, "*.png"))
            )
            neg_paths.extend(hard_paths)
            print(f"  Including {len(hard_paths)} hard negatives")

        print(f"Extracting HOG from {len(neg_paths)} negative patches...")
        for path in neg_paths:
            patch = cv2.imread(path)
            if patch is not None:
                feat = self.extract(patch)
                features.append(feat)
                labels.append(0)

        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        print(f"Total: {len(features)} samples, {self.feature_dim}-dim features")
        print(f"  Positive: {np.sum(labels == 1)}")
        print(f"  Negative: {np.sum(labels == 0)}")

        return features, labels

    def get_params(self) -> dict:
        """Return HOG parameters for logging/saving."""
        return {
            "window_size": self.window_size,
            "cell_size": self.cell_size,
            "block_size": self.block_size,
            "block_stride": self.block_stride,
            "nbins": self.nbins,
            "feature_dim": self.feature_dim,
        }


if __name__ == "__main__":
    print("=== HOG Extractor Test ===\n")

    extractor = HOGExtractor(window_size=(48, 96))
    params = extractor.get_params()
    print(f"1. HOG parameters:")
    for k, v in params.items():
        print(f"   {k}: {v}")

    # Test on synthetic patch
    print(f"\n2. Testing feature extraction...")
    patch = np.random.randint(0, 255, (96, 48, 3), dtype=np.uint8)
    feat = extractor.extract(patch)
    print(f"   Input shape: {patch.shape}")
    print(f"   Feature shape: {feat.shape}")
    print(f"   Feature dim: {extractor.feature_dim}")

    # Test batch
    print(f"\n3. Testing batch extraction...")
    patches = [np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
               for _ in range(10)]
    features = extractor.extract_batch(patches)
    print(f"   Batch size: {len(patches)}")
    print(f"   Features shape: {features.shape}")

    # Test grayscale input
    print(f"\n4. Testing grayscale input...")
    gray_patch = np.random.randint(0, 255, (96, 48), dtype=np.uint8)
    feat_gray = extractor.extract(gray_patch)
    print(f"   Grayscale feature shape: {feat_gray.shape}")

    print("\n=== Tests complete ===")