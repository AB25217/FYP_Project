"""
sliding_window.py — Multi-scale sliding window player detection.

Slides a detection window across the image at multiple scales,
extracts HOG features at each position, and classifies each
window as player or non-player using the trained SVM.

Multi-scale detection using an image pyramid handles players
at different distances from the camera (near players are large,
far players are small).

Non-maximum suppression (NMS) merges overlapping detections
into single bounding boxes.

Pipeline:
    Image → Image pyramid → Sliding window → HOG → SVM → NMS → Detections
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .hog_extractor import HOGExtractor
from .svm_classifier import SVMClassifier


@dataclass
class PlayerDetection:
    """A single player detection."""
    x: int              # Top-left x
    y: int              # Top-left y
    w: int              # Width
    h: int              # Height
    confidence: float   # SVM confidence (0-1)

    @property
    def centre(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def foot_position(self) -> Tuple[int, int]:
        """Bottom-centre of bounding box — approximates foot position."""
        return (self.x + self.w // 2, self.y + self.h)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


class SlidingWindowDetector:
    """
    Multi-scale sliding window player detector.

    Usage:
        # Build from trained components
        detector = SlidingWindowDetector(
            hog_extractor=HOGExtractor(),
            svm_classifier=SVMClassifier.load("weights/svm_player.pkl")
        )

        # Detect players
        detections = detector.detect(frame)
        for det in detections:
            print(f"Player at ({det.x}, {det.y}), conf={det.confidence:.2f}")
    """

    def __init__(
        self,
        hog_extractor: HOGExtractor = None,
        svm_classifier: SVMClassifier = None,
        scale_factor: float = 1.2,
        num_scales: int = 8,
        stride: int = 8,
        confidence_threshold: float = 0.6,
        nms_threshold: float = 0.3,
    ):
        """
        Args:
            hog_extractor: HOG feature extractor
            svm_classifier: trained SVM classifier
            scale_factor: scale multiplier between pyramid levels
            num_scales: number of scales in the image pyramid
            stride: step size for sliding window (pixels)
            confidence_threshold: minimum SVM confidence to keep
            nms_threshold: IoU threshold for NMS
        """
        self.hog = hog_extractor or HOGExtractor()
        self.svm = svm_classifier
        self.scale_factor = scale_factor
        self.num_scales = num_scales
        self.stride = stride
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

    def detect(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> List[PlayerDetection]:
        """
        Detect players in a frame using multi-scale sliding window.

        Args:
            frame: BGR image (H, W, 3)
            mask: optional binary mask (255=search, 0=skip).
                  Use grass mask to avoid detecting people outside pitch.

        Returns:
            list: PlayerDetection objects after NMS
        """
        if self.svm is None or not self.svm.is_trained:
            raise RuntimeError("SVM not trained. Train or load a model first.")

        h, w = frame.shape[:2]
        win_w, win_h = self.hog.window_size
        all_detections = []

        # Multi-scale image pyramid
        for scale_idx in range(self.num_scales):
            scale = self.scale_factor ** scale_idx

            # Resize image
            new_w = int(w / scale)
            new_h = int(h / scale)

            if new_w < win_w or new_h < win_h:
                break

            scaled = cv2.resize(frame, (new_w, new_h))
            gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

            # Resize mask if provided
            if mask is not None:
                scaled_mask = cv2.resize(mask, (new_w, new_h))
            else:
                scaled_mask = None

            # Slide window across the scaled image
            for y in range(0, new_h - win_h + 1, self.stride):
                for x in range(0, new_w - win_w + 1, self.stride):

                    # Check mask — skip if window centre is in masked area
                    if scaled_mask is not None:
                        cx = x + win_w // 2
                        cy = y + win_h // 2
                        if scaled_mask[cy, cx] == 0:
                            continue

                    # Extract window
                    window = gray[y:y+win_h, x:x+win_w]

                    # Compute HOG
                    features = self.hog.extract(window)

                    # Classify
                    confidence = self.svm.predict_confidence(features)[0]

                    if confidence >= self.confidence_threshold:
                        # Scale coordinates back to original image size
                        det = PlayerDetection(
                            x=int(x * scale),
                            y=int(y * scale),
                            w=int(win_w * scale),
                            h=int(win_h * scale),
                            confidence=float(confidence),
                        )
                        all_detections.append(det)

        # Apply non-maximum suppression
        detections = self._nms(all_detections)

        return detections

    def _nms(self, detections: List[PlayerDetection]) -> List[PlayerDetection]:
        """
        Non-maximum suppression to merge overlapping detections.

        For each group of overlapping boxes, keeps only the one
        with the highest confidence score.
        """
        if len(detections) == 0:
            return []

        # Convert to arrays for efficient computation
        boxes = np.array([(d.x, d.y, d.x + d.w, d.y + d.h) for d in detections])
        scores = np.array([d.confidence for d in detections])

        # Sort by confidence (highest first)
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            # Keep the highest confidence detection
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # Compute IoU with remaining boxes
            remaining = order[1:]

            xx1 = np.maximum(boxes[i, 0], boxes[remaining, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[remaining, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[remaining, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[remaining, 3])

            inter_w = np.maximum(0, xx2 - xx1)
            inter_h = np.maximum(0, yy2 - yy1)
            intersection = inter_w * inter_h

            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_remaining = (boxes[remaining, 2] - boxes[remaining, 0]) * \
                           (boxes[remaining, 3] - boxes[remaining, 1])
            union = area_i + area_remaining - intersection

            iou = intersection / np.maximum(union, 1e-8)

            # Remove boxes with high IoU (they overlap with the kept box)
            mask = iou <= self.nms_threshold
            order = remaining[mask]

        return [detections[i] for i in keep]

    @classmethod
    def from_saved_model(
        cls,
        model_path: str,
        window_size: Tuple[int, int] = (48, 96),
        **kwargs,
    ) -> "SlidingWindowDetector":
        """
        Create detector from a saved SVM model file.

        Args:
            model_path: path to saved .pkl model
            window_size: HOG window size (must match training)

        Returns:
            SlidingWindowDetector: ready to detect
        """
        hog = HOGExtractor(window_size=window_size)
        svm = SVMClassifier.load(model_path)
        return cls(hog_extractor=hog, svm_classifier=svm, **kwargs)


def draw_player_detections(
    frame: np.ndarray,
    detections: List[PlayerDetection],
    colour: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    show_confidence: bool = True,
) -> np.ndarray:
    """
    Draw player detection bounding boxes on a frame.

    Args:
        frame: BGR image (will be modified in place)
        detections: list of PlayerDetection objects
        colour: BGR colour for boxes
        thickness: line thickness
        show_confidence: show confidence label

    Returns:
        np.ndarray: frame with detections drawn
    """
    for det in detections:
        # Bounding box
        cv2.rectangle(
            frame,
            (det.x, det.y),
            (det.x + det.w, det.y + det.h),
            colour, thickness
        )

        # Foot position marker
        cv2.circle(frame, det.foot_position, 3, colour, -1)

        # Confidence label
        if show_confidence:
            label = f"{det.confidence:.2f}"
            cv2.putText(
                frame, label,
                (det.x, det.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1
            )

    return frame


if __name__ == "__main__":
    print("=== Sliding Window Detector Test ===\n")

    # Test NMS
    print("1. Testing NMS...")
    detector = SlidingWindowDetector()

    test_dets = [
        PlayerDetection(100, 100, 50, 100, 0.9),
        PlayerDetection(110, 105, 50, 100, 0.7),   # Overlaps with first
        PlayerDetection(105, 102, 50, 100, 0.8),   # Overlaps with first
        PlayerDetection(500, 300, 50, 100, 0.85),   # Separate
        PlayerDetection(510, 305, 50, 100, 0.6),   # Overlaps with fourth
    ]

    nms_result = detector._nms(test_dets)
    print(f"   Input: {len(test_dets)} detections")
    print(f"   After NMS: {len(nms_result)} detections")
    for d in nms_result:
        print(f"     ({d.x}, {d.y}) conf={d.confidence:.2f}")

    # Test drawing
    print("\n2. Testing visualisation...")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :] = [34, 139, 34]
    draw_player_detections(frame, nms_result)
    cv2.imwrite("player_detection_test.png", frame)
    print("   Saved player_detection_test.png")

    # Test from_saved_model (without actual model)
    print("\n3. Testing PlayerDetection dataclass...")
    det = PlayerDetection(100, 200, 50, 100, 0.85)
    print(f"   Centre: {det.centre}")
    print(f"   Foot: {det.foot_position}")
    print(f"   BBox: {det.bbox}")

    print("\n=== Tests complete ===")