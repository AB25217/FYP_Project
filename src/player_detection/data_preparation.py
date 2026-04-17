"""
data_preparation.py — Prepare training patches from annotated datasets.

Reads YOLO-format annotations and extracts:
    - Positive patches: cropped and resized player bounding boxes
    - Negative patches: random background crops with no player overlap

Also supports hard negative mining: after initial training, run the
detector on training images, collect false positives, and add them
as new negative samples for retraining.

Input: dataset in YOLO format
    images/frame_001.jpg
    labels/frame_001.txt  →  "1 0.512 0.634 0.045 0.112"
                              (class x_centre y_centre width height)
                              Class 0 = football, Class 1 = player

Output: Cropped patches in fixed size (e.g. 48x96)
    positive/player_000001.jpg
    negative/bg_000001.jpg
"""

import numpy as np
import cv2
import os
import glob
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Annotation:
    """A single YOLO-format bounding box annotation."""
    class_id: int
    x_centre: float    # Normalised (0-1)
    y_centre: float
    width: float
    height: float


class DataPreparation:
    """
    Prepare HOG training data from Roboflow YOLO-format dataset.

    Usage:
        prep = DataPreparation(window_size=(48, 96))
        prep.extract_patches(
            images_dir="data/raw/roboflow_football/train/images",
            labels_dir="data/raw/roboflow_football/train/labels",
            output_dir="data/processed/hog_training",
            negatives_per_image=5
        )
    """

    def __init__(
        self,
        window_size: Tuple[int, int] = (48, 96),
        player_class_id: int = 1,
        min_box_height: int = 20,
        iou_threshold_negative: float = 0.1,
    ):
        """
        Args:
            window_size: (width, height) of output patches
            player_class_id: YOLO class ID for players (1 in Roboflow)
            min_box_height: ignore annotations shorter than this (pixels)
            iou_threshold_negative: negative patches must have IoU < this
                                   with all player boxes
        """
        self.window_size = window_size
        self.player_class_id = player_class_id
        self.min_box_height = min_box_height
        self.iou_threshold_negative = iou_threshold_negative

    def extract_patches(
        self,
        images_dir: str,
        labels_dir: str,
        output_dir: str,
        negatives_per_image: int = 5,
        augment: bool = True,
        max_images: Optional[int] = None,
    ) -> dict:
        """
        Extract positive and negative patches from the dataset.

        Args:
            images_dir: path to images folder
            labels_dir: path to YOLO labels folder
            output_dir: where to save extracted patches
            negatives_per_image: how many negative patches per image
            augment: apply data augmentation (flip, brightness)
            max_images: limit number of images processed (for testing)

        Returns:
            dict: statistics about extracted patches
        """
        pos_dir = os.path.join(output_dir, "positive")
        neg_dir = os.path.join(output_dir, "negative")
        os.makedirs(pos_dir, exist_ok=True)
        os.makedirs(neg_dir, exist_ok=True)

        # Find image files
        image_paths = sorted(
            glob.glob(os.path.join(images_dir, "*.jpg")) +
            glob.glob(os.path.join(images_dir, "*.png"))
        )

        if max_images is not None:
            image_paths = image_paths[:max_images]

        pos_count = 0
        neg_count = 0
        skipped = 0

        for img_idx, img_path in enumerate(image_paths):
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                continue

            h, w = image.shape[:2]

            # Load corresponding label
            label_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            label_path = os.path.join(labels_dir, label_name)

            annotations = self._load_yolo_labels(label_path)

            # Convert normalised coords to pixel coords
            player_boxes = []
            for ann in annotations:
                if ann.class_id != self.player_class_id:
                    continue

                # YOLO format: centre_x, centre_y, width, height (normalised)
                bx = int((ann.x_centre - ann.width / 2) * w)
                by = int((ann.y_centre - ann.height / 2) * h)
                bw = int(ann.width * w)
                bh = int(ann.height * h)

                # Clamp to image bounds
                bx = max(0, bx)
                by = max(0, by)
                bw = min(bw, w - bx)
                bh = min(bh, h - by)

                if bh < self.min_box_height:
                    skipped += 1
                    continue

                player_boxes.append((bx, by, bw, bh))

            # Extract positive patches (players)
            for box in player_boxes:
                bx, by, bw, bh = box
                crop = image[by:by+bh, bx:bx+bw]

                if crop.size == 0:
                    continue

                # Resize to fixed window
                patch = cv2.resize(crop, self.window_size)
                save_path = os.path.join(pos_dir, f"player_{pos_count:06d}.jpg")
                cv2.imwrite(save_path, patch)
                pos_count += 1

                # Augmented: horizontal flip
                if augment:
                    flipped = cv2.flip(patch, 1)
                    save_path = os.path.join(pos_dir, f"player_{pos_count:06d}.jpg")
                    cv2.imwrite(save_path, flipped)
                    pos_count += 1

                # Augmented: brightness variation
                if augment:
                    for factor in [0.7, 1.3]:
                        bright = np.clip(patch * factor, 0, 255).astype(np.uint8)
                        save_path = os.path.join(pos_dir, f"player_{pos_count:06d}.jpg")
                        cv2.imwrite(save_path, bright)
                        pos_count += 1

            # Extract negative patches (background)
            for _ in range(negatives_per_image):
                neg_patch = self._sample_negative(
                    image, player_boxes, w, h
                )
                if neg_patch is not None:
                    save_path = os.path.join(neg_dir, f"bg_{neg_count:06d}.jpg")
                    cv2.imwrite(save_path, neg_patch)
                    neg_count += 1

            if (img_idx + 1) % 100 == 0:
                print(f"  Processed {img_idx + 1}/{len(image_paths)} images | "
                      f"pos={pos_count} neg={neg_count}")

        stats = {
            "images_processed": len(image_paths),
            "positive_patches": pos_count,
            "negative_patches": neg_count,
            "skipped_small": skipped,
            "output_dir": output_dir,
        }

        print(f"\nExtraction complete:")
        print(f"  Images: {stats['images_processed']}")
        print(f"  Positive patches: {stats['positive_patches']}")
        print(f"  Negative patches: {stats['negative_patches']}")
        print(f"  Skipped (too small): {stats['skipped_small']}")
        print(f"  Saved to: {output_dir}")

        return stats

    def extract_hard_negatives(
        self,
        detector,
        images_dir: str,
        labels_dir: str,
        output_dir: str,
        max_per_image: int = 10,
        max_images: Optional[int] = None,
    ) -> int:
        """
        Hard negative mining: run the trained detector on training images,
        collect false positives, and save them as new negative samples.

        This iterative process significantly improves SVM performance:
            1. Train initial SVM on easy negatives
            2. Run detector on training images
            3. Collect false positives (detected but no matching ground truth)
            4. Add to negative set and retrain

        Args:
            detector: trained SlidingWindowDetector instance
            images_dir: path to training images
            labels_dir: path to YOLO labels
            output_dir: where to save hard negatives
            max_per_image: max hard negatives per image
            max_images: limit images processed

        Returns:
            int: number of hard negatives extracted
        """
        hard_neg_dir = os.path.join(output_dir, "hard_negatives")
        os.makedirs(hard_neg_dir, exist_ok=True)

        image_paths = sorted(
            glob.glob(os.path.join(images_dir, "*.jpg")) +
            glob.glob(os.path.join(images_dir, "*.png"))
        )

        if max_images is not None:
            image_paths = image_paths[:max_images]

        hard_neg_count = 0

        for img_path in image_paths:
            image = cv2.imread(img_path)
            if image is None:
                continue

            h, w = image.shape[:2]

            # Load ground truth
            label_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            label_path = os.path.join(labels_dir, label_name)
            annotations = self._load_yolo_labels(label_path)

            gt_boxes = []
            for ann in annotations:
                if ann.class_id == self.player_class_id:
                    bx = int((ann.x_centre - ann.width / 2) * w)
                    by = int((ann.y_centre - ann.height / 2) * h)
                    bw = int(ann.width * w)
                    bh = int(ann.height * h)
                    gt_boxes.append((bx, by, bw, bh))

            # Run detector
            detections = detector.detect(image)

            # Find false positives
            count = 0
            for det in detections:
                is_false_positive = True
                for gt in gt_boxes:
                    if self._compute_iou(det.bbox, gt) > 0.3:
                        is_false_positive = False
                        break

                if is_false_positive and count < max_per_image:
                    dx, dy, dw, dh = det.bbox
                    dx = max(0, dx)
                    dy = max(0, dy)
                    crop = image[dy:dy+dh, dx:dx+dw]

                    if crop.size > 0:
                        patch = cv2.resize(crop, self.window_size)
                        save_path = os.path.join(
                            hard_neg_dir, f"hard_{hard_neg_count:06d}.jpg"
                        )
                        cv2.imwrite(save_path, patch)
                        hard_neg_count += 1
                        count += 1

        print(f"Extracted {hard_neg_count} hard negatives to {hard_neg_dir}")
        return hard_neg_count

    def _load_yolo_labels(self, label_path: str) -> List[Annotation]:
        """Load YOLO-format annotations from a text file."""
        annotations = []

        if not os.path.exists(label_path):
            return annotations

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    annotations.append(Annotation(
                        class_id=int(parts[0]),
                        x_centre=float(parts[1]),
                        y_centre=float(parts[2]),
                        width=float(parts[3]),
                        height=float(parts[4]),
                    ))

        return annotations

    def _sample_negative(
        self,
        image: np.ndarray,
        player_boxes: List[Tuple[int, int, int, int]],
        img_w: int,
        img_h: int,
        max_attempts: int = 50,
    ) -> Optional[np.ndarray]:
        """
        Sample a random negative patch that doesn't overlap with
        any player bounding box.
        """
        pw, ph = self.window_size

        for _ in range(max_attempts):
            # Random scale factor for variety
            scale = np.random.uniform(0.8, 2.0)
            crop_w = int(pw * scale)
            crop_h = int(ph * scale)

            if crop_w >= img_w or crop_h >= img_h:
                continue

            x = np.random.randint(0, img_w - crop_w)
            y = np.random.randint(0, img_h - crop_h)

            # Check IoU with all player boxes
            neg_box = (x, y, crop_w, crop_h)
            overlaps = False
            for pbox in player_boxes:
                if self._compute_iou(neg_box, pbox) > self.iou_threshold_negative:
                    overlaps = True
                    break

            if not overlaps:
                crop = image[y:y+crop_h, x:x+crop_w]
                patch = cv2.resize(crop, self.window_size)
                return patch

        return None

    def _compute_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
    ) -> float:
        """Compute IoU between two (x, y, w, h) boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi = max(x1, x2)
        yi = max(y1, y2)
        xf = min(x1 + w1, x2 + w2)
        yf = min(y1 + h1, y2 + h2)

        if xi >= xf or yi >= yf:
            return 0.0

        intersection = (xf - xi) * (yf - yi)
        union = w1 * h1 + w2 * h2 - intersection

        return intersection / max(union, 1e-8)


if __name__ == "__main__":
    print("=== Data Preparation Test ===\n")

    prep = DataPreparation(window_size=(48, 96))

    # Test YOLO label parsing
    print("1. Testing YOLO label parsing...")
    os.makedirs("test_labels", exist_ok=True)
    with open("test_labels/test.txt", "w") as f:
        f.write("1 0.5 0.5 0.1 0.2\n")
        f.write("1 0.3 0.4 0.08 0.15\n")
        f.write("0 0.7 0.6 0.02 0.03\n")

    anns = prep._load_yolo_labels("test_labels/test.txt")
    print(f"   Loaded {len(anns)} annotations")
    print(f"   Players: {sum(1 for a in anns if a.class_id == 1)}")
    print(f"   Balls: {sum(1 for a in anns if a.class_id == 0)}")

    # Test IoU
    print("\n2. Testing IoU computation...")
    iou1 = prep._compute_iou((0, 0, 100, 100), (50, 50, 100, 100))
    iou2 = prep._compute_iou((0, 0, 100, 100), (200, 200, 100, 100))
    print(f"   Overlapping boxes IoU: {iou1:.3f}")
    print(f"   Non-overlapping IoU: {iou2:.3f}")

    # Test negative sampling
    print("\n3. Testing negative sampling...")
    fake_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    player_boxes = [(500, 300, 50, 100), (800, 250, 40, 90)]
    neg = prep._sample_negative(fake_image, player_boxes, 1280, 720)
    if neg is not None:
        print(f"   Negative patch shape: {neg.shape}")
    else:
        print("   Could not find negative patch")

    # Cleanup
    os.remove("test_labels/test.txt")
    os.rmdir("test_labels")

    print("\n=== Tests complete ===")