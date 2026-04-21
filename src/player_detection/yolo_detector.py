"""yolo_detector.py — YOLO-based player detection with tiled inference.

Wraps Ultralytics YOLOv8 with the tiled-inference logic from the FYP tactical
pipeline notebook. Exposes the .detect(frame, mask=None) -> list[Detection]
interface that VideoProcessor expects.

Tiling divides the frame into a 2x3 grid with 60px overlap and runs YOLO
on each tile. This catches small, distant players that the default single-pass
detector misses — the notebook showed a 25% improvement in detection count.

Usage:
    detector = YOLODetector()  # downloads yolov8s.pt on first use
    detections = detector.detect(frame)
    for d in detections:
        print(d.bbox, d.confidence)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    """A single player detection."""
    x: int
    y: int
    w: int
    h: int
    confidence: float
    class_name: str = "player"

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def centre(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def foot_position(self) -> Tuple[int, int]:
        """Bottom-centre — used for pitch projection."""
        return (self.x + self.w // 2, self.y + self.h)


class YOLODetector:
    """Player detector using YOLOv8 with optional tiled inference.

    The interface (.detect() returning list[Detection] with .bbox attribute)
    matches the contract VideoProcessor expects from a player detector, so
    this can drop in as a replacement for the existing HOG+SVM path.
    """

    def __init__(
        self,
        model_name: str = "yolov8s.pt",
        device: str = "cpu",
        conf_threshold: float = 0.25,
        nms_iou_threshold: float = 0.3,
        tiled: bool = True,
        tile_rows: int = 2,
        tile_cols: int = 3,
        tile_overlap: int = 60,
        person_class: int = 0,
    ):
        """
        Args:
            model_name: ultralytics model weight file (auto-downloads if absent).
                       Options: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', etc.
            device: 'cpu', 'cuda', or 'cuda:N'
            conf_threshold: minimum confidence to accept a detection
            nms_iou_threshold: IoU threshold for non-maximum suppression
                              (applied across tiles when tiled=True)
            tiled: if True, use tiled inference for small-object recovery
            tile_rows: number of tile rows (only used if tiled)
            tile_cols: number of tile columns (only used if tiled)
            tile_overlap: pixels of overlap between adjacent tiles
            person_class: COCO class ID for 'person' (0 in standard COCO)
        """
        # Lazy import so ultralytics is only required if this class is used
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.device = device
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.tiled = tiled
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.tile_overlap = tile_overlap
        self.person_class = person_class

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> List[Detection]:
        """Detect players in a frame.

        Args:
            frame: BGR image
            mask: optional binary mask (HxW, values 0/255). If provided, only
                  detections whose foot point lies on mask>0 are kept.

        Returns:
            list[Detection]: player detections after NMS and (optional) mask filter
        """
        raw = self._detect_tiled(frame) if self.tiled else self._detect_single(frame)
        merged = self._nms(raw)
        return self._filter_by_mask(merged, mask) if mask is not None else merged

    # ------------------------------------------------------------------
    # Detection strategies
    # ------------------------------------------------------------------

    def _detect_single(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLO once on the full frame."""
        return self._run_model(frame, offset_x=0, offset_y=0)

    def _detect_tiled(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLO on overlapping tiles and merge results (coordinates remapped)."""
        fh, fw = frame.shape[:2]
        tile_h = fh // self.tile_rows
        tile_w = fw // self.tile_cols
        overlap = self.tile_overlap

        detections: List[Detection] = []
        for r in range(self.tile_rows):
            for c in range(self.tile_cols):
                y0 = max(0, r * tile_h - overlap)
                x0 = max(0, c * tile_w - overlap)
                y1 = min(fh, (r + 1) * tile_h + overlap)
                x1 = min(fw, (c + 1) * tile_w + overlap)

                tile = frame[y0:y1, x0:x1]
                detections.extend(self._run_model(tile, offset_x=x0, offset_y=y0))

        return detections

    def _run_model(
        self,
        frame_or_tile: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> List[Detection]:
        """Run YOLO on one image/tile and return detections remapped to frame coords."""
        results = self.model(frame_or_tile, verbose=False, device=self.device)

        dets: List[Detection] = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id != self.person_class:
                continue
            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            dets.append(
                Detection(
                    x=offset_x + x1,
                    y=offset_y + y1,
                    w=x2 - x1,
                    h=y2 - y1,
                    confidence=conf,
                )
            )
        return dets

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------

    def _nms(self, detections: List[Detection]) -> List[Detection]:
        """Non-maximum suppression to merge overlapping boxes (e.g. across tiles)."""
        if not detections:
            return []

        boxes = np.array([[d.x, d.y, d.w, d.h] for d in detections])
        confs = np.array([d.confidence for d in detections])

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = x1 + boxes[:, 2]
        y2 = y1 + boxes[:, 3]
        areas = boxes[:, 2] * boxes[:, 3]

        order = confs.argsort()[::-1]
        keep: List[int] = []

        while len(order) > 0:
            i = order[0]
            keep.append(int(i))

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)

            order = order[np.where(iou <= self.nms_iou_threshold)[0] + 1]

        return [detections[k] for k in keep]

    def _filter_by_mask(
        self,
        detections: List[Detection],
        mask: np.ndarray,
    ) -> List[Detection]:
        """Keep only detections whose foot position lies on mask > 0."""
        mh, mw = mask.shape[:2]
        out: List[Detection] = []
        for d in detections:
            fx, fy = d.foot_position
            fx = min(max(fx, 0), mw - 1)
            fy = min(max(fy, 0), mh - 1)
            if mask[fy, fx] > 0:
                out.append(d)
        return out


if __name__ == "__main__":
    # Smoke test: run tiled and non-tiled on a synthetic frame
    print("Initialising YOLOv8s...")
    detector = YOLODetector(model_name="yolov8s.pt", device="cpu", tiled=True)

    # Fake frame: 720p all-grass. Won't have any players, but should at least
    # run without crashing and produce an empty result list.
    fake_frame = np.full((720, 1280, 3), [34, 139, 34], dtype=np.uint8)

    print("\nRunning tiled detection on fake grass frame...")
    dets = detector.detect(fake_frame)
    print(f"  Detections: {len(dets)} (expected 0 on grass-only frame)")

    print("\nSwitching to non-tiled mode...")
    detector.tiled = False
    dets = detector.detect(fake_frame)
    print(f"  Detections: {len(dets)} (expected 0)")

    print("\nYOLODetector smoke test OK")