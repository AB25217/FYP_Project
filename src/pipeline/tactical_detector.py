
"""tactical_detector.py — Full pipeline orchestrator for tactical-camera footage.

Stages:
    1. YOLO tiled detection
    2. GAN segmentation filter (keep only on-pitch detections)
    3. Team clustering via HSV histogram + KMeans (k=5, 50-frame init)
    4. GAN-based homography estimation per frame
    5. Project foot positions to pitch coordinates
    6. Render annotated frame with bounding boxes + minimap
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.court_detection.two_gan_detector import TwoGANFieldDetector
from src.player_detection.yolo_detector import YOLODetector, Detection
from src.team_clustering.hsv_histogram import HSVFeatureExtractor
from src.team_clustering.kmeans_clustering import TeamClusterer
from src.team_clustering.tactical_assigner import TacticalAssigner
from src.camera_callibration.gan_homography_estimator import GANHomographyEstimator


@dataclass
class FrameOutput:
    frame_index: int
    detections: List[Detection] = field(default_factory=list)
    cluster_ids: List[int] = field(default_factory=list)
    team_labels: List[str] = field(default_factory=list)
    pitch_positions: List[Optional[Tuple[float, float]]] = field(default_factory=list)
    homography: Optional[np.ndarray] = None


class TacticalPipeline:
    PITCH_LENGTH = 105.0
    PITCH_WIDTH = 68.0

    def __init__(
        self,
        yolo_model: str = "yolov8s.pt",
        seg_weights: str = "src/weights/gan_weights/seg_latest_net_G.pth",
        det_weights: str = "src/weights/gan_weights/detec_latest_net_G.pth",
        siamese_weights: str = "src/weights/siamese.pth",
        pose_database: str = "src/weights/pose_database.npz",
        device: str = "cpu",
    ):
        print("Loading YOLO...")
        self.yolo = YOLODetector(model_name=yolo_model, device=device)
        print("Loading two-GAN...")
        self.gan = TwoGANFieldDetector(
            seg_weights=seg_weights, det_weights=det_weights, device=device,
        )
        print("Loading homography estimator...")
        self.homography_estimator = GANHomographyEstimator(
            gan_detector=self.gan,
            siamese_weights=siamese_weights,
            pose_database=pose_database,
            device=device,
        )
        self.hsv = HSVFeatureExtractor(
            h_bins=36, s_bins=20, v_bins=20,
            use_upper_body=True, upper_body_ratio=0.6,
        )
        self.clusterer = TeamClusterer(n_clusters=5)
        self.assigner = TacticalAssigner()
        self._initialised = False

    def initialise_from_video(
        self,
        video_path: str,
        init_frames: int = 50,
        frame_stride: int = 1,
    ) -> None:
        print(f"Initialising team clustering from first {init_frames} frames of {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(video_path)

        frame_idx = 0
        frames_used = 0
        accumulated_pitch_positions: List[np.ndarray] = []

        while frames_used < init_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            dets = self.yolo.detect(frame)
            seg_mask = self.gan.segment(frame)
            on_pitch = self._filter_by_mask(dets, seg_mask)
            bboxes = [d.bbox for d in on_pitch]
            if len(bboxes) < 3:
                frame_idx += 1
                continue

            features, valid_idx = self.hsv.extract_from_frame(frame, bboxes)
            if features.shape[0] < 3:
                frame_idx += 1
                continue

            h_result = self.homography_estimator.estimate(frame)
            if h_result.H is None:
                frame_idx += 1
                continue

            try:
                H_inv = np.linalg.inv(h_result.H)
            except np.linalg.LinAlgError:
                frame_idx += 1
                continue

            pitch_pos = np.full((features.shape[0], 2), np.nan, dtype=np.float64)
            for feat_i, vi in enumerate(valid_idx):
                d = on_pitch[vi]
                fx, fy = d.foot_position
                p = H_inv @ np.array([fx, fy, 1.0])
                if abs(p[2]) > 1e-8:
                    pitch_pos[feat_i, 0] = p[0] / p[2]
                    pitch_pos[feat_i, 1] = p[1] / p[2]

            self.clusterer.accumulate(features)
            accumulated_pitch_positions.append(pitch_pos)

            frames_used += 1
            frame_idx += 1
            print(f"  Init frame {frames_used}/{init_frames} captured "
                  f"({len(on_pitch)} on-pitch dets, {features.shape[0]} valid features)")

        cap.release()

        if frames_used == 0:
            raise RuntimeError("No usable initialisation frames found")

        print(f"Fitting KMeans on accumulated features ({frames_used} frames used)")
        result = self.clusterer.cluster_accumulated()
        all_pitch_positions = np.vstack(accumulated_pitch_positions)

        self.assigner.assign(
            cluster_labels_per_detection=result.labels,
            cluster_centres=result.centres,
            pitch_positions_per_detection=all_pitch_positions,
        )
        self._initialised = True
        print(f"Cluster labels: {self.assigner.labels}")

    def process_frame(self, frame: np.ndarray, frame_index: int = 0) -> FrameOutput:
        if not self._initialised:
            raise RuntimeError("Call initialise_from_video() before process_frame()")

        out = FrameOutput(frame_index=frame_index)

        all_detections = self.yolo.detect(frame)
        seg_mask = self.gan.segment(frame)
        on_pitch_dets = self._filter_by_mask(all_detections, seg_mask)
        out.detections = on_pitch_dets

        bboxes = [d.bbox for d in on_pitch_dets]
        if bboxes:
            features, valid_idx = self.hsv.extract_from_frame(frame, bboxes)
            if features.shape[0] > 0:
                cluster_ids = self.clusterer.predict(features)
                full_ids = [-1] * len(bboxes)
                for i, vi in enumerate(valid_idx):
                    full_ids[vi] = int(cluster_ids[i])
                out.cluster_ids = full_ids
                out.team_labels = [
                    self.assigner.labels.get(cid, "unknown") if cid >= 0 else "unknown"
                    for cid in full_ids
                ]
            else:
                out.cluster_ids = [-1] * len(bboxes)
                out.team_labels = ["unknown"] * len(bboxes)

        h_result = self.homography_estimator.estimate(frame)
        if h_result.H is not None:
            out.homography = h_result.H
            try:
                H_inv = np.linalg.inv(h_result.H)
                for d in on_pitch_dets:
                    fx, fy = d.foot_position
                    p = H_inv @ np.array([fx, fy, 1.0])
                    if abs(p[2]) < 1e-8:
                        out.pitch_positions.append(None)
                    else:
                        out.pitch_positions.append((float(p[0] / p[2]), float(p[1] / p[2])))
            except np.linalg.LinAlgError:
                out.pitch_positions = [None] * len(on_pitch_dets)
        else:
            out.pitch_positions = [None] * len(on_pitch_dets)

        return out

    def process_video(
            self,
            video_path: str,
            output_path: str,
            max_frames: int = None,
            start_frame: int = 0,
            progress_interval: int = 10,
        ) -> int:
            """Process a video and write an annotated output video.

            Args:
                video_path: input video path
                output_path: output video path (.mp4 recommended)
                max_frames: stop after this many frames (None = process all)
                start_frame: starting frame index
                progress_interval: print progress every N frames

            Returns:
                int: number of frames processed
            """
            if not self._initialised:
                raise RuntimeError("Call initialise_from_video() before process_video()")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise FileNotFoundError(video_path)

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Peek output size by running one dummy render
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, first_frame = cap.read()
            if not ret:
                cap.release()
                raise RuntimeError(f"Could not read start_frame {start_frame}")
            dummy_output = FrameOutput(frame_index=start_frame)
            sample_render = self.render(first_frame, dummy_output)
            out_h, out_w = sample_render.shape[:2]

            # Reopen at the start frame so we actually process it
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"Could not open writer for {output_path}")

            print(f"Processing video:")
            print(f"  Input:  {video_path} ({in_w}x{in_h} @ {fps:.1f}fps, {total_frames} total frames)")
            print(f"  Output: {output_path} ({out_w}x{out_h} @ {fps:.1f}fps)")
            print(f"  Frames: {start_frame} to {start_frame + (max_frames or total_frames)}")
            print()

            import time
            t_start = time.time()
            processed = 0
            frame_idx = start_frame

            while True:
                if max_frames is not None and processed >= max_frames:
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                output = self.process_frame(frame, frame_idx)
                annotated = self.render(frame, output)
                writer.write(annotated)

                processed += 1
                frame_idx += 1

                if processed % progress_interval == 0:
                    elapsed = time.time() - t_start
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = (max_frames - processed) if max_frames else (total_frames - frame_idx)
                    eta_sec = remaining / rate if rate > 0 else 0
                    print(f"  [{processed:5d}/{max_frames or total_frames}] "
                        f"{rate:.2f} fps | elapsed {elapsed/60:.1f}min | "
                        f"eta {eta_sec/60:.1f}min")

            cap.release()
            writer.release()

            elapsed = time.time() - t_start
            print(f"\nDone. {processed} frames in {elapsed/60:.1f} min "
                f"({processed/elapsed:.2f} fps average)")
            return processed

    def render(self, frame: np.ndarray, output: FrameOutput) -> np.ndarray:
        vis = frame.copy()

        for i, det in enumerate(output.detections):
            cid = output.cluster_ids[i] if i < len(output.cluster_ids) else -1
            colour = self.assigner.display_colours.get(cid, (128, 128, 128)) if cid >= 0 else (128, 128, 128)
            x, y, w, h = det.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), colour, 2)
            label = output.team_labels[i] if i < len(output.team_labels) else "?"
            cv2.putText(vis, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

        minimap = self._render_minimap(output)
        mm_h, mm_w = minimap.shape[:2]
        fh, fw = vis.shape[:2]

        canvas = np.zeros((max(fh, mm_h), fw + mm_w, 3), dtype=np.uint8)
        canvas[:fh, :fw] = vis
        mm_y = (canvas.shape[0] - mm_h) // 2
        canvas[mm_y:mm_y + mm_h, fw:fw + mm_w] = minimap
        return canvas

    def _render_minimap(
        self,
        output: FrameOutput,
        width: int = 350,
        height: int = 230,
    ) -> np.ndarray:
        mm = np.full((height, width, 3), (78, 138, 45), dtype=np.uint8)
        margin = 12
        sx = (width - 2 * margin) / self.PITCH_LENGTH
        sy = (height - 2 * margin) / self.PITCH_WIDTH

        def pitch_to_px(px, py):
            return int(margin + px * sx), int(margin + py * sy)

        cv2.rectangle(mm, pitch_to_px(0, 0),
                      pitch_to_px(self.PITCH_LENGTH, self.PITCH_WIDTH),
                      (255, 255, 255), 1)
        cv2.line(mm, pitch_to_px(self.PITCH_LENGTH / 2, 0),
                 pitch_to_px(self.PITCH_LENGTH / 2, self.PITCH_WIDTH),
                 (255, 255, 255), 1)
        cv2.circle(mm, pitch_to_px(self.PITCH_LENGTH / 2, self.PITCH_WIDTH / 2),
                   int(9.15 * sx), (255, 255, 255), 1)
        cv2.rectangle(mm, pitch_to_px(0, 13.84), pitch_to_px(16.5, 54.16),
                      (255, 255, 255), 1)
        cv2.rectangle(mm, pitch_to_px(88.5, 13.84),
                      pitch_to_px(self.PITCH_LENGTH, 54.16),
                      (255, 255, 255), 1)

        for i, pos in enumerate(output.pitch_positions):
            if pos is None:
                continue
            px, py = pos
            if not (-5 <= px <= self.PITCH_LENGTH + 5 and -5 <= py <= self.PITCH_WIDTH + 5):
                continue
            cid = output.cluster_ids[i] if i < len(output.cluster_ids) else -1
            colour = self.assigner.display_colours.get(cid, (128, 128, 128)) if cid >= 0 else (128, 128, 128)
            cx, cy = pitch_to_px(px, py)
            cv2.circle(mm, (cx, cy), 5, colour, -1)
            cv2.circle(mm, (cx, cy), 5, (255, 255, 255), 1)
        return mm

    @staticmethod
    def _filter_by_mask(
        detections: List[Detection],
        seg_mask: np.ndarray,
    ) -> List[Detection]:
        mh, mw = seg_mask.shape[:2]
        out = []
        for d in detections:
            fx, fy = d.foot_position
            fx = min(max(fx, 0), mw - 1)
            fy = min(max(fy, 0), mh - 1)
            if seg_mask[fy, fx] > 127:
                out.append(d)
        return out