"""
video_processor.py — Main pipeline: video in → analytics out.

Orchestrates all modules to process broadcast football video into
structured analytics output. This is the single entry point that
connects court detection, camera calibration, player/ball detection,
tracking, team clustering, perspective transform, and formation
detection into a coherent pipeline.

Pipeline flow per frame:
    1. Court detection: segment grass, detect edges
    2. Camera calibration: estimate/track camera pose → homography
    3. Player detection / tracking: detect every N frames, track between
    4. Ball detection: Circle Hough Transform
    5. Team clustering: HSV histograms → K-means (run periodically)
    6. Perspective transform: project foot positions to pitch coordinates
    7. Formation detection: classify team formation (run periodically)

Output per frame:
    - Player positions in pixel and pitch coordinates
    - Team labels per player
    - Ball position
    - Camera homography
    - Formation classification (when available)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
import time
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass, field

# Import all modules
from court_detection.grass_segmentation import GrassSegmenter
from court_detection.edge_detection import EdgeDetector
from camera_callibration.pose_tracker import CameraPoseTracker
from ball_detection.circle_hough import CircleHoughDetector, BallDetection
from ball_detection.background_subtraction import BackgroundSubtractor
from tracking.detection_tracker import DetectionTracker
from team_clustering.hsv_histogram import HSVFeatureExtractor
from team_clustering.kmeans_clustering import TeamClusterer
from team_clustering.team_assignment import TeamAssigner
from perspective_transform.homography import HomographyManager
from perspective_transform.pitch_projection import PitchProjector, ProjectedPlayer
from formation_detection.formation_classifier import FormationClassifier, FormationResult


@dataclass
class FrameOutput:
    """Complete output for a single processed frame."""
    frame_index: int
    timestamp: float                            # Seconds from video start

    # Player data
    player_bboxes: List[tuple]                  # (x, y, w, h) in pixels
    player_ids: List[int]                       # Persistent IDs
    player_teams: List[str]                     # "team_a", "team_b", "referee"
    player_pitch_positions: List[tuple]         # (pitch_x, pitch_y) in metres

    # Ball data
    ball_detected: bool
    ball_pixel: Optional[tuple] = None          # (x, y) in pixels
    ball_pitch: Optional[tuple] = None          # (pitch_x, pitch_y) in metres

    # Camera
    homography: Optional[np.ndarray] = None     # 3x3 homography matrix
    camera_valid: bool = False

    # Formation (None on most frames, populated periodically)
    formation: Optional[FormationResult] = None

    # Diagnostics
    processing_time_ms: float = 0.0
    is_detection_frame: bool = False


@dataclass
class PipelineConfig:
    """Configuration for the video processing pipeline."""
    # Detection/tracking
    detect_interval: int = 3                    # Run player detection every N frames
    camera_estimation_interval: int = 200       # Full camera estimation every N frames

    # Ball detection
    ball_r_min: int = 5
    ball_r_max: int = 25

    # Team clustering
    clustering_interval: int = 50               # Re-cluster every N frames
    num_clusters: int = 3                       # Teams + referee

    # Formation
    formation_window: int = 750                 # Smoothing window (frames)
    formation_update_interval: int = 125        # Re-classify every N frames

    # Processing
    process_every_n: int = 1                    # Skip frames for speed (1=every frame)
    max_frames: Optional[int] = None            # Stop after N frames (None=all)
    resize_width: Optional[int] = None          # Resize frames for speed


class VideoProcessor:
    """
    End-to-end video processing pipeline.

    Usage:
        processor = VideoProcessor()

        # Process a video file
        results = processor.process_video("match_clip.mp4")

        # Access results
        for frame_out in results:
            print(f"Frame {frame_out.frame_index}: "
                  f"{len(frame_out.player_bboxes)} players, "
                  f"formation={frame_out.formation}")

        # Or process with a callback for live visualisation
        def on_frame(frame, output):
            cv2.imshow("Pipeline", frame)
            cv2.waitKey(1)

        processor.process_video("match.mp4", callback=on_frame)
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        player_detector=None,
        camera_estimator=None,
    ):
        """
        Args:
            config: pipeline configuration
            player_detector: trained SlidingWindowDetector (or None for tracking-only mode)
            camera_estimator: callable that takes an edge image and returns a homography
                            (or None to skip camera calibration)
        """
        self.config = config or PipelineConfig()

        # Initialise modules
        self.grass_segmenter = GrassSegmenter()
        self.edge_detector = EdgeDetector()
        self.camera_tracker = CameraPoseTracker(
            estimation_interval=self.config.camera_estimation_interval
        )

        self.ball_detector = CircleHoughDetector(
            r_min=self.config.ball_r_min,
            r_max=self.config.ball_r_max,
        )
        self.bg_subtractor = BackgroundSubtractor()

        self.detection_tracker = DetectionTracker(
            detect_interval=self.config.detect_interval,
        )

        self.hsv_extractor = HSVFeatureExtractor()
        self.team_clusterer = TeamClusterer(n_clusters=self.config.num_clusters)
        self.team_assigner = TeamAssigner()

        self.homography_manager = HomographyManager()
        self.pitch_projector = PitchProjector()

        self.formation_classifier = FormationClassifier(
            smoothing_window=self.config.formation_window,
            update_interval=self.config.formation_update_interval,
        )

        # External components (must be provided for full pipeline)
        self.player_detector = player_detector
        self.camera_estimator = camera_estimator

        # State
        self._team_labels_cache: Dict[int, str] = {}
        self._frame_count = 0

    def process_video(
        self,
        video_path: str,
        callback: Optional[Callable] = None,
        output_video_path: Optional[str] = None,
    ) -> List[FrameOutput]:
        """
        Process an entire video file through the pipeline.

        Args:
            video_path: path to input video file
            callback: optional function called per frame: callback(frame, output)
            output_video_path: if provided, write annotated video to this path

        Returns:
            list: FrameOutput for each processed frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Processing: {video_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.1f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Detect interval: every {self.config.detect_interval} frames")

        # Video writer for output
        writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_w = self.config.resize_width or width
            out_h = int(height * (out_w / width)) if self.config.resize_width else height
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))

        results = []
        start_time = time.time()

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames for speed
            if frame_idx % self.config.process_every_n != 0:
                frame_idx += 1
                continue

            # Max frames limit
            if self.config.max_frames and frame_idx >= self.config.max_frames:
                break

            # Resize if configured
            if self.config.resize_width:
                scale = self.config.resize_width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            # Process single frame
            output = self.process_frame(frame, frame_idx, fps)
            results.append(output)

            # Callback for live visualisation
            if callback:
                callback(frame, output)

            # Write annotated frame
            if writer:
                annotated = self._annotate_frame(frame, output)
                writer.write(annotated)

            # Progress
            if (frame_idx + 1) % 100 == 0 or frame_idx == 0:
                elapsed = time.time() - start_time
                proc_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
                print(f"  Frame {frame_idx+1}/{total_frames} | "
                      f"{proc_fps:.1f} FPS | "
                      f"players={len(output.player_bboxes)}")

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        elapsed = time.time() - start_time
        print(f"\nDone: {len(results)} frames in {elapsed:.1f}s "
              f"({len(results)/elapsed:.1f} FPS)")

        return results

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        fps: float = 25.0,
    ) -> FrameOutput:
        """
        Process a single frame through the full pipeline.

        Args:
            frame: BGR image
            frame_index: current frame number
            fps: video FPS (for timestamp calculation)

        Returns:
            FrameOutput: complete frame analysis
        """
        t_start = time.time()
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        timestamp = frame_index / fps

        # ─── Step 1: Court Detection ───
        grass_mask = self.grass_segmenter.segment(frame)

        # Auto-calibrate grass on first frame
        if frame_index == 0:
            self.grass_segmenter.auto_calibrate(frame)
            grass_mask = self.grass_segmenter.segment(frame)

        # ─── Step 2: Camera Calibration ───
        homography = None
        camera_valid = False

        if self.camera_estimator is not None:
            edges = self.edge_detector.detect(frame, mask=grass_mask)

            if self.camera_tracker.needs_estimation():
                H = self.camera_estimator(edges)
                if H is not None:
                    state = self.camera_tracker.set_estimation(H, edges, frame_index)
                    homography = state.homography
                    camera_valid = state.tracking_valid
            else:
                state = self.camera_tracker.track(edges, frame_index)
                homography = state.homography
                camera_valid = state.tracking_valid

            if homography is not None:
                self.homography_manager.set_homography(homography)
                self.pitch_projector.set_homography(homography)

        # ─── Step 3: Player Detection / Tracking ───
        is_detection_frame = self.detection_tracker.is_detection_frame(frame_index)

        if is_detection_frame and self.player_detector is not None:
            detections = self.player_detector.detect(frame, mask=grass_mask)
            bboxes = [d.bbox for d in detections]
            track_result = self.detection_tracker.process_detection(
                gray, bboxes, frame_index
            )
        elif is_detection_frame:
            # No player detector — skip detection
            track_result = self.detection_tracker.process_tracking(gray, frame_index)
        else:
            track_result = self.detection_tracker.process_tracking(gray, frame_index)

        # Extract player data from tracks
        player_bboxes = [t.bbox for t in track_result.tracked_objects if t.label == "player"]
        player_ids = [t.object_id for t in track_result.tracked_objects if t.label == "player"]
        foot_positions = [t.foot_position for t in track_result.tracked_objects if t.label == "player"]

        # ─── Step 4: Ball Detection ───
        bg_mask = self.bg_subtractor.apply(frame, grass_mask=grass_mask)
        combined_mask = cv2.bitwise_and(grass_mask, bg_mask) if self.bg_subtractor.is_ready() else grass_mask
        ball_result = self.ball_detector.detect(frame, mask=combined_mask)

        ball_pixel = None
        ball_pitch = None
        if ball_result.detected:
            ball_pixel = ball_result.centre

        # ─── Step 5: Team Clustering ───
        player_teams = []
        if frame_index % self.config.clustering_interval == 0 and len(player_bboxes) >= 3:
            features, valid_idx = self.hsv_extractor.extract_from_frame(frame, player_bboxes)
            if len(features) >= 3:
                cluster_result = self.team_clusterer.cluster(features, valid_idx)

                # Build pitch positions for team assignment (if camera available)
                if camera_valid and len(foot_positions) > 0:
                    pixel_pts = np.array(foot_positions, dtype=np.float64)
                    pitch_pts = self.homography_manager.pixel_to_pitch_batch(pixel_pts)

                    team_labels = self.team_assigner.assign_from_positions(
                        cluster_result, pitch_pts
                    )

                    # Cache team labels by player ID
                    for i, pid in enumerate(player_ids):
                        if i < len(valid_idx) and valid_idx[i] < len(team_labels.team_labels):
                            self._team_labels_cache[pid] = team_labels.team_labels[valid_idx[i]]
                else:
                    # Without camera, use cluster-based assignment
                    team_labels = self.team_assigner.assign_from_previous(
                        cluster_result, features
                    )
                    for i, pid in enumerate(player_ids):
                        if i < len(team_labels.team_labels):
                            self._team_labels_cache[pid] = team_labels.team_labels[i]

        # Look up cached team labels
        for pid in player_ids:
            player_teams.append(self._team_labels_cache.get(pid, "unknown"))

        # ─── Step 6: Perspective Transform ───
        player_pitch_positions = []
        if camera_valid and len(foot_positions) > 0:
            projected = self.pitch_projector.project_players(
                foot_positions, player_ids, player_teams
            )
            player_pitch_positions = [
                (p.pitch_x, p.pitch_y) if p.is_valid else (None, None)
                for p in projected
            ]

            # Project ball
            if ball_pixel is not None:
                ball_pitch = self.pitch_projector.project_ball(*ball_pixel)
        else:
            player_pitch_positions = [(None, None)] * len(player_ids)

        # ─── Step 7: Formation Detection ───
        formation_result = None
        valid_positions = [
            (px, py) for (px, py) in player_pitch_positions
            if px is not None
        ]
        valid_ids_for_formation = [
            pid for pid, (px, py) in zip(player_ids, player_pitch_positions)
            if px is not None
        ]
        valid_teams_for_formation = [
            t for t, (px, py) in zip(player_teams, player_pitch_positions)
            if px is not None
        ]

        if len(valid_positions) >= 5:
            pos_array = np.array(valid_positions, dtype=np.float64)
            formation_result = self.formation_classifier.update(
                pos_array, valid_ids_for_formation, valid_teams_for_formation
            )

        # ─── Build output ───
        processing_time = (time.time() - t_start) * 1000

        return FrameOutput(
            frame_index=frame_index,
            timestamp=timestamp,
            player_bboxes=player_bboxes,
            player_ids=player_ids,
            player_teams=player_teams,
            player_pitch_positions=player_pitch_positions,
            ball_detected=ball_result.detected,
            ball_pixel=ball_pixel,
            ball_pitch=ball_pitch,
            homography=homography,
            camera_valid=camera_valid,
            formation=formation_result,
            processing_time_ms=processing_time,
            is_detection_frame=is_detection_frame,
        )

    def _annotate_frame(
        self,
        frame: np.ndarray,
        output: FrameOutput,
    ) -> np.ndarray:
        """Draw all detections and info on the frame."""
        vis = frame.copy()

        team_colours = {
            "team_a": (0, 0, 255),
            "team_b": (255, 0, 0),
            "referee": (0, 255, 255),
            "unknown": (200, 200, 200),
        }

        # Draw player bounding boxes
        for i, bbox in enumerate(output.player_bboxes):
            x, y, w, h = bbox
            team = output.player_teams[i] if i < len(output.player_teams) else "unknown"
            colour = team_colours.get(team, (200, 200, 200))

            cv2.rectangle(vis, (x, y), (x + w, y + h), colour, 2)

            pid = output.player_ids[i] if i < len(output.player_ids) else -1
            label = f"#{pid}"
            cv2.putText(vis, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

        # Draw ball
        if output.ball_detected and output.ball_pixel:
            bx, by = output.ball_pixel
            cv2.circle(vis, (bx, by), 8, (0, 255, 255), 2)
            cv2.circle(vis, (bx, by), 2, (0, 255, 255), -1)

        # Info overlay
        info_lines = [
            f"Frame: {output.frame_index}",
            f"Players: {len(output.player_bboxes)}",
            f"Ball: {'Yes' if output.ball_detected else 'No'}",
            f"Camera: {'OK' if output.camera_valid else 'N/A'}",
            f"Time: {output.processing_time_ms:.0f}ms",
        ]

        if output.formation:
            info_lines.append(f"A: {output.formation.team_a_formation}")
            info_lines.append(f"B: {output.formation.team_b_formation}")

        for j, line in enumerate(info_lines):
            cv2.putText(vis, line, (10, 25 + j * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return vis

    def reset(self) -> None:
        """Reset all pipeline state for a new video."""
        self.camera_tracker.reset()
        self.ball_detector.reset_tracking()
        self.bg_subtractor.reset()
        self.detection_tracker.reset()
        self.formation_classifier.reset()
        self._team_labels_cache.clear()
        self._frame_count = 0

    def get_pipeline_stats(self) -> dict:
        """Return diagnostic statistics from all modules."""
        return {
            "camera": self.camera_tracker.get_stats(),
            "tracking": self.detection_tracker.get_stats(),
            "formation": {
                "last_result": str(self.formation_classifier.get_last_result())
            },
        }


if __name__ == "__main__":
    print("=== Video Processor Test ===\n")

    # Test with synthetic frames (no trained models)
    print("1. Testing pipeline initialisation...")
    config = PipelineConfig(
        detect_interval=3,
        clustering_interval=10,
        formation_update_interval=5,
        formation_window=10,
    )
    processor = VideoProcessor(config=config)
    print("   Pipeline initialised (no trained models — tracking-only mode)")

    # Process synthetic frames
    print("\n2. Processing 15 synthetic frames...")

    for i in range(15):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:, :] = [34, 139, 34]

        # Add some player-like rectangles
        for px in [200, 400, 600, 800, 1000]:
            py = 300 + (i * 3) % 50
            cv2.rectangle(frame, (px, py), (px + 30, py + 80),
                         (0, 0, 200), -1)

        output = processor.process_frame(frame, i)

        if i % 5 == 0:
            print(f"   Frame {i}: players={len(output.player_bboxes)} "
                  f"ball={'yes' if output.ball_detected else 'no'} "
                  f"time={output.processing_time_ms:.0f}ms")

    # Stats
    print("\n3. Pipeline statistics:")
    stats = processor.get_pipeline_stats()
    print(f"   Tracking: {stats['tracking']}")

    # Reset
    print("\n4. Testing reset...")
    processor.reset()
    print("   Pipeline reset")

    print("\n=== Tests complete ===")
    print("\nTo process a real video:")
    print("  processor = VideoProcessor(")
    print("      player_detector=SlidingWindowDetector.from_saved_model('weights/svm_player.pkl'),")
    print("      camera_estimator=my_camera_function")
    print("  )")
    print("  results = processor.process_video('match_clip.mp4')")