"""gan_homography_estimator.py — GAN-based camera calibration.

Given a frame, produces a pitch-to-image homography using the Chen & Little
pipeline: GAN line detection -> siamese encoding -> pose database retrieval
-> distance-transform refinement -> homography.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .camera_pose_engine import CameraParams, CameraPoseEngine
from .pose_database import PoseDatabase


@dataclass
class HomographyResult:
    H: Optional[np.ndarray]
    retrieved_pose: Optional[CameraParams]
    retrieval_distance: float
    refinement_ok: bool
    refinement_correlation: float


class GANHomographyEstimator:
    _ECC_MAX_ITER = 50
    _ECC_EPSILON = 1e-4

    def __init__(
        self,
        gan_detector,
        siamese_weights: str,
        pose_database: str,
        device: str = "cpu",
        feature_dim: int = 16,
        image_width: int = 1280,
        image_height: int = 720,
    ):
        self.gan = gan_detector
        self.db = PoseDatabase.load(
            database_path=pose_database,
            model_path=siamese_weights,
            feature_dim=feature_dim,
            device=device,
        )
        self.pose_engine = CameraPoseEngine(
            image_width=image_width,
            image_height=image_height,
        )

    def estimate(self, frame: np.ndarray) -> HomographyResult:
        # Step 1: GAN line detection, resized for siamese input (320x180)
        lines = self.gan.detect_lines(frame, resize_to_input=False)
        lines_320 = cv2.resize(lines, (320, 180))
        _, lines_bin = cv2.threshold(lines_320, 50, 255, cv2.THRESH_BINARY)

        # Step 2-3: encode + query database
        retrieved = self.db.query(lines_bin, k=1)
        pose = CameraParams(
            pan=retrieved.pan,
            tilt=retrieved.tilt,
            focal_length=retrieved.focal_length,
            cx=retrieved.cx,
            cy=retrieved.cy,
            cz=retrieved.cz,
            roll=retrieved.roll,
        )

        # Step 4: render synthetic edge image from retrieved pose
        P_retrieved = self.pose_engine.build_projection_matrix(pose)
        synth_edges = self.pose_engine.render_edge_image(P_retrieved)

        # Step 5: distance transforms
        dt_query = self._distance_image(lines_bin)
        dt_synth = self._distance_image(synth_edges)

        # Step 6: ECC refinement
        warp_matrix = np.eye(3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self._ECC_MAX_ITER,
            self._ECC_EPSILON,
        )
        refinement_ok = False
        correlation = 0.0
        try:
            correlation, warp_matrix = cv2.findTransformECC(
                dt_synth,
                dt_query,
                warp_matrix,
                cv2.MOTION_HOMOGRAPHY,
                criteria,
            )
            refinement_ok = True
        except cv2.error:
            pass

        # Step 7: compose final homography
        H_retrieved_full = self.pose_engine.get_homography(pose)

        scale_down = np.array([
            [320.0 / self.pose_engine.image_width,  0, 0],
            [0, 180.0 / self.pose_engine.image_height, 0],
            [0, 0, 1],
        ], dtype=np.float64)
        H_retrieved_small = scale_down @ H_retrieved_full

        H_query_small = warp_matrix.astype(np.float64) @ H_retrieved_small

        fh, fw = frame.shape[:2]
        scale_up = np.array([
            [fw / 320.0, 0, 0],
            [0, fh / 180.0, 0],
            [0, 0, 1],
        ], dtype=np.float64)
        H_final = scale_up @ H_query_small

        return HomographyResult(
            H=H_final,
            retrieved_pose=pose,
            retrieval_distance=retrieved.distance,
            refinement_ok=refinement_ok,
            refinement_correlation=float(correlation),
        )

    @staticmethod
    def _distance_image(binary_edges: np.ndarray, truncate: float = 20.0) -> np.ndarray:
        inverted = cv2.bitwise_not(binary_edges)
        dt = cv2.distanceTransform(inverted, cv2.DIST_L2, 3)
        dt = np.clip(dt, 0, truncate)
        dt = truncate - dt
        return dt.astype(np.float32)