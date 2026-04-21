"""
camera_pose_engine.py — Synthetic camera pose generation for sports field calibration.
Where:
    K = intrinsic matrix (focal length f)
    Q_phi = tilt rotation
    Q_theta = pan rotation
    S = base rotation (camera mounted level, looking at pitch)
    C = camera centre position in world coordinates

The engine generates synthetic edge images by:
1. Sampling random camera parameters (pan, tilt, focal length)
2. Projecting pitch template points through the camera model
3. Rendering the projected points as a binary edge image

These synthetic pairs (edge_image, camera_pose) are used to:
- Train the siamese network for feature extraction
- Populate the feature-pose retrieval database

Reference parameters from World Cup 2014 dataset:
    Camera location: mean=[52, -45, 17], std=[2, 9, 3] metres
    Pan range: [-35°, 35°]
    Tilt range: [-15°, -5°]
    Focal length: [1000, 6000] pixels
"""

import numpy as np
import cv2
import os
from dataclasses import dataclass
from typing import Tuple, Optional
from .pitch_template import PitchTemplate


@dataclass
class CameraParams:
    """Stores camera parameters for a single pose."""
    pan: float          # Pan angle in degrees (horizontal rotation)
    tilt: float         # Tilt angle in degrees (vertical rotation)
    focal_length: float # Focal length in pixels
    cx: float           # Camera x position (metres)
    cy: float           # Camera y position (metres)
    cz: float           # Camera z position (metres, height)
    roll: float = 0.0   # Roll angle in degrees (usually ~0)

    def to_dict(self) -> dict:
        return {
            "pan": self.pan, "tilt": self.tilt,
            "focal_length": self.focal_length,
            "cx": self.cx, "cy": self.cy, "cz": self.cz,
            "roll": self.roll
        }


class CameraPoseEngine:
    """
    Generates synthetic camera poses and corresponding edge images
    for training the siamese network and building the pose database.

    Usage:
        engine = CameraPoseEngine()

        # Generate a single sample
        edge_image, params = engine.generate_sample()

        # Generate a full dataset
        engine.generate_dataset(
            output_dir="data/processed/synthetic_edges",
            num_samples=90000
        )
    """

    def __init__(
        self,
        image_width: int = 1280,
        image_height: int = 720,
        output_width: int = 320,
        output_height: int = 180,
        # Camera location prior (from World Cup 2014 dataset)
        camera_location_mean: Tuple[float, float, float] = (52.0, -45.0, 17.0),
        camera_location_std: Tuple[float, float, float] = (2.0, 9.0, 3.0),
        # Pan, tilt, focal length ranges
        pan_range: Tuple[float, float] = (-35.0, 35.0),
        tilt_range: Tuple[float, float] = (-15.0, -5.0),
        focal_length_range: Tuple[float, float] = (1000.0, 6000.0),
        # Roll is nearly zero for mounted cameras
        roll_range: Tuple[float, float] = (-0.1, 0.1),
        # Base tilt of camera mount
        # +90 rotates camera to look along +Y (towards pitch from behind)
        base_tilt: float = 90.0,
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.output_width = output_width
        self.output_height = output_height

        self.camera_location_mean = np.array(camera_location_mean)
        self.camera_location_std = np.array(camera_location_std)
        self.pan_range = pan_range
        self.tilt_range = tilt_range
        self.focal_length_range = focal_length_range
        self.roll_range = roll_range
        self.base_tilt = base_tilt

        # Create pitch template
        self.pitch = PitchTemplate()
        self.pitch_lines = self.pitch.get_lines_for_drawing()

    def sample_camera_params(self) -> CameraParams:
        """
        Randomly sample camera parameters from the prior distributions.

        Returns:
            CameraParams: sampled camera parameters
        """
        # Sample camera location from Gaussian
        location = np.random.normal(
            self.camera_location_mean,
            self.camera_location_std
        )

        # Sample pan, tilt, focal length uniformly
        pan = np.random.uniform(*self.pan_range)
        tilt = np.random.uniform(*self.tilt_range)
        focal_length = np.random.uniform(*self.focal_length_range)
        roll = np.random.uniform(*self.roll_range)

        return CameraParams(
            pan=pan, tilt=tilt, focal_length=focal_length,
            cx=location[0], cy=location[1], cz=location[2],
            roll=roll
        )

    def build_projection_matrix(self, params: CameraParams) -> np.ndarray:
        """
        Build the 3x4 projection matrix P = K * [R | t].

        Uses a look-at rotation pointing at the pitch centre, then
        applies pan and tilt offsets to simulate camera movement.
        Pan rotates the view left/right, tilt adjusts up/down.

        Args:
            params: camera parameters

        Returns:
            np.ndarray: 3x4 projection matrix
        """
        f = params.focal_length
        cx_img = self.image_width / 2.0
        cy_img = self.image_height / 2.0

        # Intrinsic matrix K (square pixels, principal point at centre)
        K = np.array([
            [f,  0, cx_img],
            [0,  f, cy_img],
            [0,  0,      1]
        ])

        # Camera position in world coordinates
        C = np.array([params.cx, params.cy, params.cz])

        # Default look-at target: centre of the pitch
        target = np.array([self.pitch.length / 2, self.pitch.width / 2, 0.0])

        # Apply pan offset: shift target left/right along the pitch
        # Pan of +35° at typical distance (~80m) shifts target significantly
        pan_rad = np.deg2rad(params.pan)
        dist_to_pitch = np.linalg.norm(target - C)
        target[0] += np.tan(pan_rad) * dist_to_pitch * 0.5

        # Apply tilt offset: shift target up/down (adjusts viewing angle)
        tilt_rad = np.deg2rad(params.tilt)
        target[2] += np.tan(tilt_rad) * dist_to_pitch * 0.3

        # Build rotation matrix using look-at
        up = np.array([0.0, 0.0, 1.0])
        forward = target - C
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        true_up = np.cross(right, forward)

        # Camera convention: X=right, Y=down, Z=forward
        R = np.array([
            right,
            -true_up,
            forward
        ])

        # Apply small roll rotation
        roll_rad = np.deg2rad(params.roll)
        R_roll = np.array([
            [np.cos(roll_rad), -np.sin(roll_rad), 0],
            [np.sin(roll_rad),  np.cos(roll_rad), 0],
            [0,                 0,                 1]
        ])
        R = R_roll @ R

        # Translation: t = -R * C
        t = -R @ C

        # Projection matrix: P = K * [R | t]
        Rt = np.hstack([R, t.reshape(3, 1)])
        P = K @ Rt

        return P

    def project_points(self, points_2d: np.ndarray,
                       P: np.ndarray) -> np.ndarray:
        """
        Project 2D pitch points (x, y, z=0) through the camera model.

        Args:
            points_2d: (N, 2) array of pitch coordinates in metres
            P: 3x4 projection matrix

        Returns:
            np.ndarray: (N, 2) projected pixel coordinates
        """
        N = len(points_2d)

        # Add z=0 (pitch is flat) and homogeneous coordinate
        points_3d = np.hstack([
            points_2d,
            np.zeros((N, 1)),  # z = 0 (ground plane)
            np.ones((N, 1))    # homogeneous
        ])  # Shape: (N, 4)

        # Project: pixel_coords = P * world_coords
        projected = (P @ points_3d.T).T  # Shape: (N, 3)

        # Convert from homogeneous to pixel coordinates
        # Avoid division by zero
        w = projected[:, 2]
        valid = np.abs(w) > 1e-6

        pixel_coords = np.full((N, 2), np.nan)
        pixel_coords[valid, 0] = projected[valid, 0] / w[valid]
        pixel_coords[valid, 1] = projected[valid, 1] / w[valid]

        return pixel_coords

    def render_edge_image(self, P: np.ndarray,
                          line_thickness: int = 2) -> np.ndarray:
        """
        Render pitch markings as a binary edge image using the
        projection matrix.

        Args:
            P: 3x4 projection matrix
            line_thickness: thickness of drawn lines in pixels

        Returns:
            np.ndarray: binary edge image (output_height x output_width)
        """
        # Create blank image at full resolution
        image = np.zeros((self.image_height, self.image_width), dtype=np.uint8)

        for line_points in self.pitch_lines:
            # Project all points in this line segment
            projected = self.project_points(line_points, P)

            # Draw connected line segments
            for i in range(len(projected) - 1):
                pt1 = projected[i]
                pt2 = projected[i + 1]

                # Skip if either point is invalid (behind camera)
                if np.isnan(pt1).any() or np.isnan(pt2).any():
                    continue

                # Skip if points are outside the image (with margin)
                margin = 100
                if (pt1[0] < -margin or pt1[0] > self.image_width + margin or
                    pt1[1] < -margin or pt1[1] > self.image_height + margin):
                    continue
                if (pt2[0] < -margin or pt2[0] > self.image_width + margin or
                    pt2[1] < -margin or pt2[1] > self.image_height + margin):
                    continue

                # Draw line segment
                cv2.line(
                    image,
                    (int(round(pt1[0])), int(round(pt1[1]))),
                    (int(round(pt2[0])), int(round(pt2[1]))),
                    255,
                    line_thickness
                )

        # Resize to output dimensions (320x180 as per Chen & Little)
        edge_image = cv2.resize(
            image,
            (self.output_width, self.output_height),
            interpolation=cv2.INTER_AREA
        )

        # Binarise after resize
        _, edge_image = cv2.threshold(edge_image, 127, 255, cv2.THRESH_BINARY)

        return edge_image

    def generate_sample(self) -> Tuple[np.ndarray, CameraParams]:
        """
        Generate a single synthetic sample: edge image + camera params.

        Returns:
            tuple: (edge_image, camera_params)
        """
        params = self.sample_camera_params()
        P = self.build_projection_matrix(params)
        edge_image = self.render_edge_image(P)
        return edge_image, params

    def generate_dataset(
        self,
        output_dir: str,
        num_samples: int = 90000,
        save_images: bool = True,
        print_every: int = 5000
    ) -> None:
        """
        Generate a full synthetic dataset for siamese network training.

        Creates:
            - edge_images/ folder with PNG files
            - camera_poses.csv with matching parameters

        Args:
            output_dir: where to save the dataset
            num_samples: number of synthetic samples to generate
            save_images: whether to save edge images as PNG files
            print_every: print progress every N samples
        """
        import csv

        # Create directories
        img_dir = os.path.join(output_dir, "edge_images")
        os.makedirs(img_dir, exist_ok=True)

        # CSV file for camera parameters
        csv_path = os.path.join(output_dir, "camera_poses.csv")
        csv_fields = ["index", "pan", "tilt", "focal_length",
                       "cx", "cy", "cz", "roll"]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()

            for i in range(num_samples):
                # Generate sample
                edge_image, params = self.generate_sample()

                # Check if the image has any content (some poses might
                # point completely away from the pitch)
                if np.sum(edge_image > 0) < 100:
                    # Too few visible pixels — skip and retry
                    i -= 1
                    continue

                # Save edge image
                if save_images:
                    img_path = os.path.join(img_dir, f"edge_{i:06d}.png")
                    cv2.imwrite(img_path, edge_image)

                # Save parameters
                row = {"index": i}
                row.update(params.to_dict())
                writer.writerow(row)

                if (i + 1) % print_every == 0:
                    print(f"Generated {i + 1}/{num_samples} samples")

        print(f"\nDataset generation complete!")
        print(f"  Edge images: {img_dir}")
        print(f"  Camera poses: {csv_path}")
        print(f"  Total samples: {num_samples}")

    def get_homography(self, params: CameraParams) -> np.ndarray:
        """
        Compute the 3x3 homography matrix from the camera parameters.
        This maps pitch coordinates (metres) to image coordinates (pixels).

        Since the pitch is flat (z=0), the homography is derived from
        the projection matrix by dropping the 3rd column.

        Args:
            params: camera parameters

        Returns:
            np.ndarray: 3x3 homography matrix
        """
        P = self.build_projection_matrix(params)

        # For z=0 plane, homography H = [P[:,0], P[:,1], P[:,3]]
        H = P[:, [0, 1, 3]]

        return H


def visualise_sample(edge_image: np.ndarray, params: CameraParams,
                     save_path: str = None):
    """
    Visualise a single generated sample with parameter annotations.

    Args:
        edge_image: the binary edge image
        params: corresponding camera parameters
        save_path: if provided, save to file instead of displaying
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.imshow(edge_image, cmap="gray")
    ax.set_title(
        f"Pan: {params.pan:.1f}°  |  Tilt: {params.tilt:.1f}°  |  "
        f"Focal: {params.focal_length:.0f}px",
        fontsize=11
    )
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Pixels")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    print("=== Camera Pose Engine ===\n")

    # Create engine with default (World Cup 2014) parameters
    engine = CameraPoseEngine()

    # Generate and visualise a few samples
    print("Generating sample edge images...\n")
    for i in range(5):
        edge_image, params = engine.generate_sample()
        print(f"Sample {i+1}: pan={params.pan:+.1f}°, "
              f"tilt={params.tilt:.1f}°, "
              f"focal={params.focal_length:.0f}px, "
              f"edge_pixels={np.sum(edge_image > 0)}")

        # Save first sample as visualisation
        if i == 0:
            visualise_sample(edge_image, params, "sample_edge.png")

    # Generate a small test dataset (10 samples for quick testing)
    print("\nGenerating small test dataset...")
    engine.generate_dataset(
        output_dir="test_synthetic",
        num_samples=10,
        print_every=5
    )

    print("\nTo generate the full 90k dataset for training, run:")
    print("  engine.generate_dataset('data/processed/synthetic_edges', num_samples=90000)")