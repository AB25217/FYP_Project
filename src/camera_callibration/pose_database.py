"""
pose_database.py — Feature-pose database for camera calibration retrieval.

Takes a trained siamese network and a synthetic edge image dataset,
extracts features from all images, and builds a searchable database.

Given a new edge image (from a real broadcast frame), the database:
1. Extracts its 16-dim feature using the siamese network
2. Finds the closest matching feature in the database (KD-tree)
3. Returns the camera pose parameters that generated the match


Usage:
    # Build the database (run once after training siamese)
    db = PoseDatabase()
    db.build_from_synthetic(
        model_path="weights/siamese.pth",
        data_dir="data/processed/synthetic_edges"
    )
    db.save("weights/pose_database.npz")

    # Query at inference time
    db = PoseDatabase.load("weights/pose_database.npz")
    pose = db.query(edge_image)
"""

import numpy as np
import cv2
import os
import csv
import torch
from scipy.spatial import KDTree
from dataclasses import dataclass
from typing import Tuple, Optional, List

# Import from our modules
from .siamese_network import SiameseNetwork, load_trained_model


@dataclass
class RetrievedPose:
    """Result of a database query — the retrieved camera pose."""
    pan: float
    tilt: float
    focal_length: float
    cx: float
    cy: float
    cz: float
    roll: float
    distance: float         # Feature space distance to query
    matched_index: int      # Index of the matched synthetic image

    def to_homography_params(self) -> dict:
        """Return parameters needed to build a homography matrix."""
        return {
            "pan": self.pan, "tilt": self.tilt,
            "focal_length": self.focal_length,
            "cx": self.cx, "cy": self.cy, "cz": self.cz,
            "roll": self.roll
        }


class PoseDatabase:
    """
    Feature-pose database for nearest-neighbour camera pose retrieval.

    Stores 16-dimensional siamese features alongside camera parameters
    for all synthetic edge images. Uses a KD-tree for fast lookup.

    At query time:
        1. Extract feature from input edge image
        2. Find nearest neighbour in KD-tree
        3. Return corresponding camera pose

    The database can optionally return the top-K matches for
    more robust retrieval or for pose averaging.
    """

    def __init__(self):
        self.features = None        # (N, 16) numpy array
        self.poses = None           # list of dicts with camera params
        self.kdtree = None          # scipy KDTree for fast search
        self.model = None           # siamese network for feature extraction
        self.device = None
        self.num_entries = 0

    def build_from_synthetic(
        self,
        model_path: str,
        data_dir: str,
        feature_dim: int = 16,
        batch_size: int = 64,
        device: str = None,
        print_every: int = 5000,
    ) -> None:
        """
        Build the database by extracting features from all synthetic
        edge images using the trained siamese network.

        Args:
            model_path: path to trained siamese .pth file
            data_dir: path to synthetic dataset (edge_images/ + camera_poses.csv)
            feature_dim: dimension of feature vectors (must match training)
            batch_size: batch size for feature extraction
            device: 'cuda', 'cpu', or None (auto-detect)
            print_every: print progress every N images
        """
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load trained model
        print(f"Loading siamese model from {model_path}...")
        self.model = load_trained_model(model_path, feature_dim, device)

        # Load camera poses from CSV
        print(f"Loading camera poses from {data_dir}...")
        csv_path = os.path.join(data_dir, "camera_poses.csv")
        img_dir = os.path.join(data_dir, "edge_images")

        image_paths = []
        self.poses = []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["index"])
                image_paths.append(
                    os.path.join(img_dir, f"edge_{idx:06d}.png")
                )
                self.poses.append({
                    "pan": float(row["pan"]),
                    "tilt": float(row["tilt"]),
                    "focal_length": float(row["focal_length"]),
                    "cx": float(row["cx"]),
                    "cy": float(row["cy"]),
                    "cz": float(row["cz"]),
                    "roll": float(row["roll"]),
                })

        self.num_entries = len(self.poses)
        print(f"Found {self.num_entries} synthetic samples")

        # Extract features in batches
        print(f"Extracting features (batch_size={batch_size})...")
        all_features = []

        self.model.eval()
        with torch.no_grad():
            for start_idx in range(0, self.num_entries, batch_size):
                end_idx = min(start_idx + batch_size, self.num_entries)
                batch_paths = image_paths[start_idx:end_idx]

                # Load and preprocess batch
                batch_images = []
                for path in batch_paths:
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        # Create blank image if file is missing
                        img = np.zeros((180, 320), dtype=np.uint8)
                    img = img.astype(np.float32) / 255.0
                    batch_images.append(img)

                # Stack into tensor: (batch, 1, 180, 320)
                batch_tensor = torch.from_numpy(
                    np.array(batch_images)
                ).unsqueeze(1).to(device)

                # Extract features
                features = self.model.extract_feature(batch_tensor)
                all_features.append(features.cpu().numpy())

                if (end_idx) % print_every == 0 or end_idx == self.num_entries:
                    print(f"  Processed {end_idx}/{self.num_entries} images")

        # Stack all features
        self.features = np.vstack(all_features)
        print(f"Feature matrix shape: {self.features.shape}")

        # Build KD-tree for fast nearest-neighbour search
        print("Building KD-tree...")
        self.kdtree = KDTree(self.features)
        print("Database build complete!")

    def query(
        self,
        edge_image: np.ndarray,
        k: int = 1,
    ) -> RetrievedPose:
        """
        Query the database with an edge image to retrieve the
        closest matching camera pose.

        Args:
            edge_image: binary edge image, (180, 320) uint8 or float
            k: number of nearest neighbours to consider
               (k=1 returns single best match, k>1 averages top-k poses)

        Returns:
            RetrievedPose: the retrieved camera pose with distance info
        """
        if self.kdtree is None:
            raise RuntimeError("Database not built. Call build_from_synthetic() or load() first.")

        # Extract feature from query image
        feature = self._extract_single_feature(edge_image)

        # Query KD-tree
        distances, indices = self.kdtree.query(feature, k=k)

        if k == 1:
            # Single nearest neighbour
            idx = int(indices)
            dist = float(distances)
            pose = self.poses[idx]

            return RetrievedPose(
                pan=pose["pan"],
                tilt=pose["tilt"],
                focal_length=pose["focal_length"],
                cx=pose["cx"],
                cy=pose["cy"],
                cz=pose["cz"],
                roll=pose["roll"],
                distance=dist,
                matched_index=idx,
            )
        else:
            # Average top-k poses (weighted by inverse distance)
            weights = 1.0 / (np.array(distances) + 1e-8)
            weights = weights / weights.sum()

            avg_pose = {}
            for key in ["pan", "tilt", "focal_length", "cx", "cy", "cz", "roll"]:
                values = [self.poses[int(idx)][key] for idx in indices]
                avg_pose[key] = float(np.average(values, weights=weights))

            return RetrievedPose(
                pan=avg_pose["pan"],
                tilt=avg_pose["tilt"],
                focal_length=avg_pose["focal_length"],
                cx=avg_pose["cx"],
                cy=avg_pose["cy"],
                cz=avg_pose["cz"],
                roll=avg_pose["roll"],
                distance=float(distances[0]),
                matched_index=int(indices[0]),
            )

    def query_batch(
        self,
        edge_images: List[np.ndarray],
        k: int = 1,
    ) -> List[RetrievedPose]:
        """
        Query multiple edge images at once (more efficient than
        calling query() in a loop).

        Args:
            edge_images: list of edge images
            k: number of nearest neighbours

        Returns:
            list: RetrievedPose for each query image
        """
        results = []
        for img in edge_images:
            results.append(self.query(img, k=k))
        return results

    def _extract_single_feature(self, edge_image: np.ndarray) -> np.ndarray:
        """
        Extract a 16-dim feature from a single edge image.

        Args:
            edge_image: (180, 320) uint8 or float array

        Returns:
            np.ndarray: (16,) feature vector
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call build_from_synthetic() or load() first.")

        # Preprocess
        if edge_image.dtype == np.uint8:
            img = edge_image.astype(np.float32) / 255.0
        else:
            img = edge_image.astype(np.float32)

        # Ensure correct size
        if img.shape != (180, 320):
            img = cv2.resize(img, (320, 180))

        # To tensor: (1, 1, 180, 320)
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)

        # Extract feature
        self.model.eval()
        with torch.no_grad():
            feature = self.model.extract_feature(tensor)

        return feature.cpu().numpy().flatten()

    def save(self, path: str) -> None:
        """
        Save the database to a .npz file.
        Saves features and poses. The KD-tree is rebuilt on load.
        The siamese model must be loaded separately.

        Args:
            path: output path (e.g. 'weights/pose_database.npz')
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        # Convert poses list of dicts to structured arrays
        pose_keys = ["pan", "tilt", "focal_length", "cx", "cy", "cz", "roll"]
        pose_array = np.array([
            [p[k] for k in pose_keys] for p in self.poses
        ])

        np.savez(
            path,
            features=self.features,
            pose_array=pose_array,
            pose_keys=pose_keys,
        )
        print(f"Database saved to {path}")
        print(f"  Entries: {self.num_entries}")
        print(f"  File size: {os.path.getsize(path) / 1024:.1f} KB")

    @classmethod
    def load(
        cls,
        database_path: str,
        model_path: str,
        feature_dim: int = 16,
        device: str = None,
    ) -> "PoseDatabase":
        """
        Load a saved database and its corresponding siamese model.

        Args:
            database_path: path to the .npz database file
            model_path: path to the trained siamese .pth file
            feature_dim: must match training dimension
            device: 'cuda', 'cpu', or None (auto-detect)

        Returns:
            PoseDatabase: ready to query
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        db = cls()
        db.device = device

# Load features and poses
        print(f"Loading database from {database_path}...")
        data = np.load(database_path, allow_pickle=True)
        keys = set(data.files)

        db.features = data["features"]

        # Detect schema — two formats are supported:
        #   (a) Repo format:       features + pose_array + pose_keys (7 pose dims)
        #   (b) Notebook format:   features + pans + tilts + focals  (3 pose dims)
        if "pose_array" in keys and "pose_keys" in keys:
            # Repo format — full 7-dim poses
            pose_array = data["pose_array"]
            pose_keys = list(data["pose_keys"])
            db.poses = []
            for row in pose_array:
                pose = {k: float(v) for k, v in zip(pose_keys, row)}
                db.poses.append(pose)
        elif all(k in keys for k in ("pans", "tilts", "focals")):
            # Notebook format — only 3 pose dims saved
            # Fill cx/cy/cz/roll with the prior means used during training
            # (from CameraPoseEngine defaults: World Cup 2014 camera distribution)
            pans = data["pans"]
            tilts = data["tilts"]
            focals = data["focals"]
            db.poses = []
            for pan, tilt, focal in zip(pans, tilts, focals):
                db.poses.append({
                    "pan": float(pan),
                    "tilt": float(tilt),
                    "focal_length": float(focal),
                    "cx": 52.0,   # World Cup 2014 mean camera x (metres)
                    "cy": -45.0,  # World Cup 2014 mean camera y (metres)
                    "cz": 17.0,   # World Cup 2014 mean camera height (metres)
                    "roll": 0.0,  # roll is negligible in practice
                })
            print("  (notebook schema detected — cx/cy/cz/roll filled with training priors)")
        else:
            raise KeyError(
                f"Unrecognised pose_database schema. Keys present: {sorted(keys)}. "
                f"Expected either ('pose_array' + 'pose_keys') or ('pans' + 'tilts' + 'focals')."
            )

        db.num_entries = len(db.poses)

        # Rebuild KD-tree
        print("Rebuilding KD-tree...")
        db.kdtree = KDTree(db.features)

        # Load siamese model
        print(f"Loading siamese model from {model_path}...")
        db.model = load_trained_model(model_path, feature_dim, device)

        print(f"Database loaded: {db.num_entries} entries, "
              f"feature dim: {db.features.shape[1]}")

        return db

    def evaluate_retrieval(
        self,
        test_data_dir: str,
        num_tests: int = 100,
    ) -> dict:
        """
        Evaluate retrieval accuracy by querying synthetic edge images
        with known ground truth poses and measuring the error.

        This tests the full pipeline: edge image → siamese feature →
        KD-tree lookup → retrieved pose vs actual pose.

        Args:
            test_data_dir: path to a synthetic dataset (can be same as training)
            num_tests: number of test queries

        Returns:
            dict: error statistics (mean/median/std for pan, tilt, focal)
        """
        # Load test poses and images
        csv_path = os.path.join(test_data_dir, "camera_poses.csv")
        img_dir = os.path.join(test_data_dir, "edge_images")

        test_poses = []
        test_paths = []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["index"])
                test_poses.append({
                    "pan": float(row["pan"]),
                    "tilt": float(row["tilt"]),
                    "focal_length": float(row["focal_length"]),
                })
                test_paths.append(
                    os.path.join(img_dir, f"edge_{idx:06d}.png")
                )

        # Sample random test indices
        num_tests = min(num_tests, len(test_poses))
        test_indices = np.random.choice(len(test_poses), num_tests, replace=False)

        # Query each test image
        pan_errors = []
        tilt_errors = []
        focal_errors = []

        for i, idx in enumerate(test_indices):
            # Load test edge image
            img = cv2.imread(test_paths[idx], cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Query database
            result = self.query(img, k=1)

            # Compute errors
            gt = test_poses[idx]
            pan_errors.append(abs(result.pan - gt["pan"]))
            tilt_errors.append(abs(result.tilt - gt["tilt"]))
            focal_errors.append(abs(result.focal_length - gt["focal_length"]))

        # Compute statistics
        stats = {
            "num_tests": len(pan_errors),
            "pan_error_mean": float(np.mean(pan_errors)),
            "pan_error_median": float(np.median(pan_errors)),
            "pan_error_std": float(np.std(pan_errors)),
            "tilt_error_mean": float(np.mean(tilt_errors)),
            "tilt_error_median": float(np.median(tilt_errors)),
            "tilt_error_std": float(np.std(tilt_errors)),
            "focal_error_mean": float(np.mean(focal_errors)),
            "focal_error_median": float(np.median(focal_errors)),
            "focal_error_std": float(np.std(focal_errors)),
        }

        print(f"\n=== Retrieval Evaluation ({stats['num_tests']} queries) ===")
        print(f"Pan error:   mean={stats['pan_error_mean']:.2f}°, "
              f"median={stats['pan_error_median']:.2f}°")
        print(f"Tilt error:  mean={stats['tilt_error_mean']:.2f}°, "
              f"median={stats['tilt_error_median']:.2f}°")
        print(f"Focal error: mean={stats['focal_error_mean']:.0f}px, "
              f"median={stats['focal_error_median']:.0f}px")

        return stats


if __name__ == "__main__":
    print("=== Pose Database Test ===\n")

    # Check if we have test data and a trained model
    test_data = "test_synthetic_200"
    test_model = "test_siamese.pth"

    if not os.path.exists(test_data):
        print(f"No test data at {test_data}")
        print("Run camera_pose_engine.py first to generate synthetic data,")
        print("then siamese_network.py to train the model.")
        exit(1)

    if not os.path.exists(test_model):
        print(f"No trained model at {test_model}")
        print("Run siamese_network.py first to train the model.")
        exit(1)

    # Test 1: Build database from synthetic data
    print("1. Building database from synthetic data...")
    db = PoseDatabase()
    db.build_from_synthetic(
        model_path=test_model,
        data_dir=test_data,
        batch_size=16,
        print_every=100,
    )
    print(f"   Database entries: {db.num_entries}")
    print(f"   Feature matrix: {db.features.shape}")

    # Test 2: Save and reload
    print("\n2. Testing save/load...")
    db.save("test_pose_db.npz")

    db_loaded = PoseDatabase.load(
        database_path="test_pose_db.npz",
        model_path=test_model,
    )
    print(f"   Loaded entries: {db_loaded.num_entries}")

    # Test 3: Query with a synthetic edge image
    print("\n3. Testing query...")
    img_dir = os.path.join(test_data, "edge_images")
    test_img_path = os.path.join(img_dir, "edge_000000.png")
    test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)

    if test_img is not None:
        result = db_loaded.query(test_img, k=1)
        print(f"   Query result:")
        print(f"     Pan: {result.pan:.1f}°")
        print(f"     Tilt: {result.tilt:.1f}°")
        print(f"     Focal: {result.focal_length:.0f}px")
        print(f"     Distance: {result.distance:.4f}")
        print(f"     Matched index: {result.matched_index}")

        # Test top-K query
        result_k5 = db_loaded.query(test_img, k=5)
        print(f"\n   Top-5 averaged result:")
        print(f"     Pan: {result_k5.pan:.1f}°")
        print(f"     Tilt: {result_k5.tilt:.1f}°")
        print(f"     Focal: {result_k5.focal_length:.0f}px")
    else:
        print(f"   Could not load test image from {test_img_path}")

    # Test 4: Evaluate retrieval accuracy
    print("\n4. Evaluating retrieval accuracy...")
    stats = db_loaded.evaluate_retrieval(
        test_data_dir=test_data,
        num_tests=20,
    )

    print("\n=== All tests passed ===")
    print("\nFull workflow:")
    print("  1. Generate 90k synthetic data: camera_pose_engine.py")
    print("  2. Train siamese on Colab: siamese_network.py")
    print("  3. Build database: pose_database.py")
    print("  4. Query with real edge images at inference time")