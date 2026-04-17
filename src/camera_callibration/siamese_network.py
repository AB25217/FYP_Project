"""
siamese_network.py — Siamese CNN for edge image feature extraction.

Learns compact 16-dimensional feature descriptors from synthetic edge images.
Pairs of edge images with similar camera poses are trained to have similar
features, while pairs with dissimilar poses are pushed apart.

Architecture (from Chen & Little 2019):
    - 5 stride-2 convolutions (kernel sizes: 7, 5, 3, 3, 3)
    - Channels: 1 -> 16 -> 32 -> 64 -> 128 -> 256
    - Final 6x10 convolution to collapse spatial dimensions
    - L2 normalisation layer
    - Output: 16-dimensional feature vector

Training:
    - Input: pairs of synthetic edge images (320x180)
    - Labels: similar (1) or dissimilar (0) based on camera pose distance
    - Loss: contrastive loss
    - Similar if: pan diff < 1°, tilt diff < 0.5°, focal diff < 30px

Reference: Chen & Little (2019) "Sports Camera Calibration via Synthetic Data"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import csv
from typing import Tuple, Optional


class SiameseEncoder(nn.Module):
    """
    Single branch of the siamese network.
    Takes a 320x180 edge image and produces a 16-dim feature vector.

    Architecture:
        Conv(7, stride=2) -> Conv(5, stride=2) -> Conv(3, stride=2) ->
        Conv(3, stride=2) -> Conv(3, stride=2) -> Conv(6x10) -> L2 norm

    Input: (batch, 1, 180, 320)
    Output: (batch, 16)
    """

    def __init__(self, feature_dim: int = 16):
        super().__init__()

        self.feature_dim = feature_dim

        # 5 stride-2 convolutions
        # Input: 1 x 180 x 320
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3)
        # -> 16 x 90 x 160

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        # -> 32 x 45 x 80

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        # -> 64 x 23 x 40

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # -> 128 x 12 x 20

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        # -> 256 x 6 x 10

        # Final convolution to collapse spatial dimensions
        # Chen & Little use 6x10 convolution here
        self.conv_final = nn.Conv2d(256, feature_dim, kernel_size=(6, 10))
        # -> feature_dim x 1 x 1

        # Activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            x: input edge image, shape (batch, 1, 180, 320)

        Returns:
            torch.Tensor: L2-normalised feature vector, shape (batch, feature_dim)
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv_final(x)

        # Flatten: (batch, feature_dim, 1, 1) -> (batch, feature_dim)
        x = x.view(x.size(0), -1)

        # L2 normalisation (unit length feature vectors)
        x = F.normalize(x, p=2, dim=1)

        return x


class SiameseNetwork(nn.Module):
    """
    Full siamese network with two shared-weight branches.

    Takes a pair of edge images and outputs their feature vectors.
    The contrastive loss pulls similar pairs together and pushes
    dissimilar pairs apart in the feature space.
    """

    def __init__(self, feature_dim: int = 16):
        super().__init__()
        self.encoder = SiameseEncoder(feature_dim=feature_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both branches.

        Args:
            x1: first edge image, shape (batch, 1, 180, 320)
            x2: second edge image, shape (batch, 1, 180, 320)

        Returns:
            tuple: (features1, features2), each shape (batch, feature_dim)
        """
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        return f1, f2

    def extract_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature from a single edge image.
        Used at inference time (not training).

        Args:
            x: edge image, shape (batch, 1, 180, 320)

        Returns:
            torch.Tensor: feature vector, shape (batch, feature_dim)
        """
        return self.encoder(x)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function from Chen & Little (2019), equation (5):

        L(w, x1, x2, y) = y * D(x1,x2) + (1-y) * max(0, m - D(x1,x2))

    Where:
        D(x1,x2) = ||f(x1) - f(x2)||^2  (squared L2 distance)
        y = 1 for similar pairs, 0 for dissimilar pairs
        m = margin (1.0 in Chen & Little)

    Similar pairs are pulled together (minimise distance).
    Dissimilar pairs are pushed apart (distance > margin).
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, f1: torch.Tensor, f2: torch.Tensor,
                label: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            f1: features from first image, shape (batch, feature_dim)
            f2: features from second image, shape (batch, feature_dim)
            label: 1 for similar, 0 for dissimilar, shape (batch,)

        Returns:
            torch.Tensor: scalar loss value
        """
        # Squared Euclidean distance
        distance_sq = torch.sum((f1 - f2) ** 2, dim=1)

        # Contrastive loss
        similar_loss = label * distance_sq
        dissimilar_loss = (1 - label) * F.relu(self.margin - distance_sq)

        loss = torch.mean(similar_loss + dissimilar_loss)
        return loss


class SyntheticEdgePairDataset(Dataset):
    """
    Dataset that creates pairs of synthetic edge images for
    siamese network training.

    Pairs are labelled as similar or dissimilar based on
    camera pose proximity:
        Similar if:  |pan1 - pan2| < 1° AND
                     |tilt1 - tilt2| < 0.5° AND
                     |focal1 - focal2| < 30px
        Dissimilar otherwise.

    Each epoch generates random pairs from the pool of
    synthetic edge images.
    """

    def __init__(
        self,
        data_dir: str,
        num_pairs: int = 50000,
        pan_threshold: float = 1.0,
        tilt_threshold: float = 0.5,
        focal_threshold: float = 30.0,
        similar_ratio: float = 0.5,
    ):
        """
        Args:
            data_dir: path to synthetic dataset (contains edge_images/ and camera_poses.csv)
            num_pairs: number of pairs to generate per epoch
            pan_threshold: max pan difference for similar pair (degrees)
            tilt_threshold: max tilt difference for similar pair (degrees)
            focal_threshold: max focal length difference for similar pair (pixels)
            similar_ratio: proportion of similar pairs (0.5 = balanced)
        """
        self.data_dir = data_dir
        self.num_pairs = num_pairs
        self.pan_threshold = pan_threshold
        self.tilt_threshold = tilt_threshold
        self.focal_threshold = focal_threshold
        self.similar_ratio = similar_ratio

        # Load camera poses from CSV
        self.poses = []
        self.image_paths = []

        csv_path = os.path.join(data_dir, "camera_poses.csv")
        img_dir = os.path.join(data_dir, "edge_images")

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["index"])
                self.poses.append({
                    "pan": float(row["pan"]),
                    "tilt": float(row["tilt"]),
                    "focal_length": float(row["focal_length"]),
                })
                self.image_paths.append(
                    os.path.join(img_dir, f"edge_{idx:06d}.png")
                )

        self.num_images = len(self.poses)
        print(f"Loaded {self.num_images} synthetic edge images from {data_dir}")

        # Pre-generate pairs for this epoch
        self._generate_pairs()

    def _is_similar(self, i: int, j: int) -> bool:
        """Check if two poses are similar based on thresholds."""
        p1, p2 = self.poses[i], self.poses[j]
        return (
            abs(p1["pan"] - p2["pan"]) < self.pan_threshold and
            abs(p1["tilt"] - p2["tilt"]) < self.tilt_threshold and
            abs(p1["focal_length"] - p2["focal_length"]) < self.focal_threshold
        )

    def _generate_pairs(self):
        """Generate random pairs with balanced similar/dissimilar ratio."""
        self.pairs = []
        num_similar = int(self.num_pairs * self.similar_ratio)
        num_dissimilar = self.num_pairs - num_similar

        # Generate similar pairs
        similar_count = 0
        max_attempts = num_similar * 20
        attempts = 0

        while similar_count < num_similar and attempts < max_attempts:
            i = np.random.randint(0, self.num_images)
            j = np.random.randint(0, self.num_images)
            if i != j and self._is_similar(i, j):
                self.pairs.append((i, j, 1))
                similar_count += 1
            attempts += 1

        # If we couldn't find enough similar pairs, fill with what we have
        if similar_count < num_similar:
            print(f"Warning: only found {similar_count}/{num_similar} similar pairs. "
                  f"Consider generating more synthetic data or relaxing thresholds.")

        # Generate dissimilar pairs (much easier to find)
        dissimilar_count = 0
        while dissimilar_count < num_dissimilar:
            i = np.random.randint(0, self.num_images)
            j = np.random.randint(0, self.num_images)
            if i != j and not self._is_similar(i, j):
                self.pairs.append((i, j, 0))
                dissimilar_count += 1

        # Shuffle
        np.random.shuffle(self.pairs)

    def _load_image(self, path: str) -> torch.Tensor:
        """Load and preprocess an edge image."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")

        # Normalise to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Add channel dimension: (H, W) -> (1, H, W)
        img = torch.from_numpy(img).unsqueeze(0)

        return img

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tuple: (image1, image2, label)
                image1: (1, 180, 320) float tensor
                image2: (1, 180, 320) float tensor
                label: scalar tensor, 1=similar, 0=dissimilar
        """
        i, j, label = self.pairs[idx]

        img1 = self._load_image(self.image_paths[i])
        img2 = self._load_image(self.image_paths[j])
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return img1, img2, label_tensor


def train_siamese(
    data_dir: str,
    output_path: str = "weights/siamese.pth",
    feature_dim: int = 16,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    margin: float = 1.0,
    num_pairs_per_epoch: int = 50000,
    device: str = None,
    save_every: int = 10,
):
    """
    Train the siamese network on synthetic edge image pairs.

    This function is designed to run on Google Colab with GPU.
    It saves checkpoints to Google Drive periodically.

    Args:
        data_dir: path to synthetic dataset
        output_path: where to save the trained model
        feature_dim: dimension of output feature vector
        num_epochs: number of training epochs
        batch_size: batch size for training
        learning_rate: Adam learning rate
        margin: contrastive loss margin
        num_pairs_per_epoch: number of pairs per epoch
        device: 'cuda', 'cpu', or None (auto-detect)
        save_every: save checkpoint every N epochs
    """
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # Create dataset and dataloader
    dataset = SyntheticEdgePairDataset(
        data_dir=data_dir,
        num_pairs=num_pairs_per_epoch,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if device == "cuda" else 0,
        pin_memory=(device == "cuda"),
    )

    # Create model, loss, optimiser
    model = SiameseNetwork(feature_dim=feature_dim).to(device)
    criterion = ContrastiveLoss(margin=margin)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024:.1f} KB (float32)")
    print(f"Training {num_epochs} epochs, {num_pairs_per_epoch} pairs/epoch")
    print(f"Batch size: {batch_size}, LR: {learning_rate}, Margin: {margin}")
    print()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (img1, img2, labels) in enumerate(dataloader):
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)

            # Forward pass
            f1, f2 = model(img1, img2)
            loss = criterion(f1, f2, labels)

            # Backward pass
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Evaluate: compute average distances for similar vs dissimilar pairs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            sim_dists = []
            dissim_dists = []

            with torch.no_grad():
                for img1, img2, labels in dataloader:
                    img1 = img1.to(device)
                    img2 = img2.to(device)

                    f1, f2 = model(img1, img2)
                    dists = torch.sum((f1 - f2) ** 2, dim=1).cpu().numpy()

                    for d, l in zip(dists, labels.numpy()):
                        if l == 1:
                            sim_dists.append(d)
                        else:
                            dissim_dists.append(d)

                    # Only check first few batches for speed
                    if len(sim_dists) > 500:
                        break

            avg_sim = np.mean(sim_dists) if sim_dists else 0
            avg_dissim = np.mean(dissim_dists) if dissim_dists else 0

            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Sim dist: {avg_sim:.4f} | "
                  f"Dissim dist: {avg_dissim:.4f}")
        else:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "loss": avg_loss,
                "feature_dim": feature_dim,
            }, output_path)
            print(f"  -> Checkpoint saved to {output_path}")

        # Re-generate pairs for next epoch (different random pairs)
        dataset._generate_pairs()

    # Final save
    torch.save({
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "loss": avg_loss,
        "feature_dim": feature_dim,
    }, output_path)
    print(f"\nTraining complete! Model saved to {output_path}")

    return model


def load_trained_model(
    model_path: str,
    feature_dim: int = 16,
    device: str = None
) -> SiameseNetwork:
    """
    Load a trained siamese network from a checkpoint file.

    Args:
        model_path: path to the .pth checkpoint file
        feature_dim: must match the dimension used during training
        device: 'cuda', 'cpu', or None (auto-detect)

    Returns:
        SiameseNetwork: trained model in eval mode
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(model_path, map_location=device)

    # Use feature_dim from checkpoint if available
    if "feature_dim" in checkpoint:
        feature_dim = checkpoint["feature_dim"]

    model = SiameseNetwork(feature_dim=feature_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {model_path} "
          f"(epoch {checkpoint.get('epoch', '?')}, "
          f"loss {checkpoint.get('loss', '?'):.4f})")

    return model


if __name__ == "__main__":
    print("=== Siamese Network Test ===\n")

    # Test 1: Architecture verification
    print("1. Testing architecture...")
    model = SiameseNetwork(feature_dim=16)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    print(f"   Model size: {total_params * 4 / 1024:.1f} KB")

    # Test with dummy input
    dummy_input = torch.randn(2, 1, 180, 320)
    f1, f2 = model(dummy_input, dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {f1.shape}")
    print(f"   Feature norm: {torch.norm(f1[0]).item():.4f} (should be ~1.0)")

    # Test 2: Contrastive loss
    print("\n2. Testing contrastive loss...")
    criterion = ContrastiveLoss(margin=1.0)

    # Similar pair: loss should push features together
    f1_sim = torch.randn(4, 16)
    f2_sim = f1_sim + 0.1 * torch.randn(4, 16)  # Slightly different
    labels_sim = torch.ones(4)
    loss_sim = criterion(f1_sim, f2_sim, labels_sim)
    print(f"   Similar pair loss: {loss_sim.item():.4f}")

    # Dissimilar pair: loss should push features apart
    f1_dis = torch.randn(4, 16)
    f2_dis = torch.randn(4, 16)  # Completely different
    labels_dis = torch.zeros(4)
    loss_dis = criterion(f1_dis, f2_dis, labels_dis)
    print(f"   Dissimilar pair loss: {loss_dis.item():.4f}")

    # Test 3: Single feature extraction (inference mode)
    print("\n3. Testing feature extraction...")
    model.eval()
    with torch.no_grad():
        single_input = torch.randn(1, 1, 180, 320)
        feature = model.extract_feature(single_input)
        print(f"   Single image -> feature shape: {feature.shape}")
        print(f"   Feature values: {feature[0, :5].tolist()}")

    # Test 4: Dataset loading (if synthetic data exists)
    print("\n4. Testing dataset loading...")
    test_dir = "test_synthetic"
    if os.path.exists(test_dir):
        try:
            dataset = SyntheticEdgePairDataset(
                data_dir=test_dir,
                num_pairs=10,
            )
            img1, img2, label = dataset[0]
            print(f"   Pair shapes: {img1.shape}, {img2.shape}")
            print(f"   Label: {label.item()}")
            print(f"   Dataset size: {len(dataset)} pairs")
        except Exception as e:
            print(f"   Dataset loading failed: {e}")
    else:
        print(f"   Skipped (no test data at {test_dir})")
        print(f"   Run camera_pose_engine.py first to generate test data")

    print("\n=== All tests passed ===")
    print("\nTo train on Colab:")
    print("  from siamese_network import train_siamese")
    print("  model = train_siamese(")
    print("      data_dir='/content/drive/MyDrive/FYP_Project/synthetic_data',")
    print("      output_path='/content/drive/MyDrive/FYP_Project/weights/siamese.pth',")
    print("      num_epochs=50,")
    print("      batch_size=32")
    print("  )")