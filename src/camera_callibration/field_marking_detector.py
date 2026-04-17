
"""
field_marking_detector.py — U-Net for detecting pitch lines in broadcast frames.

Takes a real RGB broadcast frame and outputs a binary edge image
showing only the pitch field markings (lines, circles, arcs).
This edge image can then be fed to the siamese network for
camera pose retrieval.

Architecture: U-Net encoder-decoder with skip connections.
    Encoder: 4 downsampling blocks (conv-conv-pool)
    Bottleneck: 2 convolutions
    Decoder: 4 upsampling blocks (upsample-concat-conv-conv)
    Output: 1-channel sigmoid (probability of field marking)

Training data: World Cup 2014 dataset
    Input: RGB broadcast frames (resized to 256x256)
    Target: binary field marking masks (derived from grass
            segmentation + edge detection on the original frames)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F8
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import glob
from typing import Tuple, Optional, List


class DoubleConv(nn.Module):
    """Two consecutive convolution-batchnorm-relu blocks."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """
    U-Net for field marking segmentation.

    Input: (batch, 3, 256, 256) — RGB broadcast frame
    Output: (batch, 1, 256, 256) — field marking probability map

    Architecture:
        Encoder path (downsampling):
            3  -> 64  -> pool (128x128)
            64 -> 128 -> pool (64x64)
            128 -> 256 -> pool (32x32)
            256 -> 512 -> pool (16x16)

        Bottleneck:
            512 -> 1024

        Decoder path (upsampling + skip connections):
            1024+512 -> 512 (32x32)
            512+256  -> 256 (64x64)
            256+128  -> 128 (128x128)
            128+64   -> 64  (256x256)

        Output:
            64 -> 1 (sigmoid)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Output
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: input RGB image, shape (batch, 3, 256, 256)

        Returns:
            torch.Tensor: field marking probability, shape (batch, 1, 256, 256)
        """
        # Encoder
        e1 = self.enc1(x)           # (b, 64, 256, 256)
        e2 = self.enc2(self.pool(e1))  # (b, 128, 128, 128)
        e3 = self.enc3(self.pool(e2))  # (b, 256, 64, 64)
        e4 = self.enc4(self.pool(e3))  # (b, 512, 32, 32)

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # (b, 1024, 16, 16)

        # Decoder with skip connections
        d4 = self.up4(b)                    # (b, 512, 32, 32)
        d4 = torch.cat([d4, e4], dim=1)     # (b, 1024, 32, 32)
        d4 = self.dec4(d4)                  # (b, 512, 32, 32)

        d3 = self.up3(d4)                   # (b, 256, 64, 64)
        d3 = torch.cat([d3, e3], dim=1)     # (b, 512, 64, 64)
        d3 = self.dec3(d3)                  # (b, 256, 64, 64)

        d2 = self.up2(d3)                   # (b, 128, 128, 128)
        d2 = torch.cat([d2, e2], dim=1)     # (b, 256, 128, 128)
        d2 = self.dec2(d2)                  # (b, 128, 128, 128)

        d1 = self.up1(d2)                   # (b, 64, 256, 256)
        d1 = torch.cat([d1, e1], dim=1)     # (b, 128, 256, 256)
        d1 = self.dec1(d1)                  # (b, 64, 256, 256)

        # Output with sigmoid activation
        out = torch.sigmoid(self.out_conv(d1))  # (b, 1, 256, 256)

        return out

    def predict(self, image: np.ndarray, device: str = "cpu",
                threshold: float = 0.5) -> np.ndarray:
        """
        Run inference on a single image. Convenience method for
        pipeline integration.

        Args:
            image: RGB image, any size, uint8 (H, W, 3)
            device: 'cuda' or 'cpu'
            threshold: probability threshold for binarisation

        Returns:
            np.ndarray: binary edge image (256, 256), uint8
        """
        self.eval()

        # Preprocess: resize to 256x256 and normalise
        original_size = (image.shape[1], image.shape[0])
        img_resized = cv2.resize(image, (256, 256))
        img_float = img_resized.astype(np.float32) / 255.0

        # HWC -> CHW -> batch
        tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device)

        # Forward pass
        with torch.no_grad():
            output = self(tensor)

        # To numpy, threshold, convert to uint8
        prob_map = output.squeeze().cpu().numpy()
        binary = (prob_map > threshold).astype(np.uint8) * 255

        return binary


class WorldCupFieldDataset(Dataset):
    """
    Dataset for training the U-Net on World Cup 2014 images.

    Each sample is a pair:
        Input: RGB broadcast frame (resized to 256x256)
        Target: binary field marking mask (256x256)

    The target mask is generated from the World Cup 2014 dataset's
    grass segmentation files and/or homography matrices. The approach:
        1. Load the RGB image
        2. Segment the grass area (using the .grass file or HSV thresholding)
        3. Detect edges within the grass area (Canny edge detection)
        4. Dilate the edges slightly to create the target mask

    Data augmentation:
        - Random horizontal flip
        - Random brightness/contrast adjustment
        - Random crop and resize (300x300 -> random crop to 256x256,
          following Chen & Little's augmentation strategy)
    """

    def __init__(
        self,
        data_dir: str,
        augment: bool = True,
        target_size: int = 256,
    ):
        """
        Args:
            data_dir: path to World Cup 2014 dataset (contains .jpg and .homographyMatrix files)
            augment: whether to apply data augmentation
            target_size: output image size (256x256)
        """
        self.data_dir = data_dir
        self.augment = augment
        self.target_size = target_size

        # Find all image files
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))

        if len(self.image_paths) == 0:
            # Try subdirectory structure
            self.image_paths = sorted(glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True))

        print(f"Found {len(self.image_paths)} images in {data_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            tuple: (image, target)
                image: (3, 256, 256) float tensor, normalised to [0, 1]
                target: (1, 256, 256) float tensor, binary mask
        """
        # Load RGB image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate target mask from the image itself
        target = self._generate_field_mask(image)

        # Data augmentation
        if self.augment:
            image, target = self._augment(image, target)

        # Resize to target size
        image = cv2.resize(image, (self.target_size, self.target_size))
        target = cv2.resize(target, (self.target_size, self.target_size))

        # Normalise image to [0, 1]
        image = image.astype(np.float32) / 255.0
        target = (target > 127).astype(np.float32)

        # Convert to tensors: HWC -> CHW
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        target_tensor = torch.from_numpy(target).unsqueeze(0)

        return image_tensor, target_tensor

    def _generate_field_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generate a binary mask of field markings from an RGB image.

        Steps:
            1. Convert to HSV and segment the grass area
            2. Apply Canny edge detection within the grass area
            3. Dilate edges to create a thicker target mask

        Args:
            image: RGB image (H, W, 3)

        Returns:
            np.ndarray: binary mask (H, W), uint8
        """
        # Step 1: Segment grass using HSV colour thresholding
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Green grass range in HSV
        # Hue: 30-85 (green range)
        # Saturation: 30-255 (not too desaturated)
        # Value: 30-255 (not too dark)
        lower_green = np.array([30, 30, 30])
        upper_green = np.array([85, 255, 255])
        grass_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Clean up the grass mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_CLOSE, kernel)
        grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_OPEN, kernel)

        # Step 2: Detect edges within the grass area
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Keep only edges within the grass area
        field_edges = cv2.bitwise_and(edges, grass_mask)

        # Step 3: Dilate edges slightly to create a thicker target
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        field_edges = cv2.dilate(field_edges, dilate_kernel, iterations=1)

        return field_edges

    def _augment(self, image: np.ndarray,
                 target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation to image and target.

        Following Chen & Little: resize to 300x300, random crop to 256x256.
        Plus random horizontal flip and brightness adjustment.
        """
        # Random horizontal flip (50% chance)
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
            target = cv2.flip(target, 1)

        # Random brightness adjustment
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.7, 1.3)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)

        # Resize to 300x300 then random crop to 256x256
        # (Chen & Little's augmentation strategy)
        image = cv2.resize(image, (300, 300))
        target = cv2.resize(target, (300, 300))

        # Random crop
        max_offset = 300 - self.target_size
        x_off = np.random.randint(0, max_offset + 1)
        y_off = np.random.randint(0, max_offset + 1)

        image = image[y_off:y_off + self.target_size,
                      x_off:x_off + self.target_size]
        target = target[y_off:y_off + self.target_size,
                        x_off:x_off + self.target_size]

        return image, target


class DiceBCELoss(nn.Module):
    """
    Combined Dice loss and Binary Cross-Entropy loss.

    Dice loss handles class imbalance well (field markings are
    a small fraction of the image). BCE provides pixel-wise
    supervision. Combining them gives better results than either alone.

    Following Chen & Little who use L1 loss with lambda=100 for
    their pix2pix GAN — our combined loss serves a similar purpose
    of balancing reconstruction accuracy with structural similarity.
    """

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCELoss()

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        # BCE component
        bce_loss = self.bce(pred, target)

        # Dice component
        smooth = 1e-6
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + smooth) / (
            pred_flat.sum() + target_flat.sum() + smooth
        )
        dice_loss = 1.0 - dice

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def train_field_detector(
    data_dir: str,
    output_path: str = "weights/field_unet.pth",
    num_epochs: int = 200,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    device: str = None,
    save_every: int = 20,
    val_split: float = 0.15,
):
    """
    Train the U-Net field marking detector on World Cup 2014 data.

    Designed to run on Google Colab with GPU.

    Args:
        data_dir: path to World Cup 2014 train_val directory
        output_path: where to save the trained model
        num_epochs: number of training epochs (Chen & Little use 200)
        batch_size: batch size (Chen & Little use 1 due to 12GB GPU limit,
                    but U-Net is smaller so we can use 4)
        learning_rate: initial learning rate (decays linearly, matching Chen & Little)
        device: 'cuda', 'cpu', or None (auto-detect)
        save_every: save checkpoint every N epochs
        val_split: fraction of data to use for validation
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # Create dataset
    full_dataset = WorldCupFieldDataset(data_dir=data_dir, augment=True)

    # Split into train/val
    num_val = int(len(full_dataset) * val_split)
    num_train = len(full_dataset) - num_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [num_train, num_val]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2 if device == "cuda" else 0,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2 if device == "cuda" else 0,
    )

    print(f"Training samples: {num_train}")
    print(f"Validation samples: {num_val}")

    # Create model, loss, optimiser
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = DiceBCELoss(bce_weight=0.5)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Linear learning rate decay (matching Chen & Little)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimiser, start_factor=1.0, end_factor=0.0,
        total_iters=num_epochs
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
    print()

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Backward
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / max(num_batches, 1)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        # Step the learning rate scheduler
        scheduler.step()
        current_lr = optimiser.param_groups[0]["lr"]

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train: {avg_train_loss:.4f} | "
                  f"Val: {avg_val_loss:.4f} | "
                  f"LR: {current_lr:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "val_loss": avg_val_loss,
                "train_loss": avg_train_loss,
            }, output_path)

        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = output_path.replace(".pth", f"_epoch{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "val_loss": avg_val_loss,
                "train_loss": avg_train_loss,
            }, checkpoint_path)
            print(f"  -> Checkpoint saved to {checkpoint_path}")

    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to {output_path}")

    return model


def load_field_detector(
    model_path: str,
    device: str = None,
) -> UNet:
    """
    Load a trained U-Net field marking detector.

    Args:
        model_path: path to .pth checkpoint
        device: 'cuda', 'cpu', or None (auto-detect)

    Returns:
        UNet: trained model in eval mode
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(model_path, map_location=device)

    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded field detector from {model_path} "
          f"(epoch {checkpoint.get('epoch', '?')}, "
          f"val_loss {checkpoint.get('val_loss', '?'):.4f})")

    return model


if __name__ == "__main__":
    print("=== Field Marking Detector Test ===\n")

    # Test 1: Architecture verification
    print("1. Testing U-Net architecture...")
    model = UNet(in_channels=3, out_channels=1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    print(f"   Size: {total_params * 4 / 1024 / 1024:.1f} MB")

    # Test with dummy input
    dummy = torch.randn(2, 3, 256, 256)
    output = model(dummy)
    print(f"   Input shape: {dummy.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    # Test 2: Loss function
    print("\n2. Testing DiceBCE loss...")
    criterion = DiceBCELoss()
    pred = torch.sigmoid(torch.randn(2, 1, 256, 256))
    target = (torch.randn(2, 1, 256, 256) > 0).float()
    loss = criterion(pred, target)
    print(f"   Loss value: {loss.item():.4f}")

    # Test 3: Single image prediction
    print("\n3. Testing single image prediction...")
    # Create a fake broadcast frame (green with white lines)
    fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    fake_frame[:, :] = [34, 139, 34]  # Green grass
    cv2.line(fake_frame, (0, 360), (1280, 360), (255, 255, 255), 3)
    cv2.line(fake_frame, (640, 0), (640, 720), (255, 255, 255), 3)
    cv2.circle(fake_frame, (640, 360), 100, (255, 255, 255), 3)

    binary = model.predict(fake_frame, threshold=0.5)
    print(f"   Input: {fake_frame.shape}")
    print(f"   Output: {binary.shape}")
    print(f"   White pixels: {np.sum(binary > 0)}")

    # Test 4: Field mask generation
    print("\n4. Testing field mask generation...")
    dataset = WorldCupFieldDataset.__new__(WorldCupFieldDataset)
    dataset.target_size = 256
    mask = dataset._generate_field_mask(fake_frame)
    print(f"   Mask shape: {mask.shape}")
    print(f"   Mask white pixels: {np.sum(mask > 0)}")

    print("\n=== All tests passed ===")
    print("\nTo train on Colab:")
    print("  from field_marking_detector import train_field_detector")
    print("  model = train_field_detector(")
    print("      data_dir='/content/drive/MyDrive/FYP_Project/datasets/world_cup_2014/train_val',")
    print("      output_path='/content/drive/MyDrive/FYP_Project/weights/field_unet.pth',")
    print("      num_epochs=200,")
    print("      batch_size=4")
    print("  )")