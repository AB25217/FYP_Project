"""
cascaded_cnn.py — Cascaded CNN player and ball detector.

Implements the cascaded CNN architecture from Lu, Chen, Little & He (2017)
"Light Cascaded Convolutional Neural Networks for Accurate Player Detection".

Architecture:
    Main network: 4 convolutional layers with shared feature maps
    Classification branches: 4 branches at increasing depth
    
    Input patch → Conv-M1 → Pool → Conv-M2 → Pool → Conv-M3 → Pool → Conv-M4
                     ↓              ↓                ↓                ↓
                  Branch 1       Branch 2          Branch 3         Branch 4
                  (easy neg)     (medium neg)      (hard neg)       (hardest neg)

Key properties:
    - Less than 100KB model size (vs 100MB+ for general-purpose detectors)
    - Cascaded rejection: most negatives rejected by early branches (fast)
    - Each branch handles different difficulty levels
    - End-to-end training with joint loss function
    - ~10 FPS on CPU for 1280x720 images

The cascaded approach is efficient because in a typical broadcast frame,
~95% of sliding windows are background. Branch 1 rejects most of these
with minimal computation, so only ~5% of windows reach the deeper branches.

Training uses the same positive/negative patches from data_preparation.py.

Reference:
    Lu, K., Chen, J., Little, J.J. & He, H. (2017)
    "Light Cascaded Convolutional Neural Networks for Accurate Player Detection"
    British Machine Vision Conference (BMVC)
"""

import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass


@dataclass
class Detection:
    """A single detection from the cascaded CNN."""
    x: int
    y: int
    w: int
    h: int
    confidence: float
    class_name: str  # "player" or "ball"

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def centre(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def foot_position(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h)


class ClassificationBranch(nn.Module):
    """
    A single classification branch.
    
    Early branches (1, 2) use pooling before classification
    to reduce spatial dimensions further.
    Later branches (3, 4) operate on the feature maps directly
    with dropout for regularisation.
    """

    def __init__(self, in_channels: int, use_pool: bool = True):
        super().__init__()

        layers = []

        if use_pool:
            layers.append(nn.AdaptiveAvgPool2d(2))

        layers.extend([
            nn.Dropout(p=0.5),
            nn.Flatten(),
        ])

        # Calculate flattened size
        self._in_features = None
        self._use_pool = use_pool
        self._in_channels = in_channels

        self.features = nn.Sequential(*layers)
        self.classifier = None  # Built on first forward pass

    def _build_classifier(self, x):
        """Build the linear layer based on actual input size."""
        flat_size = x.shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        ).to(x.device)

    def forward(self, x):
        x = self.features(x)
        if self.classifier is None:
            self._build_classifier(x)
        return self.classifier(x)


class CascadedCNN(nn.Module):
    """
    Cascaded CNN for player detection.

    Main network: 4 conv layers with shared feature maps
    Branches: 4 classification branches at different depths

    During inference, a patch is passed through branches sequentially.
    If any branch classifies it as negative, processing stops (fast rejection).
    Only patches passing all branches are classified as players.

    Usage:
        model = CascadedCNN()

        # Training
        outputs = model(batch_of_patches)  # Returns dict of branch outputs

        # Inference (cascaded rejection)
        is_player, confidence = model.classify(single_patch)
    """

    def __init__(
        self,
        input_channels: int = 3,
        input_height: int = 96,
        input_width: int = 48,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width

        # Main network — shared feature extraction
        # Conv-M1: input → 16 channels
        self.conv_m1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )
        self.pool_m1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv-M2: 16 → 32 channels
        self.conv_m2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool_m2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv-M3: 32 → 64 channels
        self.conv_m3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool_m3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv-M4: 64 → 64 channels
        self.conv_m4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Classification branches
        self.branch1 = ClassificationBranch(16, use_pool=True)
        self.branch2 = ClassificationBranch(32, use_pool=True)
        self.branch3 = ClassificationBranch(64, use_pool=False)
        self.branch4 = ClassificationBranch(64, use_pool=False)

        # Branch thresholds for cascaded rejection
        self.thresholds = [0.5, 0.5, 0.5, 0.5]

    def forward(self, x):
        """
        Forward pass through all branches (used during training).

        Args:
            x: (B, C, H, W) batch of patches

        Returns:
            dict: outputs from each branch
                'branch1': (B, 2) logits
                'branch2': (B, 2) logits
                'branch3': (B, 2) logits
                'branch4': (B, 2) logits
        """
        # Main network
        f1 = self.conv_m1(x)
        p1 = self.pool_m1(f1)

        f2 = self.conv_m2(p1)
        p2 = self.pool_m2(f2)

        f3 = self.conv_m3(p2)
        p3 = self.pool_m3(f3)

        f4 = self.conv_m4(p3)

        # Branch outputs
        b1 = self.branch1(f1)
        b2 = self.branch2(f2)
        b3 = self.branch3(f3)
        b4 = self.branch4(f4)

        return {
            'branch1': b1,
            'branch2': b2,
            'branch3': b3,
            'branch4': b4,
        }

    def classify(self, patch: torch.Tensor) -> Tuple[bool, float]:
        """
        Cascaded classification of a single patch.

        Passes through branches sequentially. If any branch
        rejects the patch, returns immediately (fast path).

        Args:
            patch: (1, C, H, W) single patch tensor

        Returns:
            tuple: (is_player, confidence)
        """
        self.eval()
        with torch.no_grad():
            # Main network forward
            f1 = self.conv_m1(patch)
            p1 = self.pool_m1(f1)

            # Branch 1 — reject easy negatives
            b1 = torch.softmax(self.branch1(f1), dim=1)
            if b1[0, 1].item() < self.thresholds[0]:
                return False, b1[0, 1].item()

            f2 = self.conv_m2(p1)
            p2 = self.pool_m2(f2)

            # Branch 2 — reject medium negatives
            b2 = torch.softmax(self.branch2(f2), dim=1)
            if b2[0, 1].item() < self.thresholds[1]:
                return False, b2[0, 1].item()

            f3 = self.conv_m3(p2)
            p3 = self.pool_m3(f3)

            # Branch 3 — reject hard negatives
            b3 = torch.softmax(self.branch3(f3), dim=1)
            if b3[0, 1].item() < self.thresholds[2]:
                return False, b3[0, 1].item()

            f4 = self.conv_m4(p3)

            # Branch 4 — final classification
            b4 = torch.softmax(self.branch4(f4), dim=1)
            confidence = b4[0, 1].item()

            return confidence >= self.thresholds[3], confidence

    def get_model_size(self) -> dict:
        """Return model size statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate model size in bytes (float32 = 4 bytes per param)
        size_bytes = total_params * 4
        size_kb = size_bytes / 1024

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'size_kb': size_kb,
            'size_mb': size_kb / 1024,
        }


class PatchDataset(Dataset):
    """
    PyTorch dataset for player/non-player patches.
    Loads patches from positive/ and negative/ directories.
    """

    def __init__(
        self,
        positive_dir: str,
        negative_dir: str,
        input_size: Tuple[int, int] = (48, 96),
        augment: bool = True,
    ):
        self.input_size = input_size
        self.augment = augment

        # Load file paths
        self.samples = []

        for f in os.listdir(positive_dir):
            if f.endswith(('.jpg', '.png')):
                self.samples.append((os.path.join(positive_dir, f), 1))

        for f in os.listdir(negative_dir):
            if f.endswith(('.jpg', '.png')):
                self.samples.append((os.path.join(negative_dir, f), 0))

        np.random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)

        if img is None:
            img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)

        # Resize to expected input
        img = cv2.resize(img, self.input_size)

        # Augmentation
        if self.augment:
            if np.random.random() > 0.5:
                img = cv2.flip(img, 1)
            if np.random.random() > 0.5:
                factor = np.random.uniform(0.7, 1.3)
                img = np.clip(img * factor, 0, 255).astype(np.uint8)

        # Convert BGR to RGB, normalise, to tensor
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.FloatTensor(img).permute(2, 0, 1) / 255.0

        return tensor, label


class CascadedCNNTrainer:
    """
    Trainer for the cascaded CNN.

    Implements the joint loss function from Lu et al. (2017):
        L(w) = sum_i lambda_i * L_i(w)
    where L_i is the cross-entropy loss for branch i and
    lambda_i weights the contribution of each branch.

    Usage:
        trainer = CascadedCNNTrainer(model, device='cuda')
        trainer.train(train_loader, val_loader, epochs=30)
        trainer.save('weights/cascaded_cnn.pth')
    """

    def __init__(
        self,
        model: CascadedCNN,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        branch_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
    ):
        self.model = model.to(device)
        self.device = device
        self.branch_weights = branch_weights

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[8, 16, 20], gamma=0.1
        )

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 25,
    ) -> dict:
        """
        Train the cascaded CNN.

        Args:
            train_loader: training data loader
            val_loader: validation data loader
            epochs: number of training epochs

        Returns:
            dict: training statistics
        """
        best_val_acc = 0.0
        best_state = None

        print(f'Training Cascaded CNN for {epochs} epochs on {self.device}...\n')

        for epoch in range(epochs):
            # Train
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for patches, labels in train_loader:
                patches = patches.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(patches)

                # Joint loss: weighted sum of branch losses
                loss = 0.0
                for i, (key, weight) in enumerate(zip(
                    ['branch1', 'branch2', 'branch3', 'branch4'],
                    self.branch_weights
                )):
                    branch_loss = self.criterion(outputs[key], labels)
                    loss += weight * branch_loss

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_loss / num_batches
            self.train_losses.append(avg_train_loss)

            # Validate
            val_loss, val_acc, branch_accs = self._validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            self.scheduler.step()

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f'Epoch {epoch+1:3d}/{epochs} | '
                    f'Train Loss: {avg_train_loss:.4f} | '
                    f'Val Loss: {val_loss:.4f} | '
                    f'Val Acc: {val_acc:.3f} | '
                    f'Branch Accs: [{", ".join(f"{a:.3f}" for a in branch_accs)}] | '
                    f'LR: {self.scheduler.get_last_lr()[0]:.6f}'
                )

        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        print(f'\nTraining complete! Best val accuracy: {best_val_acc:.3f}')

        return {
            'best_val_accuracy': best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }

    def _validate(self, val_loader):
        """Validate on the validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        correct = 0
        total = 0
        branch_correct = [0, 0, 0, 0]
        branch_total = [0, 0, 0, 0]

        with torch.no_grad():
            for patches, labels in val_loader:
                patches = patches.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(patches)

                # Joint loss
                loss = 0.0
                for i, (key, weight) in enumerate(zip(
                    ['branch1', 'branch2', 'branch3', 'branch4'],
                    self.branch_weights
                )):
                    branch_loss = self.criterion(outputs[key], labels)
                    loss += weight * branch_loss

                    # Per-branch accuracy
                    preds = outputs[key].argmax(dim=1)
                    branch_correct[i] += (preds == labels).sum().item()
                    branch_total[i] += labels.size(0)

                total_loss += loss.item()
                num_batches += 1

                # Overall accuracy (from branch 4 — final branch)
                final_preds = outputs['branch4'].argmax(dim=1)
                correct += (final_preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / num_batches
        accuracy = correct / total
        branch_accs = [c / t if t > 0 else 0 for c, t in zip(branch_correct, branch_total)]

        return avg_loss, accuracy, branch_accs

    def evaluate(self, test_loader: DataLoader) -> dict:
        """
        Full evaluation on test set.

        Returns precision, recall, F1, and per-branch statistics.
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        branch_preds = {f'branch{i+1}': [] for i in range(4)}

        with torch.no_grad():
            for patches, labels in test_loader:
                patches = patches.to(self.device)
                outputs = self.model(patches)

                # Final branch prediction
                preds = outputs['branch4'].argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

                # Per-branch predictions
                for i in range(4):
                    key = f'branch{i+1}'
                    bp = outputs[key].argmax(dim=1).cpu().numpy()
                    branch_preds[key].extend(bp)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Metrics
        tp = np.sum((all_preds == 1) & (all_labels == 1))
        fp = np.sum((all_preds == 1) & (all_labels == 0))
        fn = np.sum((all_preds == 0) & (all_labels == 1))
        tn = np.sum((all_preds == 0) & (all_labels == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(all_labels)

        stats = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
        }

        print(f'=== Cascaded CNN Evaluation ===')
        print(f'Accuracy:  {accuracy:.3f}')
        print(f'Precision: {precision:.3f}')
        print(f'Recall:    {recall:.3f}')
        print(f'F1 Score:  {f1:.3f}')
        print(f'TP={tp} FP={fp} FN={fn} TN={tn}')

        # Per-branch accuracy
        print(f'\nPer-branch accuracy:')
        for i in range(4):
            key = f'branch{i+1}'
            bp = np.array(branch_preds[key])
            acc = np.mean(bp == all_labels)
            print(f'  Branch {i+1}: {acc:.3f}')

        return stats

    def save(self, path: str):
        """Save model weights."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(self.model.state_dict(), path)
        size = os.path.getsize(path)
        print(f'Model saved to {path} ({size / 1024:.1f} KB)')

    def load(self, path: str):
        """Load model weights."""
        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        print(f'Model loaded from {path}')


class CascadedCNNDetector:
    """
    Full-image player detector using the cascaded CNN with sliding window.

    Replaces the HOG+SVM sliding window approach with the trained
    cascaded CNN for improved accuracy.

    Usage:
        detector = CascadedCNNDetector(model, device='cpu')
        detections = detector.detect(frame)
        detector.draw_detections(frame, detections)
    """

    def __init__(
        self,
        model: CascadedCNN,
        device: str = 'cpu',
        window_size: Tuple[int, int] = (48, 96),
        scales: List[float] = None,
        stride: int = 12,
        confidence_threshold: float = 0.6,
        nms_threshold: float = 0.3,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.window_size = window_size
        self.scales = scales or [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2]
        self.stride = stride
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

    def detect(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> List[Detection]:
        """
        Detect players in a full frame using sliding window.

        Args:
            frame: BGR image
            mask: optional binary mask (255=search, 0=ignore)

        Returns:
            list: Detection objects for found players
        """
        win_w, win_h = self.window_size
        raw_detections = []

        for scale in self.scales:
            resized = cv2.resize(frame, None, fx=scale, fy=scale)
            rh, rw = resized.shape[:2]

            if mask is not None:
                resized_mask = cv2.resize(mask, (rw, rh))
            else:
                resized_mask = None

            for y in range(0, rh - win_h, self.stride):
                for x in range(0, rw - win_w, self.stride):
                    # Check mask at window centre
                    if resized_mask is not None:
                        cx = x + win_w // 2
                        cy = y + win_h // 2
                        if resized_mask[cy, cx] == 0:
                            continue

                    # Extract window
                    window = resized[y:y+win_h, x:x+win_w]

                    # Convert to tensor
                    window_rgb = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
                    tensor = torch.FloatTensor(window_rgb).permute(2, 0, 1).unsqueeze(0) / 255.0
                    tensor = tensor.to(self.device)

                    # Cascaded classification
                    is_player, confidence = self.model.classify(tensor)

                    if is_player and confidence >= self.confidence_threshold:
                        # Scale back to original coordinates
                        ox = int(x / scale)
                        oy = int(y / scale)
                        ow = int(win_w / scale)
                        oh = int(win_h / scale)
                        raw_detections.append(Detection(
                            x=ox, y=oy, w=ow, h=oh,
                            confidence=confidence,
                            class_name="player",
                        ))

        # Non-maximum suppression
        final = self._nms(raw_detections)
        return final

    def _nms(self, detections: List[Detection]) -> List[Detection]:
        """Non-maximum suppression to remove overlapping detections."""
        if len(detections) == 0:
            return []

        boxes = np.array([[d.x, d.y, d.w, d.h] for d in detections])
        scores = np.array([d.confidence for d in detections])

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = x1 + boxes[:, 2]
        y2 = y1 + boxes[:, 3]

        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            area_i = boxes[i, 2] * boxes[i, 3]
            area_j = boxes[order[1:], 2] * boxes[order[1:], 3]
            iou = inter / (area_i + area_j - inter + 1e-8)

            inds = np.where(iou <= self.nms_threshold)[0]
            order = order[inds + 1]

        return [detections[k] for k in keep]

    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        detections: List[Detection],
        colour: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw detection boxes on frame."""
        vis = frame.copy()
        for det in detections:
            cv2.rectangle(
                vis,
                (det.x, det.y),
                (det.x + det.w, det.y + det.h),
                colour, thickness
            )
            cv2.putText(
                vis, f'{det.confidence:.2f}',
                (det.x, det.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1
            )
        return vis

    @staticmethod
    def load_model(
        path: str,
        device: str = 'cpu',
    ) -> 'CascadedCNNDetector':
        """Load a trained model and create a detector."""
        model = CascadedCNN()
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return CascadedCNNDetector(model, device=device)


if __name__ == "__main__":
    print("=== Cascaded CNN Test ===\n")

    # Create model
    model = CascadedCNN()
    stats = model.get_model_size()
    print(f"Model size: {stats['total_params']:,} parameters ({stats['size_kb']:.1f} KB)")

    # Test forward pass
    dummy = torch.randn(2, 3, 96, 48)
    outputs = model(dummy)
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")

    # Test cascaded classification
    single = torch.randn(1, 3, 96, 48)
    is_player, conf = model.classify(single)
    print(f"\nCascaded classify: is_player={is_player}, confidence={conf:.3f}")

    print("\n=== Tests complete ===")