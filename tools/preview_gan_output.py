"""preview_gan_output.py — Extract a frame and run the two-GAN on it.

Saves four PNGs for inspection:
    - input.png      — original BGR frame
    - seg.png        — grass segmentation output
    - lines.png      — pitch line detection output
    - composite.png  — all three side-by-side for comparison
"""
import sys
from pathlib import Path

import cv2
import numpy as np

# Add repo root to path so we can import src.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.court_detection.two_gan_detector import TwoGANFieldDetector

VIDEO_PATH = "data/test_videos/psg_newcastle_tactical.mp4"
FRAME_INDEX = 3000
OUTPUT_DIR = Path("tools/gan_preview")

SEG_WEIGHTS = "src/weights/gan_weights/seg_latest_net_G.pth"
DET_WEIGHTS = "src/weights/gan_weights/detec_latest_net_G.pth"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Opening {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {VIDEO_PATH}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        sys.exit(f"Failed to read frame {FRAME_INDEX}")

    print(f"Got frame {FRAME_INDEX}: {frame.shape}")

    print("Loading two-GAN...")
    detector = TwoGANFieldDetector(
        seg_weights=SEG_WEIGHTS,
        det_weights=DET_WEIGHTS,
        device="cpu",
    )

    print("Running both generators...")
    seg, lines = detector.detect_field(frame)

    # Save individual outputs
    cv2.imwrite(str(OUTPUT_DIR / "input.png"), frame)
    cv2.imwrite(str(OUTPUT_DIR / "seg.png"), seg)
    cv2.imwrite(str(OUTPUT_DIR / "lines.png"), lines)

    # Composite: input on left, seg in middle, lines on right
    seg_bgr = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
    lines_bgr = cv2.cvtColor(lines, cv2.COLOR_GRAY2BGR)
    composite = np.hstack([frame, seg_bgr, lines_bgr])

    # Annotate
    labels = ["input", "grass_seg", "pitch_lines"]
    w = frame.shape[1]
    for i, label in enumerate(labels):
        cv2.putText(
            composite, label, (i * w + 20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2
        )

    cv2.imwrite(str(OUTPUT_DIR / "composite.png"), composite)

    print(f"\nSaved to {OUTPUT_DIR.resolve()}:")
    for f in OUTPUT_DIR.glob("*.png"):
        print(f"  {f.name}  ({f.stat().st_size / 1024:.0f} KB)")

    print(f"\nSeg output range:   [{seg.min()}, {seg.max()}]")
    print(f"Lines output range: [{lines.min()}, {lines.max()}]")


if __name__ == "__main__":
    main()