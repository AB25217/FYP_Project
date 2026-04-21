"""Smoke test: run GANHomographyEstimator on frame 3000 and visualise."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.court_detection.two_gan_detector import TwoGANFieldDetector
from src.camera_callibration.gan_homography_estimator import GANHomographyEstimator


VIDEO_PATH = "data/test_videos/psg_newcastle_tactical.mp4"
FRAME_INDEX = 3000
OUT_DIR = Path("tools/gan_homography_preview")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get the frame
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit("Could not read frame")
    print(f"Frame {FRAME_INDEX}: {frame.shape}")

    # Load models
    print("Loading GAN...")
    gan = TwoGANFieldDetector(
        seg_weights="src/weights/gan_weights/seg_latest_net_G.pth",
        det_weights="src/weights/gan_weights/detec_latest_net_G.pth",
        device="cpu",
    )

    print("Loading estimator...")
    estimator = GANHomographyEstimator(
        gan_detector=gan,
        siamese_weights="src/weights/siamese.pth",
        pose_database="src/weights/pose_database.npz",
        device="cpu",
    )

    print("Running estimate...")
    result = estimator.estimate(frame)

    if result.H is None:
        print("FAILED: no homography produced")
        return

    print(f"\nResults:")
    print(f"  Retrieved pose: pan={result.retrieved_pose.pan:.1f}, "
          f"tilt={result.retrieved_pose.tilt:.2f}, "
          f"focal={result.retrieved_pose.focal_length:.0f}")
    print(f"  Retrieval distance (feature space): {result.retrieval_distance:.4f}")
    print(f"  Refinement converged: {result.refinement_ok}")
    print(f"  Refinement correlation: {result.refinement_correlation:.3f}")

    # Project pitch markings onto the frame using the derived homography
    # Pitch lines in metres
    H = result.H
    overlay = frame.copy()

    lines_to_draw = [
        # Boundary rectangle
        [(0, 0), (105, 0)],
        [(105, 0), (105, 68)],
        [(105, 68), (0, 68)],
        [(0, 68), (0, 0)],
        # Halfway line
        [(52.5, 0), (52.5, 68)],
        # Left penalty box
        [(0, 13.84), (16.5, 13.84)],
        [(16.5, 13.84), (16.5, 54.16)],
        [(16.5, 54.16), (0, 54.16)],
        # Right penalty box
        [(105, 13.84), (88.5, 13.84)],
        [(88.5, 13.84), (88.5, 54.16)],
        [(88.5, 54.16), (105, 54.16)],
    ]

    for (x1, y1), (x2, y2) in lines_to_draw:
        p1 = H @ np.array([x1, y1, 1.0])
        p2 = H @ np.array([x2, y2, 1.0])
        if abs(p1[2]) < 1e-8 or abs(p2[2]) < 1e-8:
            continue
        p1 = (int(p1[0] / p1[2]), int(p1[1] / p1[2]))
        p2 = (int(p2[0] / p2[2]), int(p2[1] / p2[2]))
        cv2.line(overlay, p1, p2, (0, 255, 255), 2)

    # Blend overlay onto frame
    vis = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    # Redraw lines fully opaque on top
    for (x1, y1), (x2, y2) in lines_to_draw:
        p1 = H @ np.array([x1, y1, 1.0])
        p2 = H @ np.array([x2, y2, 1.0])
        if abs(p1[2]) < 1e-8 or abs(p2[2]) < 1e-8:
            continue
        p1 = (int(p1[0] / p1[2]), int(p1[1] / p1[2]))
        p2 = (int(p2[0] / p2[2]), int(p2[1] / p2[2]))
        cv2.line(vis, p1, p2, (0, 255, 255), 2)

    cv2.imwrite(str(OUT_DIR / "overlay.png"), vis)
    print(f"\nSaved: {OUT_DIR / 'overlay.png'}")


if __name__ == "__main__":
    main()