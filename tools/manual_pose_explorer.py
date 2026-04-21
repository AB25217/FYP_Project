"""manual_pose_explorer.py — Interactively find the camera pose that matches a frame.

Renders pitch markings using a candidate pose and overlays them onto the real
frame so you can eyeball the match. Iterate the parameters until the yellow
pitch lines align with the real pitch.

Usage:
    python tools\\manual_pose_explorer.py --pan -10 --tilt -10 --focal 1500 \\
        --cx 52.5 --cy -12 --cz 22
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera_callibration.camera_pose_engine import CameraParams, CameraPoseEngine


VIDEO_PATH = "data/test_videos/psg_newcastle_tactical.mp4"
FRAME_INDEX = 3000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pan", type=float, default=0.0)
    parser.add_argument("--tilt", type=float, default=-10.0)
    parser.add_argument("--focal", type=float, default=1500.0)
    parser.add_argument("--cx", type=float, default=52.5)
    parser.add_argument("--cy", type=float, default=-12.0)
    parser.add_argument("--cz", type=float, default=22.0)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--frame", type=int, default=FRAME_INDEX)
    parser.add_argument("--out", type=str, default="tools/manual_pose_preview.png")
    args = parser.parse_args()

    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit(f"Could not read frame {args.frame}")

    pose = CameraParams(
        pan=args.pan, tilt=args.tilt, focal_length=args.focal,
        cx=args.cx, cy=args.cy, cz=args.cz, roll=args.roll,
    )
    engine = CameraPoseEngine(image_width=1280, image_height=720)
    H = engine.get_homography(pose)

    overlay = frame.copy()

    lines = [
        ((0, 0), (105, 0)),
        ((105, 0), (105, 68)),
        ((105, 68), (0, 68)),
        ((0, 68), (0, 0)),
        ((52.5, 0), (52.5, 68)),
        ((0, 13.84), (16.5, 13.84)),
        ((16.5, 13.84), (16.5, 54.16)),
        ((16.5, 54.16), (0, 54.16)),
        ((105, 13.84), (88.5, 13.84)),
        ((88.5, 13.84), (88.5, 54.16)),
        ((88.5, 54.16), (105, 54.16)),
    ]

    def project(x, y):
        p = H @ np.array([x, y, 1.0])
        if abs(p[2]) < 1e-8:
            return None
        return (int(p[0] / p[2]), int(p[1] / p[2]))

    for (x1, y1), (x2, y2) in lines:
        p1 = project(x1, y1)
        p2 = project(x2, y2)
        if p1 is None or p2 is None:
            continue
        cv2.line(overlay, p1, p2, (0, 255, 255), 2)

    # Centre circle
    cx_p, cy_p = 52.5, 34.0
    radius = 9.15
    pts = []
    for i in range(30):
        a = 2 * np.pi * i / 30
        pt = project(cx_p + radius * np.cos(a), cy_p + radius * np.sin(a))
        if pt is not None:
            pts.append(pt)
    for i in range(len(pts)):
        cv2.line(overlay, pts[i], pts[(i + 1) % len(pts)], (0, 255, 255), 2)

    cv2.putText(overlay,
                f"pan={args.pan:.1f}  tilt={args.tilt:.1f}  focal={args.focal:.0f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(overlay,
                f"cx={args.cx:.1f}  cy={args.cy:.1f}  cz={args.cz:.1f}  roll={args.roll:.1f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, overlay)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
    