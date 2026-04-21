"""Compute homography from manual pixel-pitch correspondences for frame 1,
then project the full pitch template onto the frame as a sanity check.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


VIDEO_PATH = "data/test_videos/psg_newcastle_tactical.mp4"
FRAME_INDEX = 1
OUT_PATH = Path("tools/homography_from_points.png")


# Correspondences: pitch (metres) -> pixel
CORRESPONDENCES = [
    # Label,              pitch_x, pitch_y,  px_x, px_y
    ("halfway_near",      52.5,    68.0,     690,  552),
    ("halfway_far",       52.5,     0.0,     672,  311),
    ("centre_spot",       52.5,    34.0,     677,  277),
    ("left_pen_goal_near", 0.0,    54.16,    349,  460),
    ("left_pen_16_near",  16.5,    54.16,    145,  460),
    ("right_pen_16_near", 88.5,    54.16,   1228,  456),
    ("right_pen_16_far",  88.5,    13.84,   1002,  332),
]

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit(f"Could not read frame {FRAME_INDEX}")

    pitch_pts = np.array([(p[1], p[2]) for p in CORRESPONDENCES], dtype=np.float64)
    pixel_pts = np.array([(p[3], p[4]) for p in CORRESPONDENCES], dtype=np.float64)

    print("Correspondences:")
    for label, px_m, py_m, px, py in CORRESPONDENCES:
        print(f"  {label:22s} pitch=({px_m:6.2f}, {py_m:6.2f})  pixel=({px:4d}, {py:4d})")

    # Find H mapping pitch -> pixel
    H, mask = cv2.findHomography(pitch_pts, pixel_pts, method=0)
    if H is None:
        sys.exit("cv2.findHomography returned None — check your correspondences")

    print("\nHomography matrix H (pitch -> pixel):")
    print(H)

    # Reproject the input points and check error
    print("\nReprojection check:")
    for label, px_m, py_m, px, py in CORRESPONDENCES:
        p = H @ np.array([px_m, py_m, 1.0])
        px_pred = p[0] / p[2]
        py_pred = p[1] / p[2]
        err = np.sqrt((px_pred - px) ** 2 + (py_pred - py) ** 2)
        print(f"  {label:22s} target=({px:4d}, {py:4d})  predicted=({px_pred:6.1f}, {py_pred:6.1f})  err={err:5.2f} px")

    # Visualise — project the full pitch template
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
        if p1 and p2:
            cv2.line(overlay, p1, p2, (0, 255, 255), 2)

    # Centre circle
    pts = []
    for i in range(40):
        a = 2 * np.pi * i / 40
        pt = project(52.5 + 9.15 * np.cos(a), 34 + 9.15 * np.sin(a))
        if pt:
            pts.append(pt)
    for i in range(len(pts)):
        cv2.line(overlay, pts[i], pts[(i + 1) % len(pts)], (0, 255, 255), 2)

    # Mark the input points
    for label, _, _, px, py in CORRESPONDENCES:
        cv2.circle(overlay, (px, py), 8, (0, 0, 255), 2)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_PATH), overlay)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()