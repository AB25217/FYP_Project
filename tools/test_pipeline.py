
"""Run the tactical pipeline on one frame and save the annotated output."""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.tactical_detector import TacticalPipeline


VIDEO_PATH = "data/test_videos/psg_newcastle_tactical.mp4"
INIT_FRAMES = 50  # paper default # use 10 for fast dev iteration; bump to 50 for real runs
TEST_FRAME_INDEX = 3000
OUT_PATH = Path("tools/pipeline_preview/annotated.png")


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pipeline = TacticalPipeline()
    pipeline.initialise_from_video(VIDEO_PATH, init_frames=INIT_FRAMES, frame_stride=30)

    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, TEST_FRAME_INDEX)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit("Could not read frame")

    print(f"\nProcessing frame {TEST_FRAME_INDEX}...")
    output = pipeline.process_frame(frame, TEST_FRAME_INDEX)
    annotated = pipeline.render(frame, output)

    cv2.imwrite(str(OUT_PATH), annotated)

    print(f"\nDetections on pitch:  {len(output.detections)}")
    print(f"Homography available: {output.homography is not None}")
    print(f"Team label summary:")
    for label in sorted(set(output.team_labels)):
        count = output.team_labels.count(label)
        print(f"  {label}: {count}")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()