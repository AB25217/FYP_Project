"""Run the tactical pipeline over a video clip and write an annotated output."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.tactical_detector import TacticalPipeline


VIDEO_PATH = "data/test_videos/psg_newcastle_tactical.mp4"
OUTPUT_PATH = "tools/video_output/tactical_demo.mp4"

INIT_FRAMES = 50           # was 10 — better team clustering, takes ~5 min extra
START_FRAME = 3000         # already what you want
CLIP_SECONDS = 60          # was 3 — 60 seconds = 1500 frames at 25fps
FPS = 25
MAX_FRAMES = CLIP_SECONDS * FPS     # ← this line is missing

def main():
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    pipeline = TacticalPipeline()
    pipeline.initialise_from_video(VIDEO_PATH, init_frames=INIT_FRAMES, frame_stride=30)

    print(f"\nStarting full-video processing")
    print(f"Clip length: {CLIP_SECONDS}s ({MAX_FRAMES} frames from frame {START_FRAME})")
    print(f"Expected runtime: ~{MAX_FRAMES * 6 / 60:.0f} minutes at ~6s/frame")
    print()

    pipeline.process_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        max_frames=MAX_FRAMES,
        start_frame=START_FRAME,
        progress_interval=10,
    )

    print(f"\nFinal output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()