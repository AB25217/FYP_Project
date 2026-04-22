"""
demo_app.py — Streamlit demo for the tactical football analytics pipeline.

A minimal, reliable UI for demonstrating the end-to-end pipeline:
    Upload video  →  see progress  →  watch output video at the end

Run with:
    streamlit run app/demo_app.py

Designed to be demoed live. Caveats deliberately surfaced:
    - First run loads models (~30s). Subsequent runs reuse cached models.
    - Init phase scans the first 50 frames of the upload (~5 min on CPU).
    - Processing is capped (default 10s of video) so you don't sit
      through 2 hours of inference during a viva.
"""

import sys
import tempfile
import time
from pathlib import Path

# Make the repo importable whether you run from repo root or elsewhere
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import streamlit as st

from src.pipeline.tactical_detector import TacticalPipeline, FrameOutput



# Configuration

DEFAULT_YOLO_WEIGHTS = "src/weights/yolo/v12n_best.pt"
OUTPUT_DIR = REPO_ROOT / "app" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Cached pipeline construction

# Streamlit reruns the script top-to-bottom on every interaction. Without
# caching we'd reload YOLO + TwoGAN + siamese on every click — ~30 seconds
# wasted each time. @st.cache_resource memoises the pipeline across reruns.

@st.cache_resource(show_spinner="Loading models (first run only, ~30s)...")
def get_pipeline(yolo_weights: str) -> TacticalPipeline:
    """Build and cache a TacticalPipeline. Only runs once per session."""
    return TacticalPipeline(yolo_model=yolo_weights, device="cpu")


# Pipeline execution with Streamlit progress updates


def run_init_with_progress(
    pipeline: TacticalPipeline,
    video_path: str,
    init_frames: int,
    frame_stride: int,
    progress_bar,
    status_text,
) -> None:
    """Replicate TacticalPipeline.initialise_from_video() but with Streamlit
    progress updates instead of print statements.

    We duplicate the logic here (rather than patching the pipeline) to keep
    the core library clean and UI-agnostic.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    frame_idx = 0
    frames_used = 0
    accumulated_pitch_positions = []

    while frames_used < init_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        dets = pipeline.yolo.detect(frame)
        seg_mask = pipeline.gan.segment(frame)
        on_pitch = pipeline._filter_by_mask(dets, seg_mask)
        bboxes = [d.bbox for d in on_pitch]
        if len(bboxes) < 3:
            frame_idx += 1
            continue

        features, valid_idx = pipeline.hsv.extract_from_frame(frame, bboxes)
        if features.shape[0] < 3:
            frame_idx += 1
            continue

        h_result = pipeline.homography_estimator.estimate(frame)
        if h_result.H is None:
            frame_idx += 1
            continue

        try:
            H_inv = np.linalg.inv(h_result.H)
        except np.linalg.LinAlgError:
            frame_idx += 1
            continue

        pitch_pos = np.full((features.shape[0], 2), np.nan, dtype=np.float64)
        for feat_i, vi in enumerate(valid_idx):
            d = on_pitch[vi]
            fx, fy = d.foot_position
            p = H_inv @ np.array([fx, fy, 1.0])
            if abs(p[2]) > 1e-8:
                pitch_pos[feat_i, 0] = p[0] / p[2]
                pitch_pos[feat_i, 1] = p[1] / p[2]

        pipeline.clusterer.accumulate(features)
        accumulated_pitch_positions.append(pitch_pos)

        frames_used += 1
        frame_idx += 1

        progress_bar.progress(frames_used / init_frames)
        status_text.text(
            f"Init frame {frames_used}/{init_frames} "
            f"({len(on_pitch)} on-pitch detections, "
            f"{features.shape[0]} valid features)"
        )

    cap.release()

    if frames_used == 0:
        raise RuntimeError("No usable initialisation frames found")

    status_text.text("Fitting KMeans across accumulated features...")
    result = pipeline.clusterer.cluster_accumulated()
    all_pitch_positions = np.vstack(accumulated_pitch_positions)
    pipeline.assigner.assign(
        cluster_labels_per_detection=result.labels,
        cluster_centres=result.centres,
        pitch_positions_per_detection=all_pitch_positions,
    )
    pipeline._initialised = True


def run_processing_with_progress(
    pipeline: TacticalPipeline,
    video_path: str,
    output_path: str,
    start_frame: int,
    max_frames: int,
    progress_bar,
    status_text,
) -> tuple[int, float]:
    """Run pipeline.process_video() logic but with Streamlit updates.

    Returns (frames_processed, elapsed_seconds).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Peek output size from first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Could not read start_frame {start_frame}")
    sample = pipeline.render(first_frame, FrameOutput(frame_index=start_frame))
    out_h, out_w = sample.shape[:2]

    # Rewind and open writer
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer for {output_path}")

    t_start = time.time()
    processed = 0
    frame_idx = start_frame

    while processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        output = pipeline.process_frame(frame, frame_idx)
        annotated = pipeline.render(frame, output)
        writer.write(annotated)

        processed += 1
        frame_idx += 1

        # Update Streamlit UI every 5 frames to keep it responsive
        if processed % 5 == 0 or processed == max_frames:
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            remaining_frames = max_frames - processed
            eta = remaining_frames / rate if rate > 0 else 0
            progress_bar.progress(processed / max_frames)
            status_text.text(
                f"Frame {processed}/{max_frames}  |  "
                f"{rate:.1f} fps  |  "
                f"elapsed {elapsed:.0f}s  |  "
                f"eta {eta:.0f}s"
            )

    cap.release()
    writer.release()
    elapsed = time.time() - t_start
    return processed, elapsed



# Streamlit page


def main() -> None:
    st.set_page_config(
        page_title="Football Tactical Analytics",
        page_icon="⚽",
        layout="wide",
    )

    st.title("⚽ Football Tactical Analytics")
    st.caption(
        "Upload a broadcast football clip and run the full pipeline: "
        "YOLO detection → pitch segmentation → team clustering → "
        "camera calibration → top-down minimap."
    )

    # --- Sidebar: configuration ---------------------------------------------
    with st.sidebar:
        st.header("Pipeline configuration")

        yolo_weights = st.text_input(
            "YOLO weights",
            value=DEFAULT_YOLO_WEIGHTS,
            help="Path to trained YOLO checkpoint (.pt)",
        )

        init_frames = st.slider(
            "Init frames",
            min_value=10, max_value=100, value=30, step=5,
            help="Number of frames used to fit team clustering. More = more robust, slower.",
        )

        frame_stride = st.slider(
            "Init frame stride",
            min_value=1, max_value=60, value=30, step=1,
            help="Sample every Nth frame during init to get variety across the clip.",
        )

        start_frame = st.number_input(
            "Start frame (processing)",
            min_value=0, max_value=100000, value=0, step=50,
            help="Which frame to start from for the processed output clip.",
        )

        clip_seconds = st.slider(
            "Output clip length (seconds)",
            min_value=1, max_value=30, value=5, step=1,
            help="Duration of processed output. Kept short so demos finish quickly.",
        )

        st.markdown("---")
        st.caption(
            "First run loads models (~30s). Init takes ~3-5 min. "
            "Processing is ~0.2-5 s/frame on CPU."
        )

    # --- Main content: upload + process -------------------------------------
    uploaded = st.file_uploader(
        "Choose a football video (.mp4 / .avi / .mov / .mkv)",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded is None:
        st.info("Upload a video file to begin.")
        return

    st.success(
        f"Uploaded **{uploaded.name}** ({uploaded.size / 1e6:.1f} MB)"
    )

    if not st.button("▶️ Run pipeline", type="primary", use_container_width=True):
        return

    # Save upload to a temp file on disk (OpenCV needs a real path)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(uploaded.name).suffix, dir=str(OUTPUT_DIR),
        ) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        # Peek at video metadata so the user knows what they uploaded
        peek = cv2.VideoCapture(tmp_path)
        total_frames = int(peek.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = peek.get(cv2.CAP_PROP_FPS) or 25.0
        in_w = int(peek.get(cv2.CAP_PROP_FRAME_WIDTH))
        in_h = int(peek.get(cv2.CAP_PROP_FRAME_HEIGHT))
        peek.release()

        st.write(
            f"**Video:** {in_w}×{in_h} @ {fps:.1f}fps, {total_frames} total frames "
            f"({total_frames / fps:.1f}s)"
        )

        # Validate parameters against the uploaded video
        max_frames = int(clip_seconds * fps)
        if start_frame >= total_frames:
            st.error(
                f"Start frame ({start_frame}) is past the end of the video "
                f"({total_frames} frames). Lower it in the sidebar."
            )
            return
        if start_frame + max_frames > total_frames:
            max_frames = total_frames - start_frame
            st.warning(
                f"Clip extends past end of video — truncated to {max_frames} frames."
            )

        # ----- Load pipeline (cached across reruns) --------------------------
        try:
            pipeline = get_pipeline(yolo_weights)
        except FileNotFoundError as e:
            st.error(f"Could not load YOLO weights: {e}")
            return

        # Reset the initialised flag — team clusters are match-specific, so
        # every new upload needs a fresh init. clear_accumulated() drops the
        # previous match's feature buffer so it doesn't pollute the new fit.
        pipeline._initialised = False
        if hasattr(pipeline.clusterer, "clear_accumulated"):
            pipeline.clusterer.clear_accumulated()

        # ----- Stage 1: initialisation --------------------------------------
        st.subheader("Step 1 · Team clustering initialisation")
        init_bar = st.progress(0.0)
        init_status = st.empty()
        try:
            run_init_with_progress(
                pipeline=pipeline,
                video_path=tmp_path,
                init_frames=init_frames,
                frame_stride=frame_stride,
                progress_bar=init_bar,
                status_text=init_status,
            )
        except Exception as e:
            st.error(f"Initialisation failed: {e}")
            return
        init_bar.progress(1.0)
        init_status.success(
            f"Init complete. Cluster labels: "
            f"{pipeline.assigner.labels}"
        )

        # ----- Stage 2: video processing -------------------------------------
        st.subheader("Step 2 · Processing clip")
        proc_bar = st.progress(0.0)
        proc_status = st.empty()
        output_path = str(OUTPUT_DIR / f"demo_{int(time.time())}.mp4")

        try:
            processed, elapsed = run_processing_with_progress(
                pipeline=pipeline,
                video_path=tmp_path,
                output_path=output_path,
                start_frame=int(start_frame),
                max_frames=max_frames,
                progress_bar=proc_bar,
                status_text=proc_status,
            )
        except Exception as e:
            st.error(f"Processing failed: {e}")
            return

        proc_bar.progress(1.0)
        proc_status.success(
            f"Processed {processed} frames in {elapsed:.1f}s "
            f"({processed / elapsed:.2f} fps average)"
        )

        # ----- Output display ------------------------------------------------
        st.subheader("Output")

        # Read the rendered video into memory for both preview and download
        with open(output_path, "rb") as f:
            video_bytes = f.read()

        st.video(video_bytes)

        st.download_button(
            label="⬇️ Download annotated clip",
            data=video_bytes,
            file_name=f"tactical_{uploaded.name}",
            mime="video/mp4",
            use_container_width=True,
        )

    finally:
        # Clean up the uploaded temp file (the rendered output is kept)
        if tmp_path and Path(tmp_path).exists():
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()