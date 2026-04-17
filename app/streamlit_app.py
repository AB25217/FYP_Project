"""
streamlit_app.py — Main Streamlit application for football analytics.

Run with:
    streamlit run app/streamlit_app.py

Features:
    - Upload a video clip or use a test video
    - Run the pipeline and show progress
    - Display annotated frames with bounding boxes
    - Show 2D pitch minimap with player positions
    - Show formation labels and confidence
    - Display match statistics

"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import sys
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pitch_visualiser import create_pitch_map
from app.formation_display import create_formation_summary, create_formation_timeline


def main():
    st.set_page_config(
        page_title="Football Analytics Pipeline",
        page_icon="⚽",
        layout="wide",
    )

    st.title("⚽ Football Analytics Pipeline")
    st.markdown(
        "Amateur football analytics with automated formation detection. "
        "Upload a broadcast video clip to analyse player positions, "
        "team formations, and match statistics."
    )

    # Sidebar configuration
    with st.sidebar:
        st.header("Pipeline Settings")

        detect_interval = st.slider(
            "Detection interval (frames)", 1, 10, 3,
            help="Run player detection every N frames. Lower = more accurate but slower."
        )

        process_every_n = st.slider(
            "Process every N frames", 1, 5, 1,
            help="Skip frames for faster processing."
        )

        max_frames = st.number_input(
            "Max frames to process", 0, 10000, 300,
            help="0 = process all frames."
        )
        if max_frames == 0:
            max_frames = None

        st.markdown("---")
        st.header("About")
        st.markdown(
        )

    # Main content
    tab1, tab2, tab3 = st.tabs(["📹 Video Analysis", "📊 Results", "ℹ️ Pipeline Info"])

    with tab1:
        video_analysis_tab(detect_interval, process_every_n, max_frames)

    with tab2:
        results_tab()

    with tab3:
        pipeline_info_tab()


def video_analysis_tab(detect_interval, process_every_n, max_frames):
    """Video upload and processing tab."""

    st.header("Upload Video")

    uploaded = st.file_uploader(
        "Upload a broadcast football video clip",
        type=["mp4", "avi", "mkv", "mov"],
    )

    # Demo mode with synthetic data
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎮 Run Demo (Synthetic Data)", use_container_width=True):
            run_demo()

    with col2:
        if uploaded and st.button("▶️ Process Video", use_container_width=True):
            process_uploaded_video(uploaded, detect_interval, process_every_n, max_frames)


def run_demo():
    """Run a demo with synthetic player positions."""
    st.subheader("Demo: Synthetic Formation Analysis")

    progress = st.progress(0)
    status = st.empty()

    # Simulate a 4-3-3 formation for team A
    np.random.seed(42)
    base_a = np.array([
        [3, 34],        # GK
        [25, 10], [25, 27], [25, 41], [25, 58],    # Defence
        [48, 17], [48, 34], [48, 51],                # Midfield
        [78, 14], [78, 34], [78, 54],                # Attack
    ], dtype=np.float64)

    # Simulate a 4-4-2 for team B
    base_b = np.array([
        [102, 34],      # GK
        [80, 10], [80, 27], [80, 41], [80, 58],
        [55, 10], [55, 27], [55, 41], [55, 58],
        [30, 25], [30, 43],
    ], dtype=np.float64)

    # Simulate frames with noise
    n_frames = 50
    all_positions_a = []
    all_positions_b = []

    for i in range(n_frames):
        noise_a = np.random.randn(*base_a.shape) * 2.0
        noise_b = np.random.randn(*base_b.shape) * 2.0

        pos_a = base_a + noise_a
        pos_b = base_b + noise_b

        all_positions_a.append(pos_a)
        all_positions_b.append(pos_b)

        progress.progress((i + 1) / n_frames)
        status.text(f"Processing frame {i+1}/{n_frames}...")

    # Use the last frame for display
    final_a = np.mean(all_positions_a[-10:], axis=0)
    final_b = np.mean(all_positions_b[-10:], axis=0)

    # Combine for visualisation
    all_pos = np.vstack([final_a, final_b])
    all_teams = ["team_a"] * 11 + ["team_b"] * 11
    all_ids = list(range(22))

    positions_list = [(p[0], p[1]) for p in all_pos]

    status.text("Generating pitch map...")

    # Create pitch visualisation
    pitch_img = create_pitch_map(
        positions_list, all_teams, all_ids,
        ball_position=(50, 34),
        formation_a="4-3-3",
        formation_b="4-4-2",
        title="Demo: Formation Detection"
    )

    # Formation summary
    summary_img = create_formation_summary("4-3-3", 0.92, "4-4-2", 0.87)

    # Display
    status.text("")
    progress.empty()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(pitch_img, caption="2D Pitch Map — Player Positions", use_container_width=True)

    with col2:
        st.image(summary_img, caption="Formation Detection", use_container_width=True)

        st.metric("Team A Players", "11")
        st.metric("Team B Players", "11")
        st.metric("Ball Detected", "Yes")

    # Timeline
    st.subheader("Formation Timeline")
    timestamps = list(range(0, n_frames * 2, 2))
    forms_a = ["4-3-3"] * n_frames
    forms_b = ["4-4-2"] * 30 + ["4-5-1"] * 20
    timeline_img = create_formation_timeline(timestamps, forms_a, forms_b)
    st.image(timeline_img, caption="Formation Changes Over Time", use_container_width=True)

    st.success("Demo complete! Upload a real video to analyse actual match footage.")


def process_uploaded_video(uploaded, detect_interval, process_every_n, max_frames):
    """Process an uploaded video through the pipeline."""

    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.subheader("Processing Video")

    try:
        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if max_frames:
            total = min(total, max_frames)

        st.info(f"Video: {total} frames at {fps:.1f} FPS")

        progress = st.progress(0)
        status = st.empty()
        frame_display = st.empty()

        frame_idx = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break

            if frame_idx % process_every_n == 0:
                # Display the frame (converted to RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if processed % 10 == 0:
                    frame_display.image(
                        frame_rgb, caption=f"Frame {frame_idx}",
                        use_container_width=True
                    )

                processed += 1
                progress.progress(min(frame_idx / total, 1.0))
                status.text(
                    f"Frame {frame_idx}/{total} | "
                    f"Processed: {processed}"
                )

            frame_idx += 1

        cap.release()
        progress.empty()
        status.empty()
        frame_display.empty()

        st.success(
            f"Processed {processed} frames. "
            f"Note: Full pipeline requires trained models (SVM, siamese, U-Net). "
            f"Run training notebooks on Colab first to enable full detection."
        )

    finally:
        os.unlink(tmp_path)


def results_tab():
    """Display analysis results."""
    st.header("Analysis Results")

    if "results" not in st.session_state:
        st.info("Run a video analysis or demo first to see results here.")
        st.markdown("Click **Run Demo** in the Video Analysis tab to see example output.")
        return

    st.write("Results would appear here after processing.")


def pipeline_info_tab():
    """Display information about the pipeline architecture."""
    st.header("Pipeline Architecture")

    st.markdown("""
    ### Processing Flow

    ```
    Video Frame
        │
        ├──→ Court Detection (HSV grass segmentation + edge detection)
        │        │
        │        └──→ Camera Calibration (siamese → database → ECC refinement)
        │
        ├──→ Player Detection (HOG + SVM sliding window)
        │        │
        │        └──→ Player Tracking (Lucas-Kanade optical flow)
        │
        ├──→ Ball Detection (modified Circle Hough Transform)
        │
        └──→ Team Clustering (HSV histograms + K-means)
                 │
                 └──→ Perspective Transform (homography → pitch coordinates)
                          │
                          ├──→ Formation Detection (line clustering + template matching)
                          │
                          └──→ Analytics (possession, attack/defence state)
    ```

    ### Modules

    | Module | Method | Training | Reference |
    |--------|--------|----------|-----------|
    | Court Detection | HSV + Hough Lines | No | Mavrogiannis (2022) |
    | Camera Calibration | Siamese + U-Net + ECC | Yes (Colab GPU) | Chen & Little (2019) |
    | Player Detection | HOG + SVM | Yes (CPU) | Dalal & Triggs (2005) |
    | Ball Detection | Circle Hough Transform | No | D'Orazio et al. (2002) |
    | Tracking | Lucas-Kanade | No | Bouguet (2001) |
    | Team Clustering | HSV + K-means | No | Mavrogiannis (2022) |
    | Perspective Transform | Homography | No | Mavrogiannis (2022) |
    | Formation etection | K-means + Hungarian | No | **Original contribution** |
    """)


if __name__ == "__main__":
    main()