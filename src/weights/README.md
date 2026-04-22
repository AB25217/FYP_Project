# Football Tactical Analytics Pipeline

LINK TO GOOGLE DRIVE CONTAINING ALL TRAINING OF WEIGHTS, RESULTS AND DATA RELEVANT TO THE PROJECT: https://drive.google.com/drive/folders/1S8C390UK3nwXTu4ML9aexr1DrX7C-4Ek?usp=drive_link


An end-to-end computer vision pipeline for automated tactical analysis of broadcast football footage. Extracts player positions, classifies teams, tracks object identities across frames, estimates camera pose, and renders a top-down tactical minimap — all from a single broadcast camera on consumer CPU hardware.

Developed as the final-year undergraduate dissertation project (COMP1682) at the University of Greenwich, BSc (Hons) Computer Science. The primary research contribution is a comparative evaluation of three YOLO detector variants — YOLOv11n, YOLOv11m, and YOLOv12n — integrated within a modernised extension of the Mavrogiannis and Maglogiannis (2022) amateur analytics framework.

## What the pipeline does


Given broadcast football footage as input, the pipeline produces:

1. A side-by-side annotated video with labelled player bounding boxes and persistent track identifiers
2. A top-down tactical minimap showing the projected 2D pitch positions of all detected players, colour-coded by team

The pipeline consists of six components run per frame:

| Stage | Component | Approach |
|-------|-----------|----------|
| 1 | Player detection | YOLO (v11n / v11m / v12n) trained on SoccerNet Tracking |
| 2 | Pitch segmentation | Chen and Little (2019) two-GAN architecture |
| 3 | Multi-object tracking | Lucas-Kanade optical flow with IoU-based re-identification |
| 4 | Team classification | HSV colour histograms + K-means clustering (k=5) |
| 5 | Camera calibration | Siamese network retrieval over 90,000-entry synthetic pose database |
| 6 | Spatial projection | Inverse homography to real-world pitch coordinates |

Ball detection is evaluated at the model level (as part of research question 1) but is not integrated into the downstream pipeline.

## Research context

The project investigates three research questions:

1. **RQ1**: At matched parameter count, does YOLOv12n's attention-centric architecture outperform YOLOv11n on football detection in terms of accuracy and inference speed?
2. **RQ2**: Does the performance improvement generalise to the small-object ball detection challenge, where attention-based feature aggregation is theoretically most beneficial?
3. **RQ3**: Does the modernised detection backbone within the Mavrogiannis and Maglogiannis (2022) pipeline architecture yield improved end-to-end pipeline performance compared to the original published system?

Headline results (full analysis in Chapter 5 of the dissertation):

| Model | Parameters | Overall mAP50 | Ball mAP50 | Person mAP50 |
|-------|-----------|---------------|------------|--------------|
| YOLOv11n | 2.58M | 0.654 | 0.344 | 0.964 |
| YOLOv11m | 20.0M | 0.735 | 0.494 | 0.976 |
| **YOLOv12n** | **2.57M** | **0.694** | **0.414** | **0.973** |

YOLOv12n improves on YOLOv11n by 4.0 mAP50 points overall and 7.0 mAP50 points on ball detection, at essentially identical parameter count.

## Requirements

- Python 3.11 or later
- Windows 10/11 (tested) or Linux (should work, untested)
- Approximately 8 GB RAM
- No GPU required (CPU inference supported)
- Approximately 2 GB of free disk space for weights, test videos, and cache

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AB25217/FYP_Project.git
cd FYP_Project
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
pip install streamlit
```

### 3. Download the Chen-Little pretrained weights

These are the pretrained models from Chen and Little (2019) — the segmentation GAN, line detection GAN, Siamese encoder, and 90,000-entry pose database. They are fetched by an included download script:

```bash
python src/weights/download_weights.py
```

The script also downloads the PSG-Newcastle test video for convenience.


## Running the demo

### Option 1: Streamlit web interface (recommended)

On Windows, double-click `run_demo.bat` or run from PowerShell:

```powershell
python -m streamlit run app\demo_app.py
```

On Linux or macOS:

```bash
python -m streamlit run app/demo_app.py
```

A browser tab will open at `http://localhost:8501`. Upload a broadcast football video (up to 2 GB) and click **Run pipeline**.

The first run takes approximately 30 seconds to load models, then 3-5 minutes for team clustering initialisation, followed by approximately 0.2-0.5 seconds per frame of video processing.

### Option 2: Command-line interface

For scripted runs or batch processing without the UI:

```bash
python tools/run_video.py
```

Video path and clip length are configured at the top of `tools/run_video.py`. Output is written to `tools/video_output/tactical_demo.mp4`.

## Project structure

```
FYP_Project/
├── app/
│   └── demo_app.py              Streamlit web interface
├── src/
│   ├── pipeline/
│   │   └── tactical_detector.py Main pipeline orchestrator
│   ├── player_detection/
│   │   └── yolo_detector.py     YOLO wrapper (any v11/v12 variant)
│   ├── court_detection/
│   │   └── two_gan_detector.py  Chen-Little two-GAN pitch segmentation
│   ├── team_clustering/         HSV + K-means team classification
│   ├── camera_callibration/     Siamese + pose database camera calibration
│   ├── tracking/                Lucas-Kanade multi-object tracker
│   └── weights/
│       └── download_weights.py  Fetches Chen-Little weights and test video
├── tools/
│   └── run_video.py             Headless pipeline runner for testing
├── .streamlit/
│   └── config.toml              Raises upload limit to 2 GB
├── requirements.txt             Python dependencies
├── run_demo.bat                 Windows launcher for Streamlit app
└── README.md                    This file
```

## Known limitations

As documented in the dissertation (Chapter 5, Limitations and Failure Modes):

- **Minimap positional accuracy**: the homography produced by the Chen-Little calibration exhibits 3-5 metre offsets from true pitch positions. The 90,000-entry pose database was constructed using World Cup 2014 camera-position priors that do not match all stadiums.
- **Off-pitch false positives**: the pitch segmentation mask has a fuzzy 10-20 pixel boundary band; detections near the touchline occasionally pass the filter when they shouldn't.
- **Wide-angle performance**: at wide-angle camera framings, players can occupy fewer than 20 pixels, approaching the spatial resolution limit of the detector. Detection and team classification both degrade at this scale, though track-ID propagation partially mitigates the classification issue.
- **Inline video playback in Streamlit**: the `mp4v` codec used by OpenCV's VideoWriter is not natively supported by most browsers. Annotated output videos can be downloaded via the UI button and viewed in an external player (VLC, Windows Media Player).

## License and attribution

- **Ultralytics YOLO** (v11 and v12): AGPL-3.0 — the use of YOLOv11 and YOLOv12 in this project imposes AGPL-3.0 requirements on any redistributed modified version.
- **Chen and Little (2019) two-GAN calibration**: pretrained weights used under the original authors' terms. See https://github.com/lood339/pytorch-two-GAN for reference.
- **Mavrogiannis and Maglogiannis (2022) pipeline architecture**: the overall pipeline structure (HSV clustering, Lucas-Kanade tracking, and calibration integration) is adapted from their published framework. See https://github.com/pmavr/sport-analysis for reference.
- **SoccerNet Tracking dataset**: used for detector training under a research-only NDA with the dataset authors. See https://www.soccer-net.org for dataset access.


## References

Full reference list available in the dissertation bibliography. Key citations:

- Chen, J. and Little, J. J. (2019) "Sports Camera Calibration via Synthetic Data", CVPR Workshops.
- Cioppa, A. et al. (2022) "SoccerNet-Tracking: Multiple Object Tracking Dataset and Benchmark in Soccer Videos", CVPR Workshops.
- Khanam, R. and Hussain, M. (2024) "YOLOv11: An overview of the key architectural enhancements", arXiv:2410.17725.
- Mavrogiannis, P. and Maglogiannis, I. (2022) "Amateur football analytics using computer vision", Neural Computing and Applications.
- Tian, Y., Ye, Q. and Doermann, D. (2025) "YOLOv12: Attention-centric real-time object detectors", NeurIPS.
- Zheng, H. et al. (2025) "A review of computer vision techniques for football analytics", Information.