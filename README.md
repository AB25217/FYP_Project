# FYP_Project# ⚽ Football Analytics Pipeline — Amateur Match Analysis with Formation Detection

An end-to-end computer vision pipeline that processes broadcast football video to detect players, track positions, identify teams, project positions onto a 2D pitch, and classify team formations — all trained from scratch with no pretrained models.

**Final Year Project** — University of Greenwich, 2025/26

---

## Overview

This project builds a complete football analytics system accessible to amateur clubs who cannot afford commercial tracking systems (Opta, StatsBomb, Hawk-Eye). Given a standard broadcast video, the pipeline:

1. **Detects the pitch** and estimates the camera pose
2. **Detects and tracks** players and the ball frame-by-frame
3. **Identifies teams** by jersey colour
4. **Projects positions** onto a top-down 2D pitch view
5. **Classifies formations** (e.g. 4-3-3, 4-4-2) — the original research contribution

The system achieves this using only a single broadcast camera feed, with all models trained from scratch on publicly available datasets.

---

## Pipeline Architecture

```
Video Frame
    │
    ├──→ Court Detection (HSV grass segmentation + Canny edges)
    │        │
    │        └──→ Camera Calibration (Siamese network → pose database → ECC refinement)
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
                      └──→ Visualisation (Streamlit app with pitch minimap)
```

---

## Modules

| Module | Method | Training | Reference |
|--------|--------|----------|-----------|
| Court Detection | HSV thresholding + Hough Lines | No | Mavrogiannis (2022) |
| Camera Calibration | Siamese CNN + U-Net + ECC | Yes — Colab GPU | Chen & Little (2019) |
| Player Detection | HOG + SVM | Yes — CPU | Dalal & Triggs (2005) |
| Ball Detection | Modified Circle Hough Transform | No | D'Orazio et al. (2002) |
| Object Tracking | Pyramidal Lucas-Kanade | No | Bouguet (2001) |
| Team Clustering | HSV histograms + K-means | No | Mavrogiannis (2022) |
| Perspective Transform | Homography projection | No | Mavrogiannis (2022) |
| **Formation Detection** | **K-means lines + Hungarian matching** | **No** | **Original contribution** |

---

## Repository Structure

```
football-analytics/
├── src/
│   ├── court_detection/           # Grass segmentation, edge & line detection
│   ├── camera_calibration/        # Pitch template, siamese, U-Net, pose DB, ECC
│   ├── player_detection/          # HOG extractor, SVM classifier, sliding window
│   ├── ball_detection/            # Circle Hough Transform, semicircle, background sub
│   ├── tracking/                  # Lucas-Kanade tracker, detection/tracking manager
│   ├── team_clustering/           # HSV histograms, K-means, team assignment
│   ├── perspective_transform/     # Homography, pitch projection
│   ├── formation_detection/       # GK filter, line clustering, template matching
│   └── pipeline/                  # Video processor, config loader
├── app/                           # Streamlit visualisation interface
├── notebooks/                     # Colab training notebooks
├── configs/                       # paths.yaml, parameter configs
├── data/                          # README with dataset download instructions
├── weights/                       # README — trained weights stored on Google Drive
├── evaluation/                    # Evaluation scripts and results
└── tests/                         # Unit and integration tests
```

---

## Setup

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/yourusername/football-analytics.git
cd football-analytics
pip install -r requirements.txt
```

### Dataset Setup

All datasets are stored on Google Drive due to size constraints:
[**Google Drive Folder**](https://drive.google.com/drive/folders/17V4eQzGxwpuX09RTAcQAnztrFD7qOwWq)

See `data/README.md` for full download instructions and folder structure.

| Dataset | Purpose | Source |
|---------|---------|--------|
| World Cup 2014 | Camera calibration | [nhoma.github.io](https://nhoma.github.io/data/soccer_data.tar.gz) |
| Roboflow Football | Player/ball detection | [Roboflow Universe](https://universe.roboflow.com/yolo-pw0go/football-and-player) (CC BY 4.0) |
| ISSIA-CNR | Detection evaluation | [Google Drive](https://drive.google.com/file/d/1Pj6syLRShNQWQaunJmAZttUw2jDh8L_f/view) |

### Training Models (Google Colab)

The pipeline requires three trained models. Training notebooks are in `notebooks/`:

1. **Generate synthetic data** → `notebooks/01_generate_synthetic_data.ipynb`
2. **Train siamese network** → `notebooks/02_train_siamese.ipynb`
3. **Build pose database** → `notebooks/03_build_pose_database.ipynb`
4. **Train U-Net field detector** → `notebooks/04_train_field_detector.ipynb`
5. **Train HOG+SVM player detector** → `notebooks/02_train_svm_player.ipynb`

Trained weights are saved to Google Drive at `FYP_Project/weights/`.

---

## Usage

### Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

### Process a Video Programmatically

```python
from pipeline.video_processor import VideoProcessor, PipelineConfig

config = PipelineConfig(detect_interval=3)
processor = VideoProcessor(config=config)

results = processor.process_video("match_clip.mp4")

for frame in results:
    print(f"Frame {frame.frame_index}: {len(frame.player_bboxes)} players")
    if frame.formation:
        print(f"  Team A: {frame.formation.team_a_formation}")
        print(f"  Team B: {frame.formation.team_b_formation}")
```

### Test Camera Calibration

```bash
python test_camera_calibration.py
```

---

## Formation Detection — Original Contribution

No paper in the reviewed literature performs automated formation detection from broadcast footage. This module extends the Mavrogiannis pipeline by adding:

1. **Goalkeeper filtering** — excludes GK by proximity to goal line
2. **Temporal smoothing** — averages positions over a 30-second rolling window
3. **Line clustering** — K-means on x-coordinates to find defensive lines, with silhouette score for optimal k selection
4. **Template matching** — Hungarian algorithm for optimal assignment against 8 formation templates (4-4-2, 4-3-3, 4-2-3-1, 3-5-2, 3-4-3, 4-1-4-1, 5-3-2, 4-5-1)
5. **Formation change detection** — monitors displacement between time windows

---

## Evaluation Metrics

| Module | Metric | Target |
|--------|--------|--------|
| Player Detection | Precision / Recall / F1 | > 0.80 |
| Ball Detection | Detection Rate | > 0.70 (visible) |
| Camera Calibration | IoU (part) | > 0.90 |
| Team Clustering | V-measure | > 0.85 |
| Formation Detection | Accuracy vs ground truth | Evaluated on known formations |

---

## Key Design Decisions

- **No pretrained models** — all components trained from scratch to demonstrate understanding
- **CPU-friendly detection** — HOG+SVM and CHT run on a laptop without dedicated GPU
- **Modular design** — each component is independently testable and replaceable
- **Offline processing** — designed for post-match analysis (3 FPS target, not real-time)

---

## Literature

1. Mavrogiannis, P. & Maglogiannis, I. (2022). *Amateur football analytics pipeline*. Neural Computing and Applications, 34, 19639–19654.
2. Chen, J. & Little, J.J. (2019). *Sports camera calibration via synthetic data*. CVPR Workshops, 2497–2504.
3. Lu, K., Chen, J., Little, J.J. & He, H. (2017). *Light cascaded convolutional neural networks for accurate player detection*. BMVC.
4. D'Orazio, T., Ancona, N., Cicirelli, G. & Nitti, M. (2002). *A ball detection algorithm for real soccer image sequences*. IEEE ICPR.
5. Zheng, F., Al-Hamid, D.Z., Chong, P.H.J., Yang, C. & Li, X.J. (2025). *A review of computer vision technology for football videos*. Information, 16(5), 355.

---

## Technologies

- **Languages:** Python 3.9+
- **Computer Vision:** OpenCV, scikit-image
- **Machine Learning:** scikit-learn, PyTorch
- **Visualisation:** Matplotlib, Streamlit
- **Data:** NumPy, SciPy, pandas

---

## Acknowledgements

This project was completed as a Final Year Project at the University of Greenwich. The pipeline architecture is based on Mavrogiannis & Maglogiannis (2022), with camera calibration from Chen & Little (2019). Formation detection is an original contribution.

---

## Licence

This project is for academic purposes. Datasets are used under their respective licences (CC BY 4.0 for Roboflow, academic use for World Cup 2014 and ISSIA-CNR).