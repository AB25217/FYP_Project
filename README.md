# Analysis of match footage using computer vision techniques — Match Analysis with Formation Detection

An end-to-end computer vision pipeline that processes broadcast football video to detect players, track positions, identify teams, project positions onto a 2D pitch, and classify team formations — all trained from scratch with no pretrained models.

**Final Year Project** — University of Greenwich, 2025/26

---

## Overview

This project builds a complete football analytics system accessible to amateur clubs who cannot afford commercial tracking systems (Opta, StatsBomb, Hawk-Eye) and to further test out capabilities in match footage analysis using computer vision techniques. Given a standard broadcast video, the pipeline:

1. **Detects the pitch** and estimates the camera pose
2. **Detects and tracks** players and the ball frame-by-frame
3. **Identifies teams** by jersey colour
4. **Projects positions** onto a top-down 2D pitch view
5. **Classifies formations** (e.g. 4-3-3, 4-4-2) — the original research contribution

The system achieves this using only a single broadcast camera feed, with all models trained from scratch on publicly available datasets.


## Modules

| Module | Method | Training | Reference |
|--------|--------|----------|-----------|
| Court Detection | HSV thresholding + Hough Lines | No | 
| Camera Calibration | Siamese CNN + U-Net + ECC | Yes — Colab GPU |  
| Player Detection | HOG + SVM | Yes — CPU | Dalal & Triggs (2005) |nsform | No | 
| Object Tracking | Pyramidal Lucas-Kanade | No |
| Team Clustering | HSV histograms + K-means | No |
| Perspective Transform | Homography projection | No |
| Formation Detection | K-means lines + Hungarian matching| 

---

### Prerequisites

- Python 3.9+
- pip

### Installation

### Dataset Setup

All datasets are stored on Google Drive due to size constraints:
[Google Drive Folder](https://drive.google.com/drive/folders/17V4eQzGxwpuX09RTAcQAnztrFD7qOwWq)

See `data/README.md` for full download instructions and folder structure.

| Dataset | Purpose | Source |
|---------|---------|--------|
| World Cup 2014 | Camera calibration | [nhoma.github.io](https://nhoma.github.io/data/soccer_data.tar.gz) |
| Roboflow Football | Player/ball detection | [Roboflow Universe](https://universe.roboflow.com/yolo-pw0go/football-and-player) (CC BY 4.0) |
| ISSIA-CNR | Detection evaluation | [Google Drive](https://drive.google.com/file/d/1Pj6syLRShNQWQaunJmAZttUw2jDh8L_f/view) |


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

--

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
- **Offline processing** — designed for post-match analysis (3 FPS target, not real-time due to technological constraints)

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

This project was completed as a Final Year Project at the University of Greenwich.
---

## Licence

This project is for academic purposes. Datasets are used under their respective licences (CC BY 4.0 for Roboflow, academic use for World Cup 2014 and ISSIA-CNR).