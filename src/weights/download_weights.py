"""download_weights.py — fetch pretrained model weights from Google Drive.

Downloads the four weight files needed to run inference:
  - siamese.pth              (camera calibration encoder)
  - pose_database.npz        (feature-pose retrieval database)
  - seg_latest_net_G.pth     (two-GAN: field segmentation generator)
  - detec_latest_net_G.pth   (two-GAN: field line detection generator)

Safe to re-run — files that already exist are skipped.

Usage:
    python src/weights/download_weights.py
"""
import os
import sys
from pathlib import Path

try:
    import gdown
except ImportError:
    print("gdown not installed. Run: pip install gdown")
    sys.exit(1)


# Resolve paths relative to this script, so it works regardless of where it's run from
SCRIPT_DIR = Path(__file__).resolve().parent        # src/weights/
REPO_ROOT = SCRIPT_DIR.parent.parent                # FYP_Project/
WEIGHTS_DIR = SCRIPT_DIR                            # src/weights/
GAN_WEIGHTS_DIR = SCRIPT_DIR / "gan_weights"        # src/weights/gan_weights/

WEIGHTS = {
    # Siamese + pose database go in src/weights/
    WEIGHTS_DIR / "siamese.pth":       "1yOIZyl-K38v4YAAldwOjii5Mr5Ny-5CZ",
    WEIGHTS_DIR / "pose_database.npz": "1Q_8ypSy9vbxk_vPY05d0Na-34RZVcZNx",

    # Two-GAN weights go in src/weights/gan_weights/
    GAN_WEIGHTS_DIR / "seg_latest_net_G.pth":   "1L5AqjOW_yun_AYGgoTHb_GcWMkomZIod",
    GAN_WEIGHTS_DIR / "detec_latest_net_G.pth": "14Ghtqi5v48oPmCvIheDBf23QJt_TDIiZ",
}


def main():
    # Ensure target directories exist
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    GAN_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    total = len(WEIGHTS)
    skipped = 0
    downloaded = 0

    for dest_path, file_id in WEIGHTS.items():
        if dest_path.exists() and dest_path.stat().st_size > 0:
            print(f"[skip] {dest_path.name} already present ({dest_path.stat().st_size / 1e6:.1f} MB)")
            skipped += 1
            continue

        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"[get]  {dest_path.name}")
        try:
            gdown.download(url, str(dest_path), quiet=False)
            downloaded += 1
        except Exception as e:
            print(f"[FAIL] {dest_path.name}: {e}")
            print(f"       Ensure the file at {url} is shared as 'Anyone with the link'")
            sys.exit(1)

    print(f"\nDone. {downloaded} downloaded, {skipped} already present, {total} total.")


if __name__ == "__main__":
    main()