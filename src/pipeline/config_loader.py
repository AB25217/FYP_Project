"""
config_loader.py — Load configuration paths based on environment.
"""

import os
import yaml
from typing import Dict


def get_paths(config_path: str = None) -> Dict[str, str]:
    """
    Load paths from configs/paths.yaml, selecting the correct
    environment (Colab vs local) automatically.

    Args:
        config_path: path to paths.yaml. If None, searches common locations.

    Returns:
        dict: path configuration for the detected environment
    """
    # Find the config file
    if config_path is None:
        search_locations = [
            "configs/paths.yaml",
            "../configs/paths.yaml",
            "../../configs/paths.yaml",
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "paths.yaml"),
        ]
        for loc in search_locations:
            if os.path.exists(loc):
                config_path = loc
                break

    if config_path is None or not os.path.exists(config_path):
        print("Warning: paths.yaml not found, using default Colab paths")
        return _default_colab_paths()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Auto-detect environment
    if _is_colab():
        env = "colab"
    else:
        env = "local"

    paths = config.get(env, config.get("colab"))
    return paths


def _is_colab() -> bool:
    """Check if running inside Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def _default_colab_paths() -> Dict[str, str]:
    """Default paths for Colab if config file is missing."""
    root = "/content/drive/MyDrive/FYP_Project"
    return {
        "drive_root": root,
        "synthetic_data": f"{root}/synthetic_data",
        "weights": f"{root}/weights",
        "world_cup_2014": f"{root}/datasets/world_cup_2014",
        "roboflow_football": f"{root}/datasets/roboflow_football",
        "issia_cnr": f"{root}/datasets/issia_cnr",
        "player_clustering": f"{root}/datasets/player_clustering",
        "siamese_model": f"{root}/weights/siamese.pth",
        "field_unet_model": f"{root}/weights/field_unet.pth",
        "pose_database": f"{root}/weights/pose_database.npz",
        "svm_player_model": f"{root}/weights/svm_player.pkl",
    }


def verify_paths(paths: Dict[str, str]) -> None:
    """
    Check which paths exist and which are missing.
    Useful for debugging setup issues.
    """
    print("Path verification:")
    for key, path in paths.items():
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"  [{status:7s}] {key}: {path}")


if __name__ == "__main__":
    paths = get_paths()
    print(f"Environment: {'Colab' if _is_colab() else 'Local'}\n")
    verify_paths(paths)