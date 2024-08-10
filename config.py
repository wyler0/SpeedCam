import os
def get_root_directory() -> str:
    """Get the root directory of the FastAPI application."""
    return os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = get_root_directory()


# Upload data paths
DATA_ROOT_DIR = os.path.join(ROOT_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_ROOT_DIR, "uploads")
CALIBRATION_DATA_PATH = os.path.join(DATA_ROOT_DIR, "calibration_data")
VIDEO_DATA_PATH = os.path.join(DATA_ROOT_DIR, "video_data")



