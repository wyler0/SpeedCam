# © 2024 Wyler Zahm. All rights reserved.

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Upload data paths
DATA_ROOT_DIR = os.path.join(ROOT_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_ROOT_DIR, "uploads")
CALIBRATION_DATA_PATH = os.path.join(DATA_ROOT_DIR, "calibration_data")
DETECTIONS_DATA_PATH = os.path.join(DATA_ROOT_DIR, "detections_data")
LATEST_DETECTION_IMAGE_PATH = os.path.join(DATA_ROOT_DIR, "latest_detection_image.png")
VIDEO_DATA_PATH = os.path.join(DATA_ROOT_DIR, "video_data")



