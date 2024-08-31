# © 2024 Wyler Zahm. All rights reserved.

import os

from AVFoundation import AVCaptureDevice
import cv2


def get_available_cameras():
    # Get all video devices (webcams)
    devices = AVCaptureDevice.devicesWithMediaType_("video")

    webcam_details = []

    for device in devices:
        device_name = device.localizedName()
        device_unique_id = device.uniqueID()
        
        # Force permission to access camera
        cv2.VideoCapture(device_unique_id)
        
        webcam_details.append((device_name, device_unique_id))

    return webcam_details
