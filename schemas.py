from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import IntEnum

# Add VehicleDirection enum
class VehicleDirection(IntEnum):
    LEFT_TO_RIGHT = 0
    RIGHT_TO_LEFT = 1

class CameraCalibration(BaseModel):
    id: Optional[int] = None
    camera_name: str
    calibration_date: datetime
    image_paths: Optional[List[str]] = None
    calibration_matrix: dict
    distortion_coefficients: dict
    rotation_matrix: dict
    translation_vector: dict

    class Config:
        orm_mode = True

class SpeedCalibration(BaseModel):
    id: Optional[int] = None
    name: Optional[str] = None
    description: Optional[str] = None
    calibration_date: datetime
    camera_calibration_id: int
    vehicle_detections: List['VehicleDetection'] = []

    class Config:
        orm_mode = True

class VehicleDetection(BaseModel):
    id: Optional[int] = None
    detection_date: datetime
    video_clip_path: Optional[str] = None
    direction: Optional[VehicleDirection] = None
    estimated_speed: Optional[float] = None
    confidence: Optional[float] = None
    true_speed: Optional[float] = None
    optical_flow_path: Optional[str] = None
    speed_calibration_id: int
    error: Optional[str] = None

    class Config:
        orm_mode = True