# © 2024 Wyler Zahm. All rights reserved.

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Any, TYPE_CHECKING, ForwardRef
from datetime import datetime
from custom_enums import VehicleDirection

if TYPE_CHECKING:
    from .models import SpeedCalibration, VehicleDetection

class CameraCalibrationBase(BaseModel):
    camera_name: str
    rows: int
    cols: int
    calibration_date: Optional[datetime] = None
    images_path: Optional[str] = None
    calibration_matrix: Optional[Any] = None
    distortion_coefficients: Optional[Any] = None
    rotation_matrix: Optional[Any] = None
    translation_vector: Optional[Any] = None
    valid: Optional[bool] = Field(default=False)
    horizontal_flip: Optional[bool] = Field(default=False)

class CameraCalibrationCreate(CameraCalibrationBase):
    pass

class CameraCalibration(CameraCalibrationBase):
    id: int
    speed_calibrations: List["SpeedCalibration"] = []
    
    model_config = ConfigDict(from_attributes=True)


class SpeedCalibrationBase(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    camera_calibration_id: int
    calibration_date: datetime
    valid: bool = Field(default=False)
    left_to_right_constant: Optional[float] = None
    right_to_left_constant: Optional[float] = None

class SpeedCalibrationCreate(SpeedCalibrationBase):
    pass

class SpeedCalibrationUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    camera_calibration_id: Optional[int] = None
    calibration_date: Optional[datetime] = None
    valid: Optional[bool] = None
    left_to_right_constant: Optional[float] = None
    right_to_left_constant: Optional[float] = None

class SpeedCalibration(SpeedCalibrationBase):
    id: int
    vehicle_detections: List["VehicleDetection"] = []
    
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)


class VehicleDetectionBase(BaseModel):
    detection_date: datetime
    thumbnail_path: Optional[str] = None
    direction: Optional[VehicleDirection] = None
    pixel_speed_estimate: Optional[float] = Field(default=None, ge=0, description="Pixel speed estimate must be non-negative")
    real_world_speed_estimate: Optional[float] = Field(default=None, ge=0, description="Real world speed estimate must be non-negative")
    real_world_speed: Optional[float] = Field(default=None, ge=0, description="Real world speed must be non-negative")
    optical_flow_path: Optional[str] = None
    speed_calibration_id: int
    error: Optional[str] = None

class VehicleDetectionCreate(VehicleDetectionBase):
    pass

class VehicleDetectionUpdate(BaseModel):
    thumbnail_path: Optional[str] = None
    pixel_speed_estimate: Optional[float] = Field(default=None, ge=0, description="Pixel speed estimate must be non-negative")
    real_world_speed_estimate: Optional[float] = Field(default=None, ge=0, description="Real world speed estimate must be non-negative")
    real_world_speed: Optional[float] = Field(default=None, ge=0, description="Real world speed must be non-negative")
    optical_flow_path: Optional[str] = None
    direction: Optional[VehicleDirection] = None
    error: Optional[str] = None

class VehicleDetection(VehicleDetectionBase):
    id: int
    
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

VehicleDetection.model_rebuild()

class LiveDetectionStateBase(BaseModel):
    speed_calibration_id: Optional[int] = None
    started_at: Optional[datetime] = None
    is_calibrating: Optional[bool] = Field(default=False)
    running: bool = False
    camera_source: Optional[int] = None
    error: Optional[str] = None
    has_new_image: bool = Field(default=False)
    video_path: Optional[str] = None
    processing_video: Optional[bool] = Field(default=False)

class LiveDetectionStateCreate(LiveDetectionStateBase):
    pass

class LiveDetectionStateUpdate(BaseModel):
    speed_calibration_id: Optional[int] = None
    camera_source: Optional[int] = None
    started_at: Optional[datetime] = None
    running: Optional[bool] = None
    error: Optional[str] = None
    is_calibrating: Optional[bool] = Field(default=False)
    has_new_image: Optional[bool] = Field(default=False)
    
    video_path: Optional[str] = None
    processing_video: Optional[bool] = Field(default=False)
    
class LiveDetectionState(LiveDetectionStateBase):
    id: int

    model_config = ConfigDict(from_attributes=True)