from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime
from custom_enums import VehicleDirection

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

class CameraCalibrationCreate(CameraCalibrationBase):
    pass

class CameraCalibration(CameraCalibrationBase):
    id: int
    speed_calibrations: List["SpeedCalibration"] = []
    
    class Config:
        from_attributes = True



class SpeedCalibrationBase(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    camera_calibration_id: int
    calibration_date: datetime

class SpeedCalibrationCreate(SpeedCalibrationBase):
    pass

class SpeedCalibrationUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    camera_calibration_id: Optional[int] = None
    calibration_date: Optional[datetime] = None

class SpeedCalibration(SpeedCalibrationBase):
    id: int
    camera_calibration: Optional[CameraCalibration] = None
    vehicle_detections: List["VehicleDetection"] = []
    
    class Config:
        from_attributes = True



class VehicleDetectionBase(BaseModel):
    detection_date: datetime
    video_clip_path: Optional[str] = None
    direction: Optional[VehicleDirection] = None
    estimated_speed: Optional[float] = Field(default=None, ge=0, description="Estimated speed must be non-negative")
    confidence: Optional[float] = Field(default=None, ge=0, description="Confidence must be non-negative")
    true_speed: Optional[float] = Field(default=None, ge=0, description="True speed must be non-negative")
    optical_flow_path: Optional[str] = None
    speed_calibration_id: int
    error: Optional[str] = None

class VehicleDetectionCreate(VehicleDetectionBase):
    pass

class VehicleDetectionUpdate(BaseModel):
    video_clip_path: Optional[str] = None
    estimated_speed: Optional[float] = Field(default=None, ge=0, description="Estimated speed must be non-negative")
    true_speed: Optional[float] = Field(default=None, ge=0, description="True speed must be non-negative")
    optical_flow_path: Optional[str] = None
    direction: Optional[VehicleDirection] = None
    confidence: Optional[float] = Field(default=None, ge=0, description="Confidence must be non-negative")
    error: Optional[str] = None

class VehicleDetection(VehicleDetectionBase):
    id: int
    speed_calibration: SpeedCalibration
    
    class Config:
        from_attributes = True



class LiveDetectionStateBase(BaseModel):
    speed_calibration_id: Optional[int] = None
    started_at: Optional[datetime] = None
    camera_id: Optional[int] = None
    running: bool = False
    error: Optional[str] = None

class LiveDetectionStateCreate(LiveDetectionStateBase):
    pass

class LiveDetectionStateUpdate(BaseModel):
    speed_calibration_id: Optional[int] = None
    started_at: Optional[datetime] = None
    running: Optional[bool] = None
    error: Optional[str] = None
    camera_id: Optional[int] = None

class LiveDetectionState(LiveDetectionStateBase):
    class Config:
        orm_mode = True