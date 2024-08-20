from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Enum, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from custom_enums import VehicleDirection

Base = declarative_base()

class CameraCalibration(Base):
    __tablename__ = 'camera_calibrations'

    id = Column(Integer, primary_key=True)
    camera_name = Column(String, nullable=False)
    calibration_date = Column(DateTime, default=datetime.now)
    images_path = Column(String, nullable=True)
    rows = Column(Integer, nullable=False)
    cols = Column(Integer, nullable=False)
    calibration_matrix = Column(JSON, nullable=True)
    distortion_coefficients = Column(JSON, nullable=True)
    rotation_matrix = Column(JSON, nullable=True)
    translation_vector = Column(JSON, nullable=True)
    valid = Column(Boolean, default=False, nullable=False)

    speed_calibrations = relationship("SpeedCalibration", back_populates="camera_calibration")

class SpeedCalibration(Base):
    __tablename__ = 'speed_calibrations'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(String)
    calibration_date = Column(DateTime, nullable=False)
    camera_calibration_id = Column(Integer, ForeignKey('camera_calibrations.id'), nullable=True)

    camera_calibration = relationship("CameraCalibration", back_populates="speed_calibrations")
    vehicle_detections = relationship("VehicleDetection", back_populates="speed_calibration")

class VehicleDetection(Base):
    __tablename__ = 'vehicle_detections'

    id = Column(Integer, primary_key=True)
    detection_date = Column(DateTime, nullable=False)
    video_clip_path = Column(String, nullable=True)
    direction = Column(Enum(VehicleDirection), nullable=True)
    estimated_speed = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    true_speed = Column(Float, nullable=True)
    optical_flow_path = Column(String, nullable=True)
    speed_calibration_id = Column(Integer, ForeignKey('speed_calibrations.id'), nullable=False)
    error = Column(String, nullable=True)

    speed_calibration = relationship("SpeedCalibration", back_populates="vehicle_detections")
    
class LiveDetectionState(Base):
    __tablename__ = 'live_detection_state'
    id = Column(Integer, primary_key=True)
    speed_calibration_id = Column(Integer, nullable=True)
    started_at = Column(DateTime, nullable=True)
    running = Column(Boolean, default=False)
    camera_id = Column(Integer, nullable=True)
    error = Column(String, nullable=True)
