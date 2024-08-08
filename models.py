from enum import IntEnum

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship


Base = declarative_base()

class VehicleDirection(IntEnum):
    LEFT_TO_RIGHT = 0
    RIGHT_TO_LEFT = 1

class CameraCalibration(Base):
    __tablename__ = 'camera_calibrations'

    id = Column(Integer, primary_key=True)
    camera_name = Column(String, nullable=False)
    calibration_date = Column(DateTime, nullable=False)
    image_paths = Column(JSON, nullable=True)
    calibration_matrix = Column(JSON, nullable=False)
    distortion_coefficients = Column(JSON, nullable=False)
    rotation_matrix = Column(JSON, nullable=False)
    translation_vector = Column(JSON, nullable=False)

    speed_calibrations = relationship("SpeedCalibration", back_populates="camera_calibration")

class SpeedCalibration(Base):
    __tablename__ = 'speed_calibrations'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(String)
    calibration_date = Column(DateTime, nullable=False)
    camera_calibration_id = Column(Integer, ForeignKey('camera_calibrations.id'), nullable=False)

    camera_calibration = relationship("CameraCalibration", back_populates="speed_calibrations")
    vehicle_detections = relationship("VehicleDetection", back_populates="speed_calibration")

class VehicleDetection(Base):
    __tablename__ = 'vehicle_detections'

    id = Column(Integer, primary_key=True)
    detection_date = Column(DateTime, nullable=False)
    video_clip_path = Column(String, nullable=True)
    direction = Column(Enum(VehicleDirection))
    estimated_speed = Column(Float)
    confidence = Column(Float)
    true_speed = Column(Float)
    optical_flow_path = Column(String)
    speed_calibration_id = Column(Integer, ForeignKey('speed_calibrations.id'), nullable=False)
    error = Column(String)

    speed_calibration = relationship("SpeedCalibration", back_populates="vehicle_detections")