import numpy as np

from sqlalchemy.orm import Session

from custom_enums import VehicleDirection
from models import VehicleDetection, SpeedCalibration


class SpeedCalibrationError(Exception):
    """Base class for calibration errors."""
    pass


def tune_distance_calibrations_constants(calibration_id: int, db: Session):
    left_to_right_detections = db.query(VehicleDetection).filter(VehicleDetection.speed_calibration_id == calibration_id, VehicleDetection.direction == VehicleDirection.LEFT_TO_RIGHT).all()
    right_to_left_detections = db.query(VehicleDetection).filter(VehicleDetection.speed_calibration_id == calibration_id, VehicleDetection.direction == VehicleDirection.RIGHT_TO_LEFT).all()
    
    if len(left_to_right_detections) < 2:
        raise SpeedCalibrationError("Not enough detections from left to right")
    if len(right_to_left_detections) < 2:
        raise SpeedCalibrationError("Not enough detections from right to left")
    
    speed_calibration = db.query(SpeedCalibration).filter(SpeedCalibration.id == calibration_id).first()
    
    if speed_calibration is None:
        raise SpeedCalibrationError("Speed calibration not found")
    
    if speed_calibration.left_to_right_constant is not None and speed_calibration.right_to_left_constant is not None:
        return True
    
    # Compute constants
    actual_and_estimated_speeds = [(clip.real_world_speed, clip.pixel_speed_estimate) for clip in left_to_right_detections]
    calibration_ratios_left_to_right_constant = [actual / estimate for actual, estimate in actual_and_estimated_speeds if estimate is not None]
    speed_calibration.left_to_right_constant = np.median(calibration_ratios_left_to_right_constant)

    # Calibrate estimates using actual speeds
    actual_and_estimated_speeds = [(clip.real_world_speed, clip.pixel_speed_estimate) for clip in right_to_left_detections]
    calibration_ratios_right_to_left_constant = [actual / estimate for actual, estimate in actual_and_estimated_speeds if estimate is not None]
    speed_calibration.right_to_left_constant = np.median(calibration_ratios_right_to_left_constant)

    db.commit()
    
    return True
