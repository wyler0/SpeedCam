from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from database import get_db
import models, schemas

router = APIRouter()

@router.post("/", response_model=schemas.SpeedCalibration)
async def create_speed_calibration(
    calibration: schemas.SpeedCalibrationCreate,
    db: Session = Depends(get_db)
):
    return models.create_speed_calibration(db, calibration)

@router.get("/", response_model=List[schemas.SpeedCalibration])
async def list_speed_calibrations(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    return models.get_speed_calibrations(db, skip, limit)

@router.get("/{calibration_id}", response_model=schemas.SpeedCalibration)
async def get_speed_calibration(
    calibration_id: int,
    db: Session = Depends(get_db)
):
    calibration = models.get_speed_calibration(db, calibration_id)
    if calibration is None:
        raise HTTPException(status_code=404, detail="Speed calibration not found")
    return calibration

@router.put("/{calibration_id}", response_model=schemas.SpeedCalibration)
async def update_speed_calibration(
    calibration_id: int,
    calibration: schemas.SpeedCalibrationUpdate,
    db: Session = Depends(get_db)
):
    updated_calibration = models.update_speed_calibration(db, calibration_id, calibration)
    if updated_calibration is None:
        raise HTTPException(status_code=404, detail="Speed calibration not found")
    return updated_calibration

@router.post("/{calibration_id}/vehicle-detections", response_model=schemas.VehicleDetection)
async def add_vehicle_detection_to_speed_calibration(
    calibration_id: int,
    detection: schemas.VehicleDetectionCreate,
    db: Session = Depends(get_db)
):
    return models.add_vehicle_detection_to_speed_calibration(db, calibration_id, detection)

# Additional helper function (implement in a separate utility module)
def load_image(path: str):
    # Implement image loading logic here
    pass