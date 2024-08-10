from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db

import models, schemas
from config import CALIBRATION_DATA_PATH

router = APIRouter()

@router.post("/", response_model=schemas.SpeedCalibration)
async def create_speed_calibration(
    calibration: schemas.SpeedCalibrationCreate,
    db: Session = Depends(get_db)
):
    db_calibration = models.SpeedCalibration(**calibration.dict())
    db.add(db_calibration)
    db.commit()
    db.refresh(db_calibration)
    return db_calibration

@router.get("/", response_model=List[schemas.SpeedCalibration])
async def list_speed_calibrations(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    return db.query(models.SpeedCalibration).offset(skip).limit(limit).all()

@router.get("/{calibration_id}", response_model=schemas.SpeedCalibration)
async def get_speed_calibration(
    calibration_id: int,
    db: Session = Depends(get_db)
):
    calibration = db.query(models.SpeedCalibration).get(calibration_id)
    if calibration is None:
        raise HTTPException(status_code=404, detail="Speed calibration not found")
    return calibration

@router.put("/{calibration_id}", response_model=schemas.SpeedCalibration)
async def update_speed_calibration(
    calibration_id: int,
    calibration: schemas.SpeedCalibrationUpdate,
    db: Session = Depends(get_db)
):
    db_calibration = db.query(models.SpeedCalibration).get(calibration_id)
    if db_calibration is None:
        raise HTTPException(status_code=404, detail="Speed calibration not found")
    
    for key, value in calibration.dict(exclude_unset=True).items():
        setattr(db_calibration, key, value)
    
    db.commit()
    db.refresh(db_calibration)
    return db_calibration

@router.post("/{calibration_id}/vehicle-detections", response_model=schemas.VehicleDetection)
async def add_vehicle_detection_to_speed_calibration(
    calibration_id: int,
    detection: schemas.VehicleDetectionCreate,
    db: Session = Depends(get_db)
):
    db_calibration = db.query(models.SpeedCalibration).get(calibration_id)
    if db_calibration is None:
        raise HTTPException(status_code=404, detail="Speed calibration not found")
    
    db_detection = models.VehicleDetection(**detection.model_dump_json())
    db_detection.speed_calibration = db_calibration
    db.add(db_detection)
    db.commit()
    db.refresh(db_detection)
    return db_detection
