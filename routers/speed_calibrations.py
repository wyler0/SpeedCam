# © 2024 Wyler Zahm. All rights reserved.

from typing import List
import os, shutil

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db

import models, schemas
from routers.vehicle_detections import delete_vehicle_detection
from core.calibration.speed_calibration_utils import tune_distance_calibrations_constants, SpeedCalibrationError
from config import DETECTIONS_DATA_PATH

router = APIRouter()

@router.post("/", response_model=schemas.SpeedCalibration)
async def create_speed_calibration(
    calibration: schemas.SpeedCalibrationCreate,
    db: Session = Depends(get_db)
):
    db_calibration = models.SpeedCalibration(**calibration.model_dump())
    db.add(db_calibration)
    db.commit()
    db.refresh(db_calibration)
    return db_calibration

@router.delete("/{calibration_id}", response_model=schemas.SpeedCalibration)
async def delete_speed_calibration(
    calibration_id: int,
    db: Session = Depends(get_db)
):
    
    # Delete the vehicle detections
    vehicle_detections = db.query(models.VehicleDetection).filter(models.VehicleDetection.speed_calibration_id == calibration_id).all()
    for detection in vehicle_detections:
        await delete_vehicle_detection(detection.id, db)
    
    db_spd_calib = db.query(models.SpeedCalibration).get(calibration_id)
    if db_spd_calib is None:
        raise HTTPException(status_code=404, detail="Speed calibration not found")
    
    # Delete the speed disk data
    if os.path.exists(os.path.join(DETECTIONS_DATA_PATH, str(db_spd_calib.id))):
        shutil.rmtree(os.path.join(DETECTIONS_DATA_PATH, str(db_spd_calib.id)))
    
    db.delete(db_spd_calib)
    db.commit()
    
    return db_spd_calib


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
    
    for key, value in calibration.model_dump(exclude_unset=True).items():
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


@router.post("/{calibration_id}/submit", response_model=schemas.SpeedCalibration)
async def submit_speed_calibration(calibration_id: int, db: Session = Depends(get_db)
):
    try:
        success = tune_distance_calibrations_constants(calibration_id, db)
    except SpeedCalibrationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    db_calibration: models.SpeedCalibration = db.query(models.SpeedCalibration).get(calibration_id)
    if not success:
        db_calibration.valid = False
        db_calibration.left_to_right_constant = None
        db_calibration.right_to_left_constant = None
        db.commit()
        db.refresh(db_calibration)
        raise HTTPException(status_code=400, detail="Speed calibration failed.")
    else:
        db_calibration.valid = True
        db.commit()

    return db_calibration

