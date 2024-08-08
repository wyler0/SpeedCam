from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from database import get_db
import models, schemas
import os

router = APIRouter()

@router.post("/", response_model=schemas.CameraCalibration)
async def create_camera_calibration(
    calibration: schemas.CameraCalibrationCreate,
    db: Session = Depends(get_db)
):
    return models.create_camera_calibration(db, calibration)

@router.get("/", response_model=List[schemas.CameraCalibration])
async def list_camera_calibrations(
    skip: int = 0,
    limit: int = 100,
    camera_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    return models.get_camera_calibrations(db, skip, limit, camera_id, start_date, end_date)

@router.get("/{calibration_id}", response_model=schemas.CameraCalibration)
async def get_camera_calibration(
    calibration_id: int,
    include_images: bool = False,
    db: Session = Depends(get_db)
):
    calibration = models.get_camera_calibration(db, calibration_id)
    if calibration is None:
        raise HTTPException(status_code=404, detail="Camera calibration not found")
    
    if include_images:
        calibration.images = [load_image(path) for path in calibration.image_paths]
    
    return calibration

@router.post("/{calibration_id}/images")
async def upload_calibration_image(
    calibration_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    calibration = models.get_camera_calibration(db, calibration_id)
    if calibration is None:
        raise HTTPException(status_code=404, detail="Camera calibration not found")
    
    file_path = f"calibration_images/{calibration_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    calibration.image_paths.append(file_path)
    db.commit()
    
    return {"message": "Image uploaded successfully"}
