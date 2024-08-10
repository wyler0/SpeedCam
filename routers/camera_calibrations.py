from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session

from database import get_db
import models, schemas
from utils import load_image

router = APIRouter()

@router.post("/", response_model=schemas.CameraCalibration)
async def create_camera_calibration(
    calibration: schemas.CameraCalibrationCreate,
    db: Session = Depends(get_db)
):
    db_calibration = models.CameraCalibration(**calibration.dict())
    db.add(db_calibration)
    db.commit()
    db.refresh(db_calibration)
    return db_calibration

@router.get("/", response_model=List[schemas.CameraCalibration])
async def list_camera_calibrations(
    skip: int = 0,
    limit: int = 100,
    id: Optional[str] = None,
    camera_name: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    query = db.query(models.CameraCalibration)
    if id:
        query = query.filter(models.CameraCalibration.id == id)
    if camera_name:
        query = query.filter(models.CameraCalibration.camera_name == camera_name)
    if start_date and end_date:
        query = query.filter(models.CameraCalibration.calibration_date.between(start_date, end_date))
    return query.offset(skip).limit(limit).all()

@router.get("/{calibration_id}", response_model=schemas.CameraCalibration)
async def get_camera_calibration(
    calibration_id: int,
    include_images: bool = False,
    db: Session = Depends(get_db)
):
    calibration = db.query(models.CameraCalibration).get(calibration_id)
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
    calibration = db.query(models.CameraCalibration).get(calibration_id)
    if calibration is None:
        raise HTTPException(status_code=404, detail="Camera calibration not found")
    
    file_path = f"calibration_images/{calibration_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    calibration.image_paths.append(file_path)
    db.commit()
    
    return {"message": "Image uploaded successfully"}
