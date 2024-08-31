import os, shutil

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta

from database import get_db
import models, schemas
from config import DETECTIONS_DATA_PATH

router = APIRouter()

# @router.post("/", response_model=schemas.VehicleDetection)
# async def create_vehicle_detection(
#     detection: schemas.VehicleDetectionCreate,
#     db: Session = Depends(get_db)
# ):
#     db_detection = models.VehicleDetection(**detection.dict())
#     db.add(db_detection)
#     db.commit()
#     db.refresh(db_detection)
#     return db_detection

@router.get("/", response_model=List[schemas.VehicleDetection])
async def list_vehicle_detections(
    skip: int = 0,
    limit: int = 100,
    speed_calibration_id: int = Query(..., description="ID of the SpeedCalibration to search within"),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    min_speed: Optional[float] = None,
    max_speed: Optional[float] = None,
    direction: Optional[schemas.VehicleDirection] = None,
    known_speed_only: bool = False,
    vehicle_ids: List[int] = Query(None, description="List of vehicle detection IDs to filter by"),
    predefined_filter: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(models.VehicleDetection)
    query = query.filter(models.VehicleDetection.speed_calibration_id == speed_calibration_id)
    
    if predefined_filter:
        now = datetime.utcnow()
        if predefined_filter == 'TODAY':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = now
        elif predefined_filter == 'LAST_7_DAYS':
            start_date = now - timedelta(days=7)
            end_date = now
        elif predefined_filter == 'LAST_30_DAYS':
            start_date = now - timedelta(days=30)
            end_date = now
        elif predefined_filter == 'SPEEDING':
            min_speed = 30  # Assuming 30 is the speed limit
        elif predefined_filter == 'KNOWN_SPEED':
            known_speed_only = True

    if start_date and end_date:
        query = query.filter(models.VehicleDetection.detection_date.between(start_date, end_date))
    if min_speed is not None and max_speed is not None:
        query = query.filter(models.VehicleDetection.estimated_speed.between(min_speed, max_speed))
    elif min_speed is not None:
        query = query.filter(models.VehicleDetection.estimated_speed >= min_speed)
    elif max_speed is not None:
        query = query.filter(models.VehicleDetection.estimated_speed <= max_speed)
    if known_speed_only:
        query = query.filter(models.VehicleDetection.true_speed.isnot(None))
    if vehicle_ids:
        query = query.filter(models.VehicleDetection.id.in_(vehicle_ids))
    if direction:
        query = query.filter(models.VehicleDetection.direction == direction)

    # Update the thumbnail path to be relative to the detections static mount
    detections = query.offset(skip).limit(limit).all()
    for detection in detections:
        detection.thumbnail_path = f"detection/{detection.thumbnail_path.split('detections_data/')[1]}"
        
    return detections

@router.get("/{detection_id}", response_model=schemas.VehicleDetection)
async def get_vehicle_detection(
    detection_id: int,
    db: Session = Depends(get_db)
):
    detection: Optional[models.VehicleDetection] = db.query(models.VehicleDetection).get(detection_id)
    if detection is None:
        raise HTTPException(status_code=404, detail="Vehicle detection not found")
    
    detection.thumbnail_path = f"detection/{detection.thumbnail_path.split('detections_data/')[1]}"
    return detection

@router.put("/{detection_id}", response_model=schemas.VehicleDetection)
async def update_vehicle_detection(
    detection_id: int,
    detection: schemas.VehicleDetectionUpdate,
    db: Session = Depends(get_db)
):
    db_detection = db.query(models.VehicleDetection).get(detection_id)
    if db_detection is None:
        raise HTTPException(status_code=404, detail="Vehicle detection not found")
    
    for key, value in detection.dict(exclude_unset=True).items():
        setattr(db_detection, key, value)
    
    db.commit()
    db.refresh(db_detection)
    return db_detection

@router.delete("/{detection_id}", response_model=schemas.VehicleDetection)
async def delete_vehicle_detection(
    detection_id: int,
    db: Session = Depends(get_db)
):
    db_detection = db.query(models.VehicleDetection).get(detection_id)
    if db_detection is None:
        raise HTTPException(status_code=404, detail="Vehicle detection not found")
    
    # Delete the detection data
    if db_detection.thumbnail_path:
        detection_dir = "/".join(db_detection.thumbnail_path.split("/")[:-1])
        if os.path.exists(os.path.join(DETECTIONS_DATA_PATH, detection_dir)):
            shutil.rmtree(os.path.join(DETECTIONS_DATA_PATH, detection_dir))
    
    # Delete the detection from the database
    db.delete(db_detection)
    db.commit()
    
    # Delete the image as well.
    return db_detection
