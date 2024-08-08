from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from database import get_db
import models, schemas

router = APIRouter()

@router.post("/", response_model=schemas.VehicleDetection)
async def create_vehicle_detection(
    detection: schemas.VehicleDetectionCreate,
    db: Session = Depends(get_db)
):
    return models.create_vehicle_detection(db, detection)

@router.get("/", response_model=List[schemas.VehicleDetection])
async def list_vehicle_detections(
    skip: int = 0,
    limit: int = 100,
    speed_calibration_id: int = Query(..., description="ID of the SpeedCalibration to search within"),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    min_speed: Optional[float] = None,
    max_speed: Optional[float] = None,
    known_speed_only: bool = False,
    vehicle_ids: List[int] = Query(None, description="List of vehicle detection IDs to filter by"),
    db: Session = Depends(get_db)
):
    return models.get_vehicle_detections(
        db, skip, limit, speed_calibration_id, start_date, end_date,
        min_speed, max_speed, known_speed_only, vehicle_ids
    )

@router.get("/{detection_id}", response_model=schemas.VehicleDetection)
async def get_vehicle_detection(
    detection_id: int,
    db: Session = Depends(get_db)
):
    detection = models.get_vehicle_detection(db, detection_id)
    if detection is None:
        raise HTTPException(status_code=404, detail="Vehicle detection not found")
    return detection

@router.put("/{detection_id}", response_model=schemas.VehicleDetection)
async def update_vehicle_detection(
    detection_id: int,
    detection: schemas.VehicleDetectionUpdate,
    db: Session = Depends(get_db)
):
    updated_detection = models.update_vehicle_detection(db, detection_id, detection)
    if updated_detection is None:
        raise HTTPException(status_code=404, detail="Vehicle detection not found")
    return updated_detection