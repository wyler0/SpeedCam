import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from database import get_db
from models import SpeedCalibration, LiveDetectionState as LiveDetectionStateModel
from schemas import LiveDetectionState
from core.video.webcam import get_available_cameras

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/status", response_model=LiveDetectionState)
async def get_live_detection_status(db: Session = Depends(get_db)):
    try:
        state = await initialize_or_get_live_detection_state(db)
        if state:
            return state
        else:
            logger.error("Error getting live detection status, state DNE while getting status.")
            raise HTTPException(status_code=500, detail="No live detection found, which is not possible.")
    
    except SQLAlchemyError as e:
        logger.error("Error getting live detection status: %s", str(e))
        raise HTTPException(status_code=500, detail="Error getting live detection status")


@router.post("/start")
async def start_live_detection(db: Session = Depends(get_db)):
    try:
        state = await initialize_or_get_live_detection_state(db)
        
        if state:
            if state.running:
                return {"message": "Live detection is already running"}
            if state.speed_calibration_id is None:
                raise HTTPException(status_code=400, detail="No speed calibration found, which is required for live detection. Please set via /update endpoint.")
            state.started_at = datetime.now()
            state.running = True
        else:
            logger.error("Error getting live detection status, state DNE while starting.")
            raise HTTPException(status_code=500, detail="No live detection found, which is not possible.")
        
        db.commit()
        return {"message": "Live detection started"}
    
    except SQLAlchemyError as e:
        db.rollback()
        logger.error("Error starting live detection: %s", str(e))
        raise HTTPException(status_code=500, detail="Error starting live detection")

@router.post("/stop")
async def stop_live_detection(db: Session = Depends(get_db)):
    try:
        state = await initialize_or_get_live_detection_state(db)
        if state:
            if not state.running:
                return {"message": "Live detection is already stopped"}   
            
            state.running = False
            state.started_at = None
            db.commit()
            return {"message": "Live detection stopped"}
        else:
            logger.error("Error getting live detection status, state DNE while stopping.")
            raise HTTPException(status_code=500, detail="No live detection found, which is not possible.")
        
    except SQLAlchemyError as e:
        db.rollback()
        logger.error("Error stopping live detection: %s", str(e))
        raise HTTPException(status_code=500, detail="Error stopping live detection")

@router.put("/update")
async def update_live_detection(
    speed_calibration_id: Optional[int] = None,
    camera_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    try:
        state = await initialize_or_get_live_detection_state(db)
        if state is None:
            logger.error("Error getting live detection status, state DNE while updating.")
            raise HTTPException(status_code=500, detail="No live detection found, which is not possible.")

        if speed_calibration_id is not None:
            speed_calibration = db.query(SpeedCalibration).get(speed_calibration_id)
            if speed_calibration is None:
                raise HTTPException(status_code=404, detail="Speed calibration not found")
            state.speed_calibration_id = speed_calibration_id

        if camera_id is not None and camera_id != "":
            state.camera_id = camera_id
        elif camera_id == "":
            logger.error("Received empty camera_id in request body")
            raise HTTPException(status_code=400, detail="camera_id cannot be empty")

        if state.running:
            state.started_at = datetime.now()
        
        db.commit()
        return {"message": "Live detection updated"}
        
    except SQLAlchemyError as e:
        db.rollback()
        logger.error("Error updating live detection: %s", str(e))
        raise HTTPException(status_code=500, detail="Error updating live detection")

@router.get("/available_cameras")
async def list_available_cameras():
    try:
        devices = get_available_cameras()
        available_cameras = {}
        for device_index, device_name in enumerate(devices):
            available_cameras[device_index] = device_name[0]

        return {"available_cameras": available_cameras}
    
    except Exception as e:
        logger.error("Error listing available cameras: %s", str(e))
        raise HTTPException(status_code=500, detail="Error listing available cameras")


async def initialize_or_get_live_detection_state(db: Session):
    state = db.query(LiveDetectionStateModel).first()
    if state is None:
        state = LiveDetectionStateModel(speed_calibration_id=None, running=False, started_at=None, camera_id=None)
        db.add(state)
        db.commit()
    return state

