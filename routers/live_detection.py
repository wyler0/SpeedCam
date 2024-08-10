import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from database import get_db
from models import SpeedCalibration, LiveDetectionState as LiveDetectionStateModel
from schemas import LiveDetectionState, LiveDetectionStateCreate, LiveDetectionStateUpdate

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
    speed_calibration_id: int,
    db: Session = Depends(get_db)
):
    try:
        speed_calibration = db.query(SpeedCalibration).get(speed_calibration_id)
        if speed_calibration is None:
            raise HTTPException(status_code=404, detail="Speed calibration not found")
        
        state = await initialize_or_get_live_detection_state(db)
        if state:
            state.speed_calibration_id = speed_calibration_id
            if state.running:
                state.started_at = datetime.now()
            db.commit()
            return {"message": "Live detection updated"}
        else:
            logger.error("Error getting live detection status, state DNE while updating.")
            raise HTTPException(status_code=500, detail="No live detection found, which is not possible.")
        
    except SQLAlchemyError as e:
        db.rollback()
        logger.error("Error updating live detection: %s", str(e))
        raise HTTPException(status_code=500, detail="Error updating live detection")



async def initialize_or_get_live_detection_state(db: Session):
    state = db.query(LiveDetectionStateModel).first()
    if state is None:
        state = LiveDetectionStateModel(speed_calibration_id=None, running=False, started_at=None)
        db.add(state)
        db.commit()
    return state

