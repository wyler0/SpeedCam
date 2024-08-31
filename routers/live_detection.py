import logging
from datetime import datetime
from typing import Optional
import multiprocessing
import uuid, os, shutil

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload


from database import get_db
from models import SpeedCalibration, LiveDetectionState as LiveDetectionStateModel, CameraCalibration
from schemas import LiveDetectionState, SpeedCalibrationCreate
from routers.speed_calibrations import create_speed_calibration, delete_speed_calibration
from core.estimation.estimator import SpeedEstimator, EstimatorConfig
from core.video.webcam_utils import get_available_cameras
from config import UPLOADS_DIR

logger = logging.getLogger(__name__)

stop_event: Optional[multiprocessing.Event] = None
estimator: SpeedEstimator = SpeedEstimator(EstimatorConfig())

router = APIRouter()
@router.get("/status", response_model=LiveDetectionState)
async def get_live_detection_status(db: Session = Depends(get_db)
):
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
async def start_live_detection(db: Session = Depends(get_db)
):
    try:
        state = await initialize_or_get_live_detection_state(db)
        
        if state:
            if state.running:
                return {"message": "Live detection is already running"}
            if state.speed_calibration_id is None:
                raise HTTPException(status_code=400, detail="No speed calibration found, which is required for live detection. Please set via /update endpoint.")
            if state.video_path is not None and state.camera_source is not None:
                state.video_path = None
                state.camera_source = None
                raise HTTPException(status_code=400, detail="Cannot update both camera_source and video_path. Please reconfigure live detection state.")
            
            state.started_at = datetime.now()
            state.running = True
            
            # Configure the estimator
                
            estimator.config.input_video = state.camera_source if state.camera_source is not None else state.video_path
            if state.video_path is not None and state.camera_source is None:
                state.processing_video = True
            
            # Start the estimator
            estimator.start()
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
async def stop_live_detection(db: Session = Depends(get_db)
):
    try:
        state = await initialize_or_get_live_detection_state(db)
        if state:
            if not state.running:
                return {"message": "Live detection is already stopped"}   
        
            db.commit()
            
            # Stop the estimator
            estimator.stop()
            state.running = False
            state.started_at = None
            
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
    camera_calibration_id: Optional[int] = None,
    camera_source: Optional[str] = None,
    video_path: Optional[str] = None,
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

        if camera_calibration_id is not None:
            camera_calibration: Optional[CameraCalibration] = db.query(CameraCalibration).get(camera_calibration_id)
            if camera_calibration is None:
                raise HTTPException(status_code=404, detail="Camera calibration not found")
            speed_calibration: Optional[SpeedCalibration] = db.query(SpeedCalibration).get(state.speed_calibration_id)
            if speed_calibration is None:
                raise HTTPException(status_code=404, detail="Speed calibration not found, cannot update live detection state with selected camera calibration.")
            speed_calibration.camera_calibration_id = camera_calibration_id

        if camera_source is not None:
            if camera_source == "":
                logger.error("Received empty camera_source in request body")
                raise HTTPException(status_code=400, detail="Camera source cannot be empty, only null or a valid camera index is allowed.")
            
            if video_path is not None and video_path != "":
                logger.error("Received both camera_source and video_path")
                raise HTTPException(status_code=400, detail="Cannot update both camera_source and video_path")
            
            state.camera_source = camera_source
            state.video_path = None
        
        if video_path is not None:
            if video_path == "":
                logger.error("Received empty video_path in request body")
                raise HTTPException(status_code=400, detail="Video path cannot be empty, only null or a valid video path is allowed.")
            state.camera_source = None
            state.video_path = video_path
        
        if state.running:
            await stop_live_detection(db)
            await start_live_detection(db)
        
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

@router.get("/latest_image_status")
async def get_latest_image_status(db: Session = Depends(get_db)
):
    try:
        state = await initialize_or_get_live_detection_state(db)
        
        response = {
            "has_new_image": state.has_new_image,
            "image_url": "/static/latest_detection_image.png" 
        }
        
        if state.has_new_image:
            state.has_new_image = False 
            db.commit()
        
        return response
    
    except SQLAlchemyError as e:
        logger.error("Error retrieving latest image status: %s", str(e))
        raise HTTPException(status_code=500, detail="Error retrieving latest image status")


@router.put("/calibration_mode")
async def set_calibration_mode(is_calibrating: bool, db: Session = Depends(get_db)
):
    try:
        state = await initialize_or_get_live_detection_state(db)
        if state is None:
            logger.error("Error getting live detection status, state DNE while updating calibration mode.")
            raise HTTPException(status_code=500, detail="No live detection found, which is not possible.")

        state.is_calibrating = is_calibrating
        
        # If live detection is running, stop it.
        if state.running:
            await stop_live_detection(db)
        
        # Create a speed calibration if is_calibrating is True
        if is_calibrating:
            speed_calibration = await create_speed_calibration(
                SpeedCalibrationCreate(
                    name = "New Speed Calibration",
                    description = "New Speed Calibration",
                    calibration_date = datetime.now(),
                    camera_calibration_id = -1,
                    valid = False,
                    left_to_right_constant = None,
                    right_to_left_constant = None
                ),
                db=db
            )
            
            # Update the live detection state with the new speed calibration id
            state.speed_calibration_id = speed_calibration.id
            db.commit()
            
        # Delete the speed calibration if is_calibrating is False and calibration_id is not None and not valid
        elif state.speed_calibration_id is not None:
            speed_calibration = db.query(SpeedCalibration).options(joinedload(SpeedCalibration.vehicle_detections)).get(state.speed_calibration_id)
            if speed_calibration is not None and not speed_calibration.valid:
                await delete_speed_calibration(speed_calibration.id, db)
                
            # Update the live detection state with the new speed calibration id
            state.speed_calibration_id = None
            db.commit()

        if is_calibrating:
            return {"message": "Calibration mode enabled and speed calibration created.", "speed_calibration_id": speed_calibration.id}
        else:
            return {"message": "Calibration mode disabled and newly created speed calibration deleted if invalid."}
        
    except SQLAlchemyError as e:
        db.rollback()
        logger.error("Error updating calibration mode: %s", str(e))
        raise HTTPException(status_code=500, detail="Error updating calibration mode")



@router.post("/{calibration_id}/upload-video")
async def upload_speed_calibration_video(
    calibration_id: int,
    video: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    state = db.query(LiveDetectionStateModel).first()
    if state is None:
        raise HTTPException(status_code=404, detail="Live detection state not found. Please only use the upload endpoint when creating a new speed calibration.")
    
    if state.speed_calibration_id != calibration_id:
        raise HTTPException(status_code=400, detail="Speed calibration id mismatch. Please only use the upload endpoint when creating a new speed calibration.")
    
    speed_calibration: Optional[SpeedCalibration] = db.query(SpeedCalibration).get(state.speed_calibration_id)
    if speed_calibration is None or speed_calibration.camera_calibration is None:
        raise HTTPException(status_code=400, detail="Please set a camera calibration before uploading a video.")
    
    # Fetch the video to disk
    video_id = uuid.uuid4()
    video_path = f"{UPLOADS_DIR}/{calibration_id}"
    os.makedirs(video_path, exist_ok=True)
    video_path = f"{video_path}/{video_id}.mp4"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    
    await update_live_detection(video_path=video_path, db=db)
    
    if not state.running:
        await start_live_detection(db)

async def initialize_or_get_live_detection_state(db: Session):
    state = db.query(LiveDetectionStateModel).first()
    if state is None:
        state = LiveDetectionStateModel(speed_calibration_id=None, running=False, started_at=None, camera_source=None)
        db.add(state)
        db.commit()
    return state
