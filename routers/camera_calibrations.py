# © 2024 Wyler Zahm. All rights reserved.

from typing import List, Optional
from datetime import datetime
from uuid import uuid4
import shutil
import os

import asyncio
import cv2
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi.responses import FileResponse

import models, schemas
from database import get_db
from utils import load_image, save_image
from config import CALIBRATION_DATA_PATH
from core.calibration.camera_calibration_utils import find_corners, get_calibration_matrix

router = APIRouter()

@router.post("/", response_model=schemas.CameraCalibration)
async def create_camera_calibration(
    calibration: schemas.CameraCalibrationCreate,
    db: Session = Depends(get_db)
):
    # Check if a calibration with the same camera name already exists
    existing_calibration = db.query(models.CameraCalibration).filter(
        models.CameraCalibration.camera_name == calibration.camera_name
    ).first()
    
    if existing_calibration:
        raise HTTPException(status_code=400, detail="A calibration with this camera name already exists")
    
    db_calibration = models.CameraCalibration(**calibration.model_dump(exclude_unset=True))
    db.add(db_calibration)
    try:
        db.commit()
        db.refresh(db_calibration)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="A calibration with this camera name already exists")
    return db_calibration

@router.delete("/{calibration_id}")
async def delete_camera_calibration(
    calibration_id: int,
    db: Session = Depends(get_db)
):
    calibration: models.CameraCalibration = db.query(models.CameraCalibration).get(calibration_id)
    if calibration is None:
        raise HTTPException(status_code=404, detail="Camera calibration not found")
    
    # Delete the images folder
    images_path = os.path.join(CALIBRATION_DATA_PATH, str(calibration_id))
    if os.path.exists(images_path):
        shutil.rmtree(images_path)
    
    db.delete(calibration)
    db.commit()
    return {"message": "Camera calibration deleted successfully"}

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
    include_thumbnail: bool = False,
    include_images: bool = False,
    db: Session = Depends(get_db)
):
    calibration: models.CameraCalibration = db.query(models.CameraCalibration).get(calibration_id)
    if calibration is None:
        raise HTTPException(status_code=404, detail="Camera calibration not found")
    
    if include_thumbnail:
        image = os.listdir(calibration.images_path)[0]  
        calibration.thumbnail = load_image(f"{calibration.images_path}/{image}")
        
    if include_images:
        calibration.images = [load_image(path) for path in calibration.images_path]
    
    return calibration

@router.post("/{calibration_id}/upload-image")
async def upload_calibration_image(
    calibration_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    calibration: models.CameraCalibration = db.query(models.CameraCalibration).get(calibration_id)
    if calibration is None:
        raise HTTPException(status_code=404, detail="Camera calibration not found")
    
    img_id = uuid4().hex
    file_path = f"{CALIBRATION_DATA_PATH}/{calibration_id}/{''.join(file.filename.split('.')[:-1])}_{img_id}.jpg"
    await save_image(file_path, file)
    
    if calibration.horizontal_flip:
        img = cv2.imread(file_path)
        cv2.flip(img, 1)
        cv2.imwrite(file_path, img)
    
    calibration.images_path = f"{CALIBRATION_DATA_PATH}/{calibration_id}"
    db.commit()
    
    return {"message": "Image uploaded successfully"}

@router.post("/validate")
async def validate_calibration_image(
    file: UploadFile = File(...),
    rows: int = Form(...),
    columns: int = Form(...)
):
    # Read the uploaded image
    contents = await file.read()
    
    # Call the find_corners function
    corners_found = find_corners(contents, rows, columns)
    
    return {
        "corners_found": corners_found
    }
    
@router.post("/{calibration_id}/process")
async def process_calibration_image(
    calibration_id: int,
    db: Session = Depends(get_db)
):
    calibration: models.CameraCalibration = db.query(models.CameraCalibration).get(calibration_id)
    if calibration is None:
        raise HTTPException(status_code=404, detail="Camera calibration not found")
    
    image_path = os.path.join(CALIBRATION_DATA_PATH, str(calibration_id))
    try:
        # Run get_calibration_matrix in a separate thread to avoid blocking
        matrix, distortion, r_vecs, t_vecs = await asyncio.to_thread(
            get_calibration_matrix, 
            calibration.rows, 
            calibration.cols, 
            image_path
        )
        calibration.calibration_matrix = matrix.tolist()  # Convert numpy array to list
        calibration.distortion_coefficients = distortion.tolist()
        calibration.rotation_matrix = [rv.tolist() for rv in r_vecs]
        calibration.translation_vector = [tv.tolist() for tv in t_vecs]
        calibration.valid = True
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    db.commit()
    
    return {
        "message": "Calibration processed successfully."
    }

@router.get("/{calibration_id}/images")
async def get_calibration_images(
    calibration_id: int,
    db: Session = Depends(get_db)
):
    calibration: models.CameraCalibration = db.query(models.CameraCalibration).get(calibration_id)
    if calibration is None:
        raise HTTPException(status_code=404, detail="Camera calibration not found")
    
    calibration_path = os.path.join(CALIBRATION_DATA_PATH, str(calibration_id))
    if not os.path.exists(calibration_path):
        raise HTTPException(status_code=404, detail="No images found for this calibration")
    
    image_files = [f for f in os.listdir(calibration_path) if f.lower().endswith('.jpg')]
    
    image_urls = [
        f"/api/v1/camera-calibrations/{calibration_id}/images/{image}"
        for image in image_files
    ]
    return {"images": image_urls}

@router.get("/{calibration_id}/images/{image_name}")
async def get_calibration_image(
    calibration_id: int,
    image_name: str,
    db: Session = Depends(get_db)
):
    calibration: models.CameraCalibration = db.query(models.CameraCalibration).get(calibration_id)
    if calibration is None:
        raise HTTPException(status_code=404, detail="Camera calibration not found")
    
    image_path = os.path.join(CALIBRATION_DATA_PATH, str(calibration_id), image_name)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(image_path)