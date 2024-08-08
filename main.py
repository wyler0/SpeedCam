from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from database import get_db
from models import CameraCalibration, SpeedCalibration, VehicleDetection
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()

# Pydantic models for request bodies
class CameraCalibrationCreate(BaseModel):
    camera_id: str
    calibration_date: datetime
    image_paths: list
    calibration_matrix: dict
    distortion_coefficients: dict
    rotation_matrix: dict
    translation_vector: dict

@app.post("/camera-calibrations/", response_model=dict)
def create_camera_calibration(calibration: CameraCalibrationCreate, db: Session = Depends(get_db)):
    db_calibration = CameraCalibration(**calibration.dict())
    db.add(db_calibration)
    db.commit()
    db.refresh(db_calibration)
    return {"id": db_calibration.id, "message": "Camera calibration created successfully"}

@app.get("/camera-calibrations/", response_model=list)
def read_camera_calibrations(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    calibrations = db.query(CameraCalibration).offset(skip).limit(limit).all()
    return calibrations

# Add similar endpoints for SpeedCalibration and VehicleDetection

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)