# © 2024 Wyler Zahm. All rights reserved.

import os
from typing import List


from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles

from database import get_default_session_factory
from models import LiveDetectionState

from routers import (
    camera_calibrations_router,
    vehicle_detections_router,
    speed_calibrations_router,
    live_detection_router
)

app = FastAPI(title="Vehicle Tracking System API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust this to your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
@app.get("/", tags=["index"])
async def index():
    return FileResponse(os.path.join("static", "index.html"))

# Serve static files
app.mount("/static", StaticFiles(directory="data"), name="static")

# Launch detector
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Reset the live detection state on startup
    try:
        session_factory = get_default_session_factory()
        with session_factory.get_db_with() as db:
            state = db.query(LiveDetectionState).first()
            if state:
                db.delete(state)
                db.commit()
    except Exception as e:
        print(f"Error resetting live detection state on startup: {e}")
    yield


app.router.lifespan_context = lifespan

# Endpoints
app.include_router(camera_calibrations_router, prefix="/api/v1/camera-calibrations", tags=["camera calibrations"])
app.include_router(vehicle_detections_router, prefix="/api/v1/vehicle-detections", tags=["vehicle detections"])
app.include_router(speed_calibrations_router, prefix="/api/v1/speed-calibrations", tags=["speed calibrations"])
app.include_router(live_detection_router, prefix="/api/v1/live-detection", tags=["live detection"])

# Mount static files at the root level
static_dir = os.path.abspath("data/detections_data")
app.mount("/detections", StaticFiles(directory=static_dir), name="detections")

@app.get("/detection/{calibration_date}/vehicles/{vehicle_id}/images", response_model=List[str])
async def list_vehicle_images(
    calibration_date: str,
    vehicle_id: int,
):
    # Construct the path to the images directory
    images_dir = os.path.join(static_dir, calibration_date, "vehicles", str(vehicle_id), "images")
    
    # Check if the directory exists
    if not os.path.exists(images_dir):
        raise HTTPException(status_code=404, detail="Images directory not found")
    
    # List all files in the directory
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Construct the full URLs for each image
    base_url = f"/detections/{calibration_date}/vehicles/{vehicle_id}/images"
    image_urls = [f"{base_url}/{image}" for image in image_files]
    
    return image_urls

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)