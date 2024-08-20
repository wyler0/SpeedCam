import os

import asyncio
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.estimation.estimator import SpeedEstimator
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

# Launch detector
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    #estimator = SpeedEstimator(None)
    #asyncio.create_task(estimator.start())
    yield
    # Shutdown
    # Add any cleanup code here if needed

app.router.lifespan_context = lifespan

# Endpoints
app.include_router(camera_calibrations_router, prefix="/api/v1/camera-calibrations", tags=["camera calibrations"])
app.include_router(vehicle_detections_router, prefix="/api/v1/vehicle-detections", tags=["vehicle detections"])
app.include_router(speed_calibrations_router, prefix="/api/v1/speed-calibrations", tags=["speed calibrations"])
app.include_router(live_detection_router, prefix="/api/v1/live-detection", tags=["live detection"])



# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
