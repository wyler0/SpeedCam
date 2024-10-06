# © 2024 Wyler Zahm. All rights reserved.

import os
from typing import List, Optional

from pydantic import BaseModel, Field

from custom_enums import VehicleDirection

class TrackedVehicleEvent(BaseModel):
    frame_number: Optional[int] = None
    event_time: Optional[float] = None  # Relative to video start time
    bbox: Optional[List[float]] = None  # Changed to List[float] for JSON serialization
    image_path: Optional[str] = None 

    class Config:
        json_encoders = {
            float: lambda v: round(v, 6) if v is not None else None  # Ensure float precision
        }
        
class TrackedVehicle(BaseModel):
    vehicle_id: Optional[int] = None
    start_time: Optional[float] = None  # Video time in sec
    elapsed_time: Optional[float] = None  # Vehicle Last Event Time - Start time
    pixel_speed_estimate: Optional[float] = None
    world_speed_estimate: Optional[float] = None
    speed_error: Optional[float] = None
    direction: Optional[VehicleDirection] = None  # See directions enum
    events: List[TrackedVehicleEvent] = Field(default_factory=list)

    @classmethod
    def load_from_events_data(cls, events_data_path: str):
        if not os.path.exists(events_data_path):
            raise FileNotFoundError(f"Events data file not found: {events_data_path}")
        
        with open(events_data_path, 'r') as f:
            json_data = f.read()
        
        return cls.model_validate_json(json_data)
    class Config:
        json_encoders = {
            float: lambda v: round(v, 6) if v is not None else None  # Ensure float precision
        }
        arbitrary_types_allowed = True
