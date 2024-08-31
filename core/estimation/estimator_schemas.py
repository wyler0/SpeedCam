from pydantic import BaseModel, Field
from typing import List, Optional
from custom_enums import VehicleDirection

class TrackedVehicleEvent(BaseModel):
    frame_number: Optional[int] = None
    event_time: Optional[int] = None  # Relative to video start time
    flows: Optional[any] = None 
    bbox: Optional[any] = None 
    avg_flow: Optional[any] = None 
    image_path: Optional[str] = None 

    class Config:
        arbitrary_types_allowed = True
        
class TrackedVehicle(BaseModel):
    vehicle_id: Optional[int] = None
    start_time: Optional[int] = None  # Video time in ms
    elapsed_time: Optional[int] = None  # Vehicle Last Event Time - Start time
    pixel_speed_estimate: Optional[float] = None
    world_speed_estimate: Optional[float] = None
    speed_error: Optional[float] = None
    direction: Optional[VehicleDirection] = None  # See directions enum
    events: List[TrackedVehicleEvent] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
