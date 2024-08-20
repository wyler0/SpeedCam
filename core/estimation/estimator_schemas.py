from pydantic import BaseModel, Field
from typing import List, Optional
from custom_enums import VehicleDirection


class TrackedVehicleEvent(BaseModel):
    frame_number: Optional[int] = None
    event_time: Optional[int] = None  # Relative to video start time
    bbox: Optional[List[float]] = None  # 0 - x, 1 - y, 2 - xWidth, 3 - yWidth

class TrackedVehicle(BaseModel):
    vehicle_id: Optional[int] = None
    start_time: Optional[int] = None  # Video time in ms
    elapsed_time: Optional[int] = None  # Vehicle Last Event Time - Start time
    speed_estimate: Optional[float] = None
    speed_error: Optional[float] = None
    direction: Optional[VehicleDirection] = None  # See directions enum
    events: List[TrackedVehicleEvent] = Field(default_factory=list)

