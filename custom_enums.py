# © 2024 Wyler Zahm. All rights reserved.

from enum import IntEnum

class VehicleDirection(IntEnum):
    LEFT_TO_RIGHT = 0
    RIGHT_TO_LEFT = 1
    
class EstimationStatus(IntEnum):
    PENDING = 0
    PROCESSING = 1
    SUCCESS = 2
    FAILED = 3