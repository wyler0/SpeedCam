import logging
import time
import multiprocessing
from typing import List
import json
import os

import numpy as np
import cv2
from sqlalchemy.orm import Session
from pydantic import BaseModel

from core.optical_flow.optical_flow_base import OpticalFlowDetector, flow_registry
from core.detection.detection_schemas import TrackedVehicleEvent, TrackedVehicle
import models
import schemas
from database import get_default_session_factory
from custom_enums import VehicleDirection, EstimationStatus

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class EstimatorConfig(BaseModel):
    optical_flow_estimator: str = 'opencv_lucaskanade'
    is_calibration: bool = False
    
    @staticmethod
    def validate_config(config: 'EstimatorConfig') -> bool:
        assert config.optical_flow_estimator in flow_registry.keys(), "Invalid detector"
        return True

class Estimator:
    def __init__(self, config: EstimatorConfig=None, speed_calibration_id: int=None, left_to_right_constant: float=None, right_to_left_constant: float=None):
        self.config: EstimatorConfig = config
        self.speed_calibration_id = speed_calibration_id
        self.left_to_right_constant = left_to_right_constant
        self.right_to_left_constant = right_to_left_constant
        self.running = False
        self.stop_signal = None
        self.optical_flow_detector: OpticalFlowDetector = None

    def start(self):
        self.running = True
        self.stop_signal = multiprocessing.Event()
        self.process = multiprocessing.Process(target=self.run_estimation_loop, args=(self.stop_signal,))
        self.process.start()

    def stop(self):
        if self.stop_signal:
            self.stop_signal.set()
            self.process.join()

    def run_estimation_loop(self, stop_signal: multiprocessing.Event):
        self.optical_flow_detector: OpticalFlowDetector = flow_registry[self.config.optical_flow_estimator]()
        while not stop_signal.is_set():
            with get_default_session_factory().get_db_with() as db:
                self.process_pending_vehicles(db)
            time.sleep(1)  # Adjust sleep time as needed
        
        # Process any remaining vehicles
        self.process_pending_vehicles(db)
        
        logger.info("Stopping estimation loop")

    def process_pending_vehicles(self, db: Session):
        pending_vehicles = db.query(models.VehicleDetection).filter(
            models.VehicleDetection.speed_calibration_id == self.speed_calibration_id,
            models.VehicleDetection.estimation_status == EstimationStatus.PENDING
        ).all()

        for vehicle in pending_vehicles:
            vehicle.estimation_status = EstimationStatus.PROCESSING
            db.commit()
            
            try:
                self.estimate_vehicle_speed(db, vehicle)
            except Exception as e:
                logger.error(f"Error estimating speed for vehicle {vehicle.id}: {str(e)}")
                vehicle.estimation_status = EstimationStatus.FAILED
                db.commit()

    def estimate_vehicle_speed(self, db: Session, vehicle_model: models.VehicleDetection):
        # Load events data
        vehicle: TrackedVehicle = self.load_events_data(vehicle_model.events_data_path)
        event_images = self.load_event_images([i.image_path for i in vehicle.events])
        logger.info(f"Loaded {len(event_images)} event images for vehicle {vehicle.vehicle_id}")
        
        # Compute pixel speed estimate
        logger.info(f"Estimating pixel speed for vehicle {vehicle.vehicle_id}...")
        pixel_speed = self.run_optical_flow(vehicle.events, event_images)
        
        # Convert to world speed if not calibrating
        logger.info(f"Estimating world speed for vehicle {vehicle.vehicle_id}...")
        world_speed = None
        if not self.config.is_calibration:
            world_speed = self.convert_pixel_speed_to_world_speed(pixel_speed, vehicle.direction)
        
        # Update database
        vehicle_model.pixel_speed_estimate = pixel_speed
        vehicle_model.real_world_speed_estimate = world_speed
        vehicle_model.estimation_status = EstimationStatus.SUCCESS
        db.commit()
        
        logger.info(f"Successfully estimated speed for vehicle {vehicle.vehicle_id}")

    def run_optical_flow(self, events: List[TrackedVehicleEvent], event_images: List[np.ndarray]) -> float:
        if len(events) < 2 or len(event_images) < 2:
            return 0.0

        # Find the global y-range for cropping
        min_y = min(event.bbox[1] for event in events)
        max_y = max(event.bbox[1] + event.bbox[3] for event in events)
        margin = 20  # pixels
        y1 = max(int(min_y) - margin, 0)
        y2 = min(int(max_y) + margin, event_images[0].shape[0])

        # Crop all images based on this y-range
        cropped_images = [img[y1:y2, :] for img in event_images]

        total_flow = 0.0
        valid_flow_count = 0

        for i in range(len(cropped_images) - 1):
            prev_img = cropped_images[i]
            curr_img = cropped_images[i+1]

            # Use the optical flow detector
            flows = self.optical_flow_detector.compute_flow([prev_img, curr_img])
            
            if flows and len(flows[0]) > 0:
                flow = flows[0]

                # Filter flows based on the current event's bounding box
                event = events[i]
                x1, y1_event, w, h = event.bbox
                x2, y2_event = x1 + w, y1_event + h
                
                # Adjust bounding box coordinates relative to the cropped image
                y1_event = max(y1_event - y1, 0)
                y2_event = min(y2_event - y1, prev_img.shape[0])

                mask = (flow[:, 0] >= x1) & (flow[:, 0] <= x2) & (flow[:, 1] >= y1_event) & (flow[:, 1] <= y2_event)
                filtered_flow = flow[mask]

                if len(filtered_flow) > 0:
                    # Compute average flow magnitude
                    flow_magnitude = np.mean(np.linalg.norm(filtered_flow[:, 2:], axis=1))
                    total_flow += flow_magnitude
                    logger.info(f"Flow magnitude: {flow_magnitude}")
                    valid_flow_count += 1

        if valid_flow_count > 0:
            return total_flow / valid_flow_count
        else:
            return 0.0

    def load_events_data(self, events_data_path: str) -> TrackedVehicle:
        return TrackedVehicle.load_from_events_data(events_data_path)
    
    def load_event_images(self, event_images_paths: List[str]) -> List[np.ndarray]:
        images = []
        for path in event_images_paths:
            images.append(cv2.imread(path))
        return images
        
    # def compute_pixel_speed(self, events_data: List[dict]) -> float:
    #     # Implement pixel speed computation using events data
    #     # This is a placeholder implementation
    #     return np.mean([event['bbox'][2] for event in events_data])  # Using width as a simple proxy for speed

    def convert_pixel_speed_to_world_speed(self, pixel_speed: float, direction: VehicleDirection) -> float:
        if self.config.is_calibration or (self.left_to_right_constant is None or self.right_to_left_constant is None):
            return None  # Return None if in calibration mode or if constants are not set
        
        if direction == VehicleDirection.LEFT_TO_RIGHT:
            world_speed = pixel_speed * self.left_to_right_constant
        elif direction == VehicleDirection.RIGHT_TO_LEFT:
            world_speed = pixel_speed * self.right_to_left_constant
        else:
            world_speed = None
        return world_speed

