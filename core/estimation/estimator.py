# © 2024 Wyler Zahm. All rights reserved.

import os
import logging
from typing import Literal, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import time
import multiprocessing
import shutil

from sqlalchemy.orm import Session
import numpy as np
import cv2
from pydantic import BaseModel
from scipy.optimize import linear_sum_assignment

from core.estimation.estimator_schemas import TrackedVehicleEvent, TrackedVehicle
from core.optical_flow.optical_flow_base import OpticalFlowDetector, flow_registry
from core.video.video import VideoStream
from custom_enums import VehicleDirection
import models
import schemas
from database import SessionFactory, get_default_session_factory
from config import DETECTIONS_DATA_PATH, LATEST_DETECTION_IMAGE_PATH


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

"""
#TODO:
    - [ ] Add vector saving, rendering of vehicle vectors, and writing of key frames to dir for vehicle
    - [ ] make all logging write to console in all files and to database.
    - [ ] optimize speed estimation, currently takes about 6.6x as long as the video/ real time.
    
#WORKFLOW
    - ANY MODE:
        State is managed outside of the estimator, before it started and before it is stopped. Must manually call stop.
        
        [START] 
        - When started, all previous data clears, the latest state is read, and the (potentially invalid) speed calibration is set and the camera calibration is set.
        - A video source is read frame by frame, frames are camera calibrated, and optical flow is computed.
        - Then, the bounding boxes are detected and tracked.
        - Then, the speed is estimated based on the average of the motion flows excluding outliers (and some other filters).
        - If a vehicle is finalized, it is saved to the database, and a keyframe is posted to the image dir with some flows
        
        [STOP] 
        - If stop signal fires, we stop reading the video stream, finalize any detections and close the process.
        
    - CALIBRATION MODE: (Indicated via state.is_calibrating = True)
        - Here, the speed calibration must be created and empty, then set the camera calibration for it.
        - Then, as we make our estimations, we save all data and estimate speeds, but do not calibrate them with the constant.
    - NON-CALIBRATION MODE: (Indicated via state.is_calibrating = False)
        - Here, the speed calibration is set and the camera calibration is set.
        - Then, as we make our estimations, we save all data and estimate speeds, and calibrate them with the constant.
"""

class EstimatorConfig(BaseModel):
    #### PROGRAM ####
    mode: Literal['speed', 'calibration'] = 'speed'
    detector: str = 'opencv_farneback'
    input_video: Optional[Union[int, str]] = None
    image_write_interval_frames: int = 20 # Minimum time between detected vehilce image writes
    
    #### CORRELATION ####
    MAX_Y_DELTA: int = 350 # Maximum vertical distance between events to still be considered the same event.
    MAX_MS_DELTA: int = 800 # Maximum time between events to still be considered the same event. 
    X_DELTA_WEIGHT: float = 0.1 # Larger = more likely to be considered new vehicle (less likely to be considered same vehicle)
    Y_DELTA_WEIGHT: float = 10 # Larger = more likely to be considered new vehicle (less likely to be considered same vehicle)
    NEW_EVENT_THRESH: int = 500 # Larger = less likely to accept uncorrealted event as new vehicle
    OVERLAP_REWARD_WEIGHT: int = 2000 # Larger = less likely to be considered new vehicle.
    
    #### DETECTION ####
    BLUR_SIZE: int = 10
    MIN_BBOX_WIDTH: int = 150
    MIN_BBOX_HEIGHT: int = 10
    MAX_BBOX_WIDTH: int = 900
    MAX_BBOX_HEIGHT: int = 1e9
    MIN_BBOX_Y: int = 550
    THRESHOLD_SENSITIVITY: int = 3
    LEFT_CROP_l2r: int = 0 # Distance from LHS of frame where detection does not occur
    RIGHT_CROP_l2r: int = 0 # Distance from RHS of frame where detection does not occur
    LEFT_CROP_r2l: int = 200 # Distance from LHS of frame where detection does not occur
    RIGHT_CROP_r2l: int = 100 # Distance from RHS of frame where detection does not occur

    #### Checks ####
    @staticmethod
    def validate_config(config: 'EstimatorConfig') -> bool:
        assert config.mode in ['speed', 'calibration'], "Invalid mode"
        assert config.detector in flow_registry.keys(), "Invalid detector"
        assert config.input_video is not None, "No input video source provided"
        # Check if input video is a non numerical string or a numerical string
        # if config.input_video.isdigit() == False:
        #     assert os.path.exists(config.input_video), "Input video does not exist"
        # else:
        #     pass
        
        return True



class SpeedEstimator():
    """ Speed Estimator class for vehicle detection. """ 
    state: schemas.LiveDetectionState = None
    running: bool = False
    cam_calib: schemas.CameraCalibration = None
    spd_calib: schemas.SpeedCalibration = None
    tracking_dict: dict[int, TrackedVehicle] = None
    video: VideoStream = None
    stop_signal: multiprocessing.Event = None
    data_path: str = None
    
    ##### INITIALIZATION #####
    def __init__(self,config: EstimatorConfig = None):
        """ Setup the estimator.
        """
        self.config = config

    def init_video(self) -> VideoStream:
        src = int(self.config.input_video) if isinstance(self.config.input_video, int) or self.config.input_video.isdigit() else self.config.input_video
        video = VideoStream(src)
        
        assert video.isOpened(), "Could not open video device or file."
        
        video.start()
        
        # Set the width and height of the video capture
        # self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
        # self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
        
        # Read the first frame to ensure the stream is working
        ret, frame = video.read()
        assert ret is not None, "Failed to read from video"
        
        return video
    
    def setup(self):
        with get_default_session_factory().get_db_with() as db:
            # Load the state
            db_state = db.query(models.LiveDetectionState).first()
            self.state = schemas.LiveDetectionState.model_validate(db_state) if db_state else None

            assert self.state is not None, "Live detection state not initialized."
                    
            # Set calibrations
            self.setup_speed_calibration(db)
            self.setup_camera_dewarp(db)
            
        # Create data paths
        self.data_path = f"{DETECTIONS_DATA_PATH}/{self.state.speed_calibration_id}"
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            
        self.images_path = f"{self.data_path}/images"
        if not os.path.exists(self.images_path):
            os.makedirs(self.images_path)
        
        # Create detector
        self.detector: OpticalFlowDetector = flow_registry[self.config.detector]()

        # Get target stream
        self.config.input_video = self.state.camera_source if self.state.camera_source is not None else self.state.video_path
        assert EstimatorConfig.validate_config(self.config), "Invalid config."
        self.video = self.init_video()
        
        # Get initial images
        img1, _ = self.video.read()
        img2, _ = self.video.read()
        self.target_imgs = [img1, img2]
        assert self.target_imgs[0] is not None, "Error loading target video"

        # Setup tracking dict
        self.tracking_dict = {}

    ##### MAIN LOOP ######
    def run_loop(self, stop_signal: multiprocessing.Event):
        self.setup()
        
        with get_default_session_factory().get_db_with() as db:
            # Write the latest frame to LATEST_DETECTION_IMAGE_PATH
            cv2.imwrite(LATEST_DETECTION_IMAGE_PATH, self.target_imgs[0])
            # Signal the frontend
            self.state = db.query(models.LiveDetectionState).first()
            self.state.has_new_image = True
            db.commit()
                        
            start_frame = self.video.frames_read
            last_image_write_time = start_frame
            
            try:
                while self.running and self.get_next_image() and not stop_signal.is_set():
                    # Apply dewarp
                    self.target_imgs[1] = self.apply_camera_dewarp(self.target_imgs[1])

                    # Get optical flow flows
                    self.flows = self.detector.compute_flow(self.target_imgs)
                    self.flow_groups = self.detector.group_flows(self.flows)
                    
                    if len(self.flows[0]) > 0:
                        # Extract events from bboxes
                        events = self.get_filter_new_events()
                        if len(events) > 0:
                            events = self.correlate_events_and_vehicles(events)
                            img_written = False
                            # Write the all detections latest detection frame for the frontend to render
                            if self.video.frames_read - last_image_write_time > self.config.image_write_interval_frames:
                                paths = self.detector.visualize_flow_groups([[event.flows for _, event in events]], self.target_imgs, output_path=f"{self.data_path}/images/{time.time()}")
                                shutil.move(paths[0], LATEST_DETECTION_IMAGE_PATH)
                                last_image_write_time = self.video.frames_read
                                img_written = True
                                
                                # Signal the frontend
                                self.state = db.query(models.LiveDetectionState).first()
                                self.state.has_new_image = True
                                db.commit()
                            
                            # Add event and or new vehicles
                            for event_id, event in events:
                                # Visualize flow groups for the event
                                if img_written:
                                    # Flow groups viz expects [Frame, Group, Flow Data]
                                    paths = self.detector.visualize_flow_groups([[event.flows]], self.target_imgs, output_path=f"{self.data_path}/images/{time.time()}")
                                    event.image_path = paths[0]
                                        
                                self.add_vehicle_event(event, vehicle = self.tracking_dict.get(event_id, None))
                            
                    # Check for finished events
                    self.check_finished_and_stale_events(db)
            
                # Check for finished events
                self.running = False
                self.check_finished_and_stale_events(db, final=True)
                
                # Clean up
                shutil.rmtree(self.images_path)
                
                # Update state
                self.state: schemas.LiveDetectionState = db.query(models.LiveDetectionState).first()
                self.state.running = False
                self.state.processing_video = False
                db.commit()
            except Exception as e:
                logger.error(f"Error in run_loop: {e}. Disabling estimator.")
                self.state.running = False
                self.state.processing_video = False
                self.state.error = str(e)
                db.commit()
    
    def start(self) -> multiprocessing.Event:
        self.running = True
        self.stop_signal = multiprocessing.Event()
        self.process = multiprocessing.Process(target=self.run_loop, args=(self.stop_signal,))
        self.process.start()
        return self.stop_signal
    
    def await_completion(self):
        if hasattr(self, 'process'):
            self.process.join()
    
    def stop(self):
        if hasattr(self, 'stop_signal'):
            self.stop_signal.set()
            self.await_completion()
    
    #### PROCESSING UTILITIES ####
    def get_filter_new_events(self) -> List[TrackedVehicleEvent]:
        """ Extract events from flow groups via filtering according to configured parameters. 
        
        config:
            MIN_BBOX_WIDTH (int): Minimum width of bbox to be considered an event.
            MAX_BBOX_WIDTH (int): Maximum width of bbox to be considered an event.
            MIN_BBOX_HEIGHT (int): Minimum height of bbox to be considered an event.
            MAX_BBOX_HEIGHT (int): Maximum height of bbox to be considered an event.
            MIN_BBOX_Y (int): Minimum Y coordinate of the bottom of the bbox.
        Returns: [event, event, ...]
        """
        events: List[TrackedVehicleEvent] = []
        for frame_number, frame_flow_groups in enumerate(self.flow_groups):
            for flow_group in frame_flow_groups:
                # Compute bounding box for the flow group
                min_x, min_y = np.min(flow_group[:, :2], axis=0)
                max_x, max_y = np.max(flow_group[:, :2], axis=0)
                width = max_x - min_x
                height = max_y - min_y
                
                # Apply filters
                if not (self.config.MIN_BBOX_WIDTH < width < self.config.MAX_BBOX_WIDTH):
                    continue
                if not (self.config.MIN_BBOX_HEIGHT < height < self.config.MAX_BBOX_HEIGHT):
                    continue
                if not (max_y > self.config.MIN_BBOX_Y):
                    continue
                
                # Create event
                new_event = TrackedVehicleEvent(
                    frame_number=self.video.frames_read + frame_number,
                    event_time=int(self.video.get_time_ms()),
                    flows=flow_group,
                    bbox=[min_x, min_y, width, height],
                    avg_flow=np.mean(flow_group[:, 2:], axis=0).tolist()
                )
                events.append(new_event)
        
        return events
    
    def correlate_events_and_vehicles(self, events: List[TrackedVehicleEvent]) -> List[Tuple[Optional[int], TrackedVehicleEvent]]:
        """
        Correlates events to ongoing or new tracked vehicles.
        
        Returns: 
            List of tuples (vehicle_id, event) where vehicle_id is None for new vehicles
        """
        candidate_vehicle_ids = list(self.tracking_dict.keys())
        candidate_vehicles = list(self.tracking_dict.values())
        n_events = len(events)
        n_candidates = len(candidate_vehicles)
        
        # Initialize correlation matrix
        correlation_mat = np.full((n_events, max(n_events, n_candidates)), 1e9)
        
        if n_candidates > 0:
            # Vectorized calculations for all events and candidates
            event_centers = np.array([(e.bbox[0] + e.bbox[2]/2, e.bbox[1] + e.bbox[3]/2) for e in events])
            candidate_centers = np.array([(v.events[-1].bbox[0] + v.events[-1].bbox[2]/2, 
                                           v.events[-1].bbox[1] + v.events[-1].bbox[3]/2) for v in candidate_vehicles])
            
            # Calculate deltas
            y_deltas = np.abs(event_centers[:, 1, np.newaxis] - candidate_centers[:, 1])
            x_deltas = event_centers[:, 0, np.newaxis] - candidate_centers[:, 0]
            time_deltas = np.array([e.event_time for e in events])[:, np.newaxis] - \
                          np.array([v.events[-1].event_time for v in candidate_vehicles])
            
            # Apply filters
            valid_matches = (y_deltas < self.config.MAX_Y_DELTA) & (time_deltas < self.config.MAX_MS_DELTA)
            
            # Calculate overlap rewards
            overlap_rewards = self._calculate_overlap_rewards(events, candidate_vehicles)
            
            # Calculate correlations
            correlations = x_deltas * self.config.X_DELTA_WEIGHT + y_deltas * self.config.Y_DELTA_WEIGHT
            correlations -= overlap_rewards * self.config.OVERLAP_REWARD_WEIGHT
            
            # Apply valid matches mask
            correlations[~valid_matches] = 1e9
            
            # Update correlation matrix
            correlation_mat[:, :n_candidates] = correlations
        
        # Use the Hungarian algorithm to find the optimal assignment
        event_indices, candidate_indices = linear_sum_assignment(correlation_mat)
        
        results = []
        for event_idx, candidate_idx in zip(event_indices, candidate_indices):
            if candidate_idx >= n_candidates or correlation_mat[event_idx, candidate_idx] >= 1e9:
                if np.min(correlation_mat[event_idx, :]) > self.config.NEW_EVENT_THRESH:
                    results.append((None, events[event_idx]))
            else:
                results.append((candidate_vehicle_ids[candidate_idx], events[event_idx]))
        
        return results

    def _calculate_overlap_rewards(self, events: List[TrackedVehicleEvent], candidates: List[TrackedVehicle]) -> np.ndarray:
        """Calculate overlap rewards between new events and existing tracked vehicles."""
        n_events = len(events)
        n_candidates = len(candidates)
        overlap_rewards = np.zeros((n_events, n_candidates))
        
        event_bboxes = np.array([e.bbox for e in events])
        candidate_bboxes = np.array([c.events[-1].bbox for c in candidates])
        
        # Calculate areas
        event_areas = event_bboxes[:, 2] * event_bboxes[:, 3]
        candidate_areas = candidate_bboxes[:, 2] * candidate_bboxes[:, 3]
        
        # Calculate intersections
        x_left = np.maximum(event_bboxes[:, 0, np.newaxis], candidate_bboxes[:, 0])
        y_top = np.maximum(event_bboxes[:, 1, np.newaxis], candidate_bboxes[:, 1])
        x_right = np.minimum(event_bboxes[:, 0, np.newaxis] + event_bboxes[:, 2, np.newaxis], 
                             candidate_bboxes[:, 0] + candidate_bboxes[:, 2])
        y_bottom = np.minimum(event_bboxes[:, 1, np.newaxis] + event_bboxes[:, 3, np.newaxis], 
                              candidate_bboxes[:, 1] + candidate_bboxes[:, 3])
        
        intersection_areas = np.maximum(0, x_right - x_left) * np.maximum(0, y_bottom - y_top)
        
        # Calculate IoU
        union_areas = event_areas[:, np.newaxis] + candidate_areas - intersection_areas
        overlap_rewards = intersection_areas / union_areas
        
        return overlap_rewards
    
    def check_finished_and_stale_events(self, db: Session, final=False,
                                        MAX_MS_SINCE_LAST_EVENT=1000, MAX_VEHILCE_TOTAL_ELPASED_MS=3000):
        """ Finds ongoing tracked vehicles that have exited the frame. Finishes all open events if final is true. 
        
        config:
            MAX_MS_SINCE_LAST_EVENT: Maximum time since the vehicle was last tracked / appeared in a frame for event to be ongoing
            MAX_VEHILCE_TOTAL_ELPASED_MS: Maximum time a vehicle can be considered still ongoing from its first detection.
        """
        # If data complete, move all tracked vehicles to finalized
        finished_vehicles = {}
        if final: finished_vehicles = self.tracking_dict
        # If data ongoing, compute completed tracked vehicles and move them to finalized
        else:
            for vehicle_id,vehicle in self.tracking_dict.items():
                finished = False
                # Get overall event time
                total_elapsed_time = self.video.get_time_ms() - vehicle.start_time
                # Get last detected time delta
                last_event_delta = self.video.get_time_ms() - vehicle.events[-1].event_time
                # Time cutoff comparison
                if total_elapsed_time > MAX_VEHILCE_TOTAL_ELPASED_MS: finished = True
                if last_event_delta > MAX_MS_SINCE_LAST_EVENT: finished = True
                # Finish
                if finished: finished_vehicles[vehicle_id] = vehicle
        
        # Finish the vehicles
        for vehicle in finished_vehicles.values(): 
            self.finish_tracked_vehicle(db, vehicle) 
        # Remove finished vehicles
        self.tracking_dict = {k: v for k, v in self.tracking_dict.items() if k not in finished_vehicles}

    def apply_camera_dewarp(self, img):
        h, w = img.shape[:2]
        
        #TODO Might be able to store this step instead of computing every time. I think it scales in case the original image is different w/h from the original matrix.
        newcameramtx, roi= cv2.getOptimalNewCameraMatrix(self.cam_calib.calibration_matrix,self.cam_calib.distortion_coefficients,(w,h),1,(w,h))

        # Apply undistortion and Crop to ROI
        dst = cv2.undistort(img, self.cam_calib.calibration_matrix, self.cam_calib.distortion_coefficients, None, newcameramtx)
        #x, y, w, h = roi
        #dst = dst[y:y+h, x:x+w]
        return dst
    
    def get_pixel_speed_estimate(self, vehicle: TrackedVehicle):
        # Average speed of all events, avg_flow is dx dy want to include both components in case of not perflect horizontal road perspective.
        #TODO: Optimize this? Pixel speed can be computed from the flow data in the grouping operation so no need to recompute just average.
        return sum([np.linalg.norm(event.avg_flow) for event in vehicle.events]) / len(vehicle.events)
    
    def convert_pixel_speed_to_world_speed(self, pixel_speed: float, direction: VehicleDirection): #TODO
        if direction == VehicleDirection.LEFT_TO_RIGHT:
            return pixel_speed * self.spd_calib.left_to_right_constant
        elif direction == VehicleDirection.RIGHT_TO_LEFT:
            return pixel_speed * self.spd_calib.right_to_left_constant
        else:
            raise Exception("Invalid direction")
    
    def get_direction_estimate(self, events: List[TrackedVehicleEvent]):
        #TODO Optimize this? Direction can be returned from the grouping operation, as its already computed there...
        # average direction of all events
        dx_avg = sum([event.avg_flow[0] for event in events]) / len(events)
        return VehicleDirection.RIGHT_TO_LEFT if dx_avg < 0 else VehicleDirection.LEFT_TO_RIGHT
    
    ##### TRACKING DATA UTILITIES ######
    def add_vehicle_event(self, event: TrackedVehicleEvent, vehicle: TrackedVehicle = None):
        # Add new vehicle if id is None
        if vehicle is None or vehicle.vehicle_id is None:
            vehicle_id = self.add_tracked_vehicle()
            logger.debug(f"Added new vehicle {vehicle_id}")
        else:
            vehicle_id = vehicle.vehicle_id
            logger.debug(f"Adding event to existing vehicle {vehicle_id}")
            
        # Add new event to vehicle
        self.tracking_dict[vehicle_id].events.append(event)
        
        # If events > 1, update direction computation.
        #TODO: Optimize this? We don't need to recompute direction every time.
        if len(self.tracking_dict[vehicle_id].events) > 1:
            self.tracking_dict[vehicle_id].direction = self.get_direction_estimate(self.tracking_dict[vehicle_id].events)

    def finish_tracked_vehicle(self, db: Session, vehicle: TrackedVehicle, MIN_EVENTS_FOR_VALID=20) -> bool: #TODO (Saving of flows)
        if len(vehicle.events) < MIN_EVENTS_FOR_VALID: 
            logger.warning(f"Vehicle Failed: not enough events for vehicle {vehicle.vehicle_id}")
            self.clean_image_paths(vehicle)
            return False
        
        vehicle.direction = self.get_direction_estimate(vehicle.events)
        if vehicle.direction == None: 
            logger.warning(f"Vehicle Failed: unable to determine direction for vehicle {vehicle.vehicle_id}")
            self.clean_image_paths(vehicle)
            return False
        
        vehicle.elapsed_time = vehicle.events[-1].event_time - vehicle.start_time
        vehicle.pixel_speed_estimate = self.get_pixel_speed_estimate(vehicle)
        
        if not self.state.is_calibrating: 
            if self.spd_calib.valid:
                vehicle.world_speed_estimate = self.convert_pixel_speed_to_world_speed(vehicle.pixel_speed_estimate, vehicle.direction)
        
        # Write to database
        new_tracking_data = models.VehicleDetection(
            detection_date = datetime.now() - timedelta(seconds=2),
            thumbnail_path = None,
            direction = vehicle.direction,
            pixel_speed_estimate = vehicle.pixel_speed_estimate,
            real_world_speed_estimate = vehicle.world_speed_estimate,
            real_world_speed = None,
            optical_flow_path = None,
            speed_calibration_id = self.state.speed_calibration_id,
            error = None,
        )  # Assuming vehicle has the necessary attributes
        db.add(new_tracking_data)
        db.commit()
        
        # Get the UID of the vehicle to write the flow data
        optical_flow_path = self.write_flow_data(new_tracking_data.id, vehicle)
        
        # Update the optical_flow_path and thumbnail path in the database
        new_tracking_data.optical_flow_path = optical_flow_path
        new_tracking_data.thumbnail_path = f"{self.data_path}/vehicles/{new_tracking_data.id}/images"
        
        db.commit()
        
        logger.info(f"Finished vehicle {vehicle.vehicle_id}")
        
        return True
    
    def clean_image_paths(self, vehicle: TrackedVehicle):
        # Remove images for invalid detections
        try:
            for path in [event.image_path for event in vehicle.events]:
                if path and os.path.exists(path):
                    os.remove(path)
        except AttributeError as e:
            # Not sure why but sometimes vehicle is a dict not a TrackedVehicle object and we get an AttributeError. 
            logger.error(f"Error cleaning image paths: {e}")
    
    def add_tracked_vehicle(self):
        # Get new vehicle ID
        vehicle_id = next((i for i in range(len(self.tracking_dict) + 1) if i not in self.tracking_dict), len(self.tracking_dict))
        
        # Add new vehicle to tracking dict
        self.tracking_dict[vehicle_id] = TrackedVehicle(
            vehicle_id=vehicle_id,
            start_time=int(self.video.get_time_ms())
        )
        return vehicle_id

    def get_next_image(self, num_to_read=1):
        """ Gets next target and label image. """
        self.target_imgs[0] = self.target_imgs[1]
        for i in range(num_to_read): 
            self.target_imgs[1],_ = self.video.read()
        
        return not self.target_imgs[1] is None
    
    def write_flow_data(self, vhl_db_id: int, vehicle: TrackedVehicle): #TODO (Saving of flows)
        vehicle_path = f"{self.data_path}/vehicles/{vhl_db_id}"
        if not os.path.exists(vehicle_path):
            os.makedirs(vehicle_path)
        
        # Write flows
        flows_path = f"{vehicle_path}/flows/"
        if not os.path.exists(flows_path):
            os.makedirs(flows_path)
        
        all_flows = [event.flows for event in vehicle.events]
        np.savez(flows_path + "stacked_flows.npz", *all_flows)
        
        # Move images
        img_path = f"{vehicle_path}/images/"
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        
        for event in vehicle.events:
            if event.image_path:
                shutil.move(event.image_path, img_path + os.path.basename(event.image_path))
        
        return flows_path

    
    ##### SETUP ######
    def setup_speed_calibration(self, db: Session):
        if self.state is None: 
            raise Exception("Live detection state not initialized.")
        
        db_speed_calibration = db.query(models.SpeedCalibration).get(self.state.speed_calibration_id)
        self.spd_calib = schemas.SpeedCalibration.model_validate(db_speed_calibration) if db_speed_calibration else None
    
        if self.spd_calib is None:
            raise Exception(f"Speed calibration not found for id {self.state.speed_calibration_id}.")

        assert self.spd_calib.camera_calibration_id is not None, "Speed calibration must be set."
        
        if not self.state.is_calibrating:
            assert self.spd_calib.valid, "Speed calibration is not valid and state is not in speed calibration mode."
            assert self.spd_calib.left_to_right_constant is not None, "Speed calibration constants (left) must be set."
            assert self.spd_calib.right_to_left_constant is not None, "Speed calibration constants (right) must be set."
        
    def setup_camera_dewarp(self, db: Session):
        if self.spd_calib is None:
            raise Exception("Speed calibration not initialized.")
        
        db_camera_calibration = db.query(models.CameraCalibration).get(self.spd_calib.camera_calibration_id)
        self.cam_calib = schemas.CameraCalibration.model_validate(db_camera_calibration) if db_camera_calibration else None
        
        if self.cam_calib is None:
            raise Exception(f"Camera calibration not found for speed calibration id {self.spd_calib.camera_calibration_id}.")
        
        assert self.cam_calib.valid, "Camera calibration is not valid."
        assert self.cam_calib.calibration_matrix is not None, "Camera calibration matrix not found. "
        assert self.cam_calib.distortion_coefficients is not None, "Camera distortion coefficients not found."
        assert self.cam_calib.rotation_matrix is not None, "Camera rotation matrix not found."
        assert self.cam_calib.translation_vector is not None, "Camera translation vector not found."
        
        self.cam_calib.distortion_coefficients = np.asarray(self.cam_calib.distortion_coefficients)
        self.cam_calib.calibration_matrix = np.asarray(self.cam_calib.calibration_matrix)    

