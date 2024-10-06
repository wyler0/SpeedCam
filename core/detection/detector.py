# © 2024 Wyler Zahm. All rights reserved.

from typing import Literal, Optional, List, Tuple, Union, Dict
from datetime import datetime, timedelta
import os
import logging
import time
import multiprocessing
import shutil
import json

from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from pydantic import BaseModel
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
import torch


from core.detection.sort import Sort
from core.detection.detection_schemas import TrackedVehicleEvent, TrackedVehicle
from core.detection.estimator import EstimatorConfig, Estimator
from core.video.video import VideoStream
from custom_enums import VehicleDirection, EstimationStatus

import models
import schemas
from database import SessionFactory, get_default_session_factory
from config import DETECTIONS_DATA_PATH, LATEST_DETECTION_IMAGE_PATH, TEMP_DATA_PATH


logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

"""
#TODO:
    - [ ] Update installation instructions and run instructions
    - [ ] Fix speed limit API issues
    - [ ] Fix frontend latest detection image to render the crops correctly (or don't render them at all?)
    - [ ] Add thumbnails to the table in the frotnend detections view
     
#WORKFLOW
    - ANY MODE:
        State is managed outside of the detector, before it started and before it is stopped. Must manually call stop.
        
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
        
        
    - Data Created:
        - frontend visualization image
        - events and images saved in DETECTIONS_DATA_PATH/{speed_calibration_id}/{vehicle_id}/... events.json and images/.
"""

class DetectorConfig(BaseModel):
    #### PROGRAM ####
    mode: Literal['speed', 'calibration'] = 'speed'
    estimator_config: EstimatorConfig = EstimatorConfig(optical_flow_estimator='opencv_lucaskanade')
    input_video: Optional[Union[int, str]] = None
    image_write_interval_frames: int = 1 # Minimum time between detected vehilce image writes
    yolo_file_path: str = 'core/models/yolov8n.pt'
    
    #### DETECTION ####
    LEFT_CROP_l2r: int = 0#375 # Distance from LHS of frame where detection does not occur, as percentage of frame width
    RIGHT_CROP_l2r: int = 100 #1450 # Distance from LHS of frame where detection does not occur, as percentage of frame width
    LEFT_CROP_r2l: int = 0 # Distance from LHS of frame where detection does not occur, as percentage of frame width
    RIGHT_CROP_r2l: int = 100 # Distance from LHS of frame where detection does not occur, as percentage of frame width

    #### Checks ####
    @staticmethod
    def validate_config(config: 'DetectorConfig') -> bool:
        assert config.mode in ['speed', 'calibration'], "Invalid mode"
        assert config.input_video is not None, "No input video source provided"
        assert config.estimator_config.validate_config(config.estimator_config), "Invalid estimator config"
        
        return True


class SpeedDetector():
    """ Speed Detector class for vehicle detection using Ultralytics YOLO. """
    state: schemas.LiveDetectionState = None
    cam_calib: schemas.CameraCalibration = None
    spd_calib: schemas.SpeedCalibration = None
    tracking_dict: dict[int, TrackedVehicle] = None
    video: VideoStream = None
    stop_signal: multiprocessing.Event = None
    data_path: str = None
    model: YOLO = None
    tracker: Sort = None
    target_img: np.ndarray = None
    id_mapping: Dict[int, int] = {}
    
    ##### INITIALIZATION #####
    def __init__(self, config: DetectorConfig = None):
        self.config = config
        self.model: YOLO = None
        self.tracker: Sort = None
        self.estimator: Estimator = None

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
            
            # Update estimator config
            self.config.estimator_config.is_calibration = self.state.is_calibrating

            
        # Create data paths
        self.detections_data_path = f"{DETECTIONS_DATA_PATH}/{self.state.speed_calibration_id}"
        if not os.path.exists(self.detections_data_path):
            os.makedirs(self.detections_data_path)
            
        self.temp_images_path = f"{TEMP_DATA_PATH}/images"
        if not os.path.exists(self.temp_images_path):
            os.makedirs(self.temp_images_path)

        # Get target stream
        self.config.input_video = self.state.camera_source if self.state.camera_source is not None else self.state.video_path
        assert DetectorConfig.validate_config(self.config), "Invalid config."
        self.video = self.init_video()
        
        # Get initial images
        if not self.get_next_image():
            raise Exception("Error loading target video")
        assert self.target_img is not None, "Error loading target video"
        
        # Setup dewarp and crop
        h, w = self.target_img.shape[:2]
        
        self.config.LEFT_CROP_l2r = self.spd_calib.left_crop_l2r/100*w if self.spd_calib.left_crop_l2r is not None else self.config.LEFT_CROP_l2r/100*w
        self.config.RIGHT_CROP_l2r = self.spd_calib.right_crop_l2r/100*w if self.spd_calib.right_crop_l2r is not None else self.config.RIGHT_CROP_l2r/100*w
        self.config.LEFT_CROP_r2l = self.spd_calib.left_crop_r2l/100*w if self.spd_calib.left_crop_r2l is not None else self.config.LEFT_CROP_r2l/100*w
        self.config.RIGHT_CROP_r2l = self.spd_calib.right_crop_r2l/100*w if self.spd_calib.right_crop_r2l is not None else self.config.RIGHT_CROP_r2l/100*w

        newcameramtx, roi= cv2.getOptimalNewCameraMatrix(self.cam_calib.calibration_matrix,self.cam_calib.distortion_coefficients,(w,h),1,(w,h))
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.cam_calib.calibration_matrix, self.cam_calib.distortion_coefficients, None, newcameramtx, (w,h), cv2.CV_32FC1)

        self.target_img = self.apply_camera_dewarp(self.target_img)
        
        # Setup tracking dict
        self.tracking_dict = {}

        # Initialize YOLO model and tracker
        self.model = YOLO(self.config.yolo_file_path).to('mps')
        self.tracker = Sort(max_age=100, min_hits=3, iou_threshold=0.3)
        
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

    ##### MAIN LOOP ######
        
    def start(self, profile=False) -> multiprocessing.Event:
        self.stop_signal = multiprocessing.Event()
        self.process = multiprocessing.Process(target=self.run_loop if profile==False else self.run_loop_profile, args=(self.stop_signal,))
        self.process.start()
        
        return self.stop_signal
    
    def await_completion(self):
        if hasattr(self, 'process'):
            self.process.join()
    
    def stop(self):
        if hasattr(self, 'stop_signal'):
            self.stop_signal.set()
            self.await_completion()

    def handle_exception(self, e, db):
        import traceback
        traceback.print_exc()
        logger.error(f"Error in run_loop: {e}. Disabling detector.")
        self.state.running = False
        self.state.processing_video = False
        self.state.error = str(e)
        db.commit()

    def run_loop_profile(self, stop_signal: multiprocessing.Event):
        import cProfile
        cProfile.runctx("self.run_loop(stop_signal)", globals(), locals(), filename="profile.prof")
    
    def run_loop(self, stop_signal: multiprocessing.Event):
        self.setup()
        
        # Start estimator
        left_to_right_constant = None
        right_to_left_constant = None
        if not self.state.is_calibrating:
            left_to_right_constant = self.spd_calib.left_to_right_constant
            right_to_left_constant = self.spd_calib.right_to_left_constant

        self.estimator = Estimator(
            config=self.config.estimator_config,
            speed_calibration_id=self.spd_calib.id,
            left_to_right_constant=left_to_right_constant,
            right_to_left_constant=right_to_left_constant
        )
        self.estimator_process = multiprocessing.Process(target=self.estimator.run_estimation_loop, args=(self.stop_signal,))
        self.estimator_process.start()
        
        with get_default_session_factory().get_db_with() as db:
            # Write the latest frame to LATEST_DETECTION_IMAGE_PATH
            start_time = time.time()
            
            cv2.imwrite(LATEST_DETECTION_IMAGE_PATH, self.target_img)
            self.push_image_to_frontend(LATEST_DETECTION_IMAGE_PATH, db)
            
            results, detections = None, None
            
            last_read_time_wall = time.time()
            
            try:
                while self.get_next_image() and not stop_signal.is_set():
                    # Clear detections
                    detections = []
                    
                    # Apply dewarping
                    self.target_img = self.apply_camera_dewarp(self.target_img)
                    
                    time_a = time.time()

                    # Run YOLO detection
                    results: Results = self.model(self.target_img, task="detect", conf=0.7, classes=[2, 3], iou=0.5,verbose=False)[0]  # Detect cars and trucks
                    
                    time_b = time.time()
                
                    if len(results) > 0:
                        # Visualize events, push to frontend
                        img_path = f"{self.temp_images_path}/{time.time()}_visualized.jpg"
                        results.plot(
                            conf=False, line_width=None, font_size=None, font=None, labels=False,
                            boxes=True, masks=False, probs=False, show=False, save=True, filename=img_path,
                        )
                        self.push_image_to_frontend(img_path, db)

                        # Process detections
                        detections = self.process_detections(results)
                        

                    # Update tracker
                    tracked_objects = self.tracker.update(detections if detections is not None else [])

                    # Process tracked objects
                    if detections is not None:
                        self.process_tracked_objects(tracked_objects)

                    # Check for finished events
                    self.check_finished_and_stale_events(db)
                    
                    # Debug time deltas
                    new_read_time_wall = time.time()
                    logger.debug(f"Overall: {new_read_time_wall - last_read_time_wall: .4f}, YOLO: {time_b - time_a: .4f}, FPS: {1 / (new_read_time_wall - last_read_time_wall): .4f}")
                    last_read_time_wall = new_read_time_wall

                # Check for finished events
                self.check_finished_and_stale_events(db, final=True)
                end_time = time.time()
                logger.info(f"Total time: {end_time - start_time}")
                
                # Send stop signal to estimator
                if not self.stop_signal.is_set():
                    self.stop_signal.set()
                    self.estimator_process.join()
                    
                # Update state
                self.state: schemas.LiveDetectionState = db.query(models.LiveDetectionState).first()
                self.state.processing_video = False
                self.state.running = False
                db.commit()

            except Exception as e:
                self.handle_exception(e, db)

    #### PROCESSING UTILITIES ####

    def apply_camera_dewarp(self, img):
        # Apply undistortion and Crop to ROI
        dst = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
        return dst
    
    def get_direction_estimate(self, events: List[TrackedVehicleEvent]):
        if len(events) < 2:
            return None
        
        # Calculate dx for each pair of consecutive events
        dxes = [events[i+1].bbox[0] - events[i].bbox[0] for i in range(len(events)-1)]
        
        # Calculate the average dx
        avg_dx = sum(dxes) / len(dxes)
        
        # Determine direction based on average dx
        if avg_dx > 0:
            return VehicleDirection.LEFT_TO_RIGHT
        elif avg_dx < 0:
            return VehicleDirection.RIGHT_TO_LEFT
        else:
            # In the rare case that avg_dx is exactly 0, we can't determine the direction
            return None
    
    """
    def get_pixel_speed_estimate(self, vehicle: TrackedVehicle):
        # Average speed of all events, avg_flow is dx dy want to include both components in case of not perflect horizontal road perspective.
        #TODO: Optimize this? Pixel speed can be computed from the flow data in the grouping operation so no need to recompute just average.
        return np.abs(sum(event.avg_flow for event in vehicle.events) / len(vehicle.events))
    """
    
    """
    def convert_pixel_speed_to_world_speed(self, pixel_speed: float, direction: VehicleDirection): #TODO
        if direction == VehicleDirection.LEFT_TO_RIGHT:
            return pixel_speed * self.spd_calib.left_to_right_constant
        elif direction == VehicleDirection.RIGHT_TO_LEFT:
            return pixel_speed * self.spd_calib.right_to_left_constant
        else:
            raise Exception("Invalid direction")
    """

    ##### TRACKING DATA UTILITIES ######
    
    def process_detections(self, result: Results):
        boxes: Boxes = result.boxes
        dets = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            dets.append([x1.item(), y1.item(), x2.item(), y2.item(), conf.item()])
        
        dets = torch.tensor(dets).cpu().numpy()
        
        return dets

    def process_tracked_objects(self, tracked_objects):
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            
            # Create a TrackedVehicleEvent
            event = TrackedVehicleEvent(
                frame_number=self.video.frames_read,
                event_time=self.video.get_time_sec(),
                bbox=[x1, y1, x2 - x1, y2 - y1],
                image_path=None  # We'll set this after saving the image
            )
            
            # Check if we have a mapping for this tracker ID
            if obj_id not in self.id_mapping:
                # If not, create a new database entry and get its ID
                with get_default_session_factory().get_db_with() as db:
                    new_vehicle = models.VehicleDetection(
                        detection_date=func.now(),
                        speed_calibration_id=self.state.speed_calibration_id,
                        estimation_status=EstimationStatus.FAILED
                    )
                    db.add(new_vehicle)
                    db.flush()  # This will assign an ID to new_vehicle
                    db.commit()
                    self.id_mapping[obj_id] = new_vehicle.id

            db_id = self.id_mapping[obj_id]
            
            # Check if vehicle with that id exists in tracking dict
            if db_id not in self.tracking_dict:
                self.tracking_dict[db_id] = TrackedVehicle(
                    events=[], 
                    vehicle_id=db_id,
                    start_time=self.video.get_time_sec(),
                )
                
                # Create vehicle image directory if it doesn't exist
                os.makedirs(f"{self.detections_data_path}/{db_id}/bboxes", exist_ok=True)
                os.makedirs(f"{self.detections_data_path}/{db_id}/images", exist_ok=True)
            vehicle = self.tracking_dict[db_id]
            
            # Save the raw image
            image_filename = f"{self.video.frames_read}_{self.video.get_time_sec():.3f}.jpg"
            image_path = os.path.join(f"{self.detections_data_path}/{db_id}/images", image_filename)
            cv2.imwrite(image_path, self.target_img)
            
            # Save bbox image
            final_img = self.target_img.copy()
            cv2.rectangle(final_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            image_filename = f"{self.video.frames_read}_{self.video.get_time_sec():.3f}.jpg"
            image_path = os.path.join(f"{self.detections_data_path}/{db_id}/bboxes", image_filename)
            cv2.imwrite(image_path, final_img)
            
            # Update the event's image path
            event.image_path = image_path
            
            # Add the event to the vehicle
            vehicle.events.append(event)
            
            if not vehicle.direction and len(vehicle.events) > 1:
                vehicle.direction = self.get_direction_estimate(vehicle.events)

    def check_finished_and_stale_events(self, db: Session, final=False, MAX_MS_SINCE_LAST_EVENT=1000, MAX_VEHILCE_TOTAL_ELPASED_MS=4000):
        """ Finds ongoing tracked vehicles that have exited the frame. Finishes all open events if final is true. 
        args:
            read_time_delta: Time delta since last read in seconds. Used to scale the time deltas for the comparison based on effective frame rate.
        config:
            MAX_MS_SINCE_LAST_EVENT: Maximum time since the vehicle was last tracked / appeared in a frame for event to be ongoing
            MAX_VEHILCE_TOTAL_ELPASED_MS: Maximum time a vehicle can be considered still ongoing from its first detection.
        """
        # Scale delta based thresholds by read_time_delta
        #MAX_MS_SINCE_LAST_EVENT *= read_time_delta/0.06 # Roughly give the vehicle twenty five read opportunities for a detection. This handles most large occlusions (e.g., trees.)
        
        # If data complete, move all tracked vehicles to finalized
        finished_vehicles = {}
        if final: finished_vehicles = self.tracking_dict
        # If data ongoing, compute completed tracked vehicles and move them to finalized
        else:
            for vehicle_id,vehicle in self.tracking_dict.items():
                finished = False
                # Get overall event time
                total_elapsed_time_sec = self.video.get_time_sec() - vehicle.start_time
                # Get last detected time delta
                last_event_delta_sec = self.video.get_time_sec() - vehicle.events[-1].event_time
                # Time cutoff comparison
                if total_elapsed_time_sec > MAX_VEHILCE_TOTAL_ELPASED_MS/1000: finished = True
                if last_event_delta_sec > MAX_MS_SINCE_LAST_EVENT/1000: finished = True
                # Finish
                #logger.info(f"{vehicle_id} - {finished}: total_elapsed_time {total_elapsed_time_sec} last_event_delta {last_event_delta_sec}, MAX_SEC_SINCE_LAST_EVENT {MAX_MS_SINCE_LAST_EVENT/1000}")
                if finished: finished_vehicles[vehicle_id] = vehicle
        
        # Finish the vehicles
        for vehicle in finished_vehicles.values(): 
            self.finish_tracked_vehicle(db, vehicle) 
        
        # Remove finished vehicles
        self.tracking_dict = {k: v for k, v in self.tracking_dict.items() if k not in finished_vehicles}
    
    def finish_tracked_vehicle(self, db: Session, vehicle: TrackedVehicle, MIN_EVENTS_FOR_VALID=3) -> bool:
        vehicle.direction = self.get_direction_estimate(vehicle.events)
        if vehicle.direction == None: 
            logger.warning(f"Vehicle Failed: unable to determine direction for vehicle {vehicle.vehicle_id}")
            self.cleanup_vehicle_data(vehicle, db)
            return False
        
        # Apply cropping based on vehicle direction
        if vehicle.direction == VehicleDirection.LEFT_TO_RIGHT:
            valid_events = [event for event in vehicle.events 
                            if event.bbox[0] + event.bbox[2] >= self.config.LEFT_CROP_l2r 
                            and event.bbox[0] <= self.config.RIGHT_CROP_l2r]
        elif vehicle.direction == VehicleDirection.RIGHT_TO_LEFT:
            valid_events = [event for event in vehicle.events 
                            if event.bbox[0] + event.bbox[2] >= self.config.LEFT_CROP_r2l 
                            and event.bbox[0] <= self.config.RIGHT_CROP_r2l]
        else:
            logger.warning(f"Vehicle Failed: invalid direction for vehicle {vehicle.vehicle_id}")
            self.cleanup_vehicle_data(vehicle, db)
            return False
        
        # Delete images associated with removed events
        for event in vehicle.events:
            if event not in valid_events:
                if event.image_path and os.path.exists(event.image_path):
                    os.remove(event.image_path)
        
        # Update vehicle events
        vehicle.events = valid_events
        
        # Check if we have enough events after cropping
        if len(vehicle.events) < MIN_EVENTS_FOR_VALID:
            logger.warning(f"Vehicle Failed: not enough events after cropping for vehicle {vehicle.vehicle_id} with {len(vehicle.events)} events and a threshold of {MIN_EVENTS_FOR_VALID}")
            self.cleanup_vehicle_data(vehicle, db)
            return False
        
        vehicle.elapsed_time = vehicle.events[-1].event_time - vehicle.start_time
        
        # Write events to disk
        events_data_path = self.write_events_data(vehicle)
        
        # Get thumbnail
        middle_index = len(vehicle.events) // 2

        # Update existing database entry instead of creating a new one
        db_vehicle = db.query(models.VehicleDetection).get(vehicle.vehicle_id)
        if db_vehicle is None:
            logger.warning(f"Vehicle {vehicle.vehicle_id} not found in database")
            return False

        db_vehicle.thumbnail_path = f"{self.detections_data_path}/{vehicle.vehicle_id}/bboxes"
        db_vehicle.events_data_path = events_data_path
        db_vehicle.direction = vehicle.direction
        db_vehicle.estimation_status = EstimationStatus.PENDING
        
        db.commit()
        
        logger.info(f"Finished tracking vehicle {vehicle.vehicle_id}... queued for estimation.")
        
        return True

    #### UTILS ####
    
    def get_next_image(self, num_to_read=1):
        """ Gets next target and label image. """
        for i in range(num_to_read): 
            self.target_img, _ = self.video.read()
            
            if self.target_img is None: return False
        
        return True
    
    def clean_image_paths_and_pending_detections(self, vehicle: TrackedVehicle):
        # Remove images for invalid detections
        try:
            for path in [event.image_path for event in vehicle.events]:
                if path and os.path.exists(path):
                    os.remove(path)
        except AttributeError as e:
            # Not sure why but sometimes vehicle is a dict not a TrackedVehicle object and we get an AttributeError. 
            logger.error(f"Error cleaning image paths: {e}")
    
    def push_image_to_frontend(self, image_path: str, db: Session):
        shutil.move(image_path, LATEST_DETECTION_IMAGE_PATH)
                        
        # Signal the frontend
        self.state = db.query(models.LiveDetectionState).first()
        self.state.has_new_image = True
        db.commit()
    
    def write_events_data(self, vehicle: TrackedVehicle):
        events_data_path = f"{self.detections_data_path}/{vehicle.vehicle_id}/events.json"
        os.makedirs(os.path.dirname(events_data_path), exist_ok=True)
        
        # Convert TrackedVehicle to JSON
        vehicle_json = vehicle.model_dump_json()
        
        # Write the JSON data to file
        with open(events_data_path, 'w') as f:
            f.write(vehicle_json)
            
        return events_data_path
    
    def load_tracked_vehicle(self, vehicle_id: str) -> TrackedVehicle:
        events_data_path = f"{self.detections_data_path}/{vehicle_id}/events.json"
        return TrackedVehicle.load_from_events_data(events_data_path)
    
    
    def cleanup_vehicle_data(self, vehicle: TrackedVehicle, db: Session):
        # Delete the vehicle directory if it's empty
        vehicle_dir = f"{self.detections_data_path}/{vehicle.vehicle_id}"
        if os.path.exists(vehicle_dir) and not os.listdir(vehicle_dir):
            shutil.rmtree(vehicle_dir)
        
        # Delete the database entry
        db_vehicle = db.query(models.VehicleDetection).get(vehicle.vehicle_id)
        if db_vehicle:
            db.delete(db_vehicle)
            db.commit()
            logger.debug(f"Deleted database entry for invalid vehicle {vehicle.vehicle_id}")
        
        # Remove the mapping
        for key, value in self.id_mapping.items():
            if value == vehicle.vehicle_id:
                del self.id_mapping[key]
                break