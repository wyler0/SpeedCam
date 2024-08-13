import os
os.environ["OPENCV_LOG_LEVEL"]="SILENT"
import json
import copy
import glob
import logging

import numpy as np
import cv2
# from scipy.optimize import linear_sum_assignment

# from detector import Detector
# from video import VideoStream

logger = logging.getLogger(__name__)

class SpeedEstimator():
    """ Speed Estimator class for vehicle detection. """
    
    
    tracked_vehicle_template = {
        "vehicle_id": None,
        "start_time": None, # Video time ms
        "elapsed_time": None, # Vehicle Last Event Time - Start time
        "speed_estimate": None,
        "speed_error": None,
        "direction": None, # See directions enum
        "events": [],
    }
    
    tracked_vehicle_event_template = {
        "frame_number": None,
        "event_time": None, # Relative to video start time
        "bbox": None # 0 - x, 1, - y, 2 xWidth, 3 yWidth
    }
    statusbar_height = 200
        
    def __init__(self, camera_calibration_id, config):
        """ Execute processor. 
            mode:
                LIVE - Stream and predict on live data
                FILE - Stream and predict on saved file
        """
        self.config = config
        self.init_constants()
        
        # Create detector
        self.detector = Detector(config)

        # Setup camera calibration
        if self.config.should_dewarp:
            print(f"Started dewarping calibration...")
            self.setup_camera_dewarp()
            print(f"Completed dewarping calibration.")
        
        # Setup distance calibration
        if self.config.use_calibration_clips:
            print(f"Started known vehicle calibration...")
            self.setup_distance_calibration()
            print(f"Completed known vehicle calibration.")
        
        # Get target stream
        self.video = VideoStream(
                config.input_video if config.input_video is not None else config.webcam_source, 
                WEBCAM_HFLIP=config.flip_input_h, WEBCAM_VFLIP=config.flip_input_v)
        self.video.start()
        
        # Skip forward as needed
        if config.skip_seconds > 0:
            self.video.offset_seeker_ms(config.skip_seconds*1000)
        
        # Get initial images
        self.target_imgs = [self.video.read(), self.video.read()]
        assert self.target_imgs[0] is not None, "Error loading target video"
    
    def start(self, writeout=True):
        self.start_time = self.video.get_time_ms()
        self.tracking_dict = {} #ID: tracked_vehicle_template
        self.finalized_tracking_dict = {} #ID: tracked_vehicle_template
        self.run_loop()
        self.check_finished_and_stale_events(final=True)
        self.process_results(writeout=writeout)
    
    def init_constants(self):
        ### Fix rounding problems with picamera resolution
        self.CAMERA_WIDTH = (self.config.CAMERA_WIDTH + 31) // 32 * 32 
        self.CAMERA_HEIGHT = (self.config.CAMERA_HEIGHT + 15) // 16 * 16

        # Define constants & conversion units
        #self.__PX_TO_KPH_L2R = float(cal_obj_mm_L2R / cal_obj_px_L2R * 0.0036) ### pixel width to speed conversion
        #self.__PX_TO_KPH_R2L = float(cal_obj_mm_R2L / cal_obj_px_R2L * 0.0036) 
        #if SPEED_MPH:
        #    self.__SPEED_UNITS = "mph"
        #    self.__SPEED_L2R_CONV = 0.621371 * self.__PX_TO_KPH_L2R
        #    self.__SPEED_R2L_CONV = 0.621371 * self.__PX_TO_KPH_R2L
        #else:
        #    self.__SPEED_UNITS = "kph"
        #    self.__SPEED_L2R_CONV = self.__PX_TO_KPH_L2R
        #    self.__SPEED_R2L_CONV = self.__PX_TO_KPH_R2L
            
        self.__SPEED_CONST_L2R = self.config.speed_const_l2r
        self.__SPEED_CONST_R2L = self.config.speed_const_r2l
        self.__MAX_SPEED_FILTER=160

        self.__cvWHITE = (255, 255, 255) ### basic opencv colors
        self.__cvBLACK = (0, 0, 0)
        self.__cvBLUE = (255, 0, 0)
        self.__cvGREEN = (0, 255, 0)
        self.__cvRED = (0, 0, 255)
        
        self.last_speed = -1
    
    ##### CORE ######
    def run_loop(self):
        while self.get_next_image():
            # Get bboxes
            self.bboxes = self.detector(self.target_imgs[0], self.target_imgs[1])
            
            # Update GUI
            if self.config.show_gui: self.update_gui(text="")
            #else: print(f"Status Update: {self.bboxes}")
            
            # If bboxes
            if len(self.bboxes) > 0:
                # Extract events from bboxes
                events = self.get_filter_new_events()
                events = self.correlate_events_and_vehicles(events)
                
                # Add event and or new vehicles
                for event_id, event in events:
                    self.add_vehicle_event(event["bbox"], event_id)
            
            # Check for finished events
            self.check_finished_and_stale_events()
            
            # Update GUI
            if self.config.show_gui: self.update_gui(text="", use_saved=True)
        
    def get_speed_estimate_constant(self, vehicle_id, direction):
        """ Estimates speed of vehicle or describes error given its id and completed events. """
        # Filter incompatibiles
        if len(self.tracking_dict[vehicle_id]["events"]) <= 2: return None, "Invalid event count."
        
        # Filter w/ left and right crops
        if direction == self.directions["RIGHTWARDS"]:
            self.tracking_dict[vehicle_id]["events"] = [
                k for k in self.tracking_dict[vehicle_id]["events"] if
                    k['bbox'][0] > self.config.LEFT_CROP_l2r and k['bbox'][0]+k['bbox'][2] < self.video.width-self.config.RIGHT_CROP_l2r
            ]
        elif direction == self.directions["LEFTWARDS"]:
            self.tracking_dict[vehicle_id]["events"] = [
                k for k in self.tracking_dict[vehicle_id]["events"] if
                    k['bbox'][0] > self.config.LEFT_CROP_r2l and k['bbox'][0]+k['bbox'][2] < self.video.width-self.config.RIGHT_CROP_r2l
            ]
        
        # Use the constant depending on the direction
        CONST = self.__SPEED_CONST_L2R if self.tracking_dict[vehicle_id]['direction'] == SpeedEstimator.directions['LEFTWARDS'] else self.__SPEED_CONST_R2L
        
        # Compute speed
        event_pairs = zip(self.tracking_dict[vehicle_id]["events"][:-1],self.tracking_dict[vehicle_id]["events"][1:]) # Get consequitive event pairs
        intra_frame_speeds = []
        for a, b in event_pairs:
            # Compute distance in pixels
            distance_pixels = abs(a['bbox'][0] - b['bbox'][0])
            
            # Compute time difference in seconds
            time_diff_ms = b['event_time'] - a['event_time']
            
            # Check if time difference is zero (which might be the case if frame rate is very high)
            if time_diff_ms == 0: 
                continue
            
            # Compute speed in feet per second
            speed_fps = distance_pixels / time_diff_ms * CONST

            # Convert speed to miles per hour
            speed_mph = speed_fps * 0.681818

            # Filter out speeds that exceed maximum speed
            if speed_mph <= self.__MAX_SPEED_FILTER:
                intra_frame_speeds.append(speed_mph)
                
        ##### Naive approach used by original code.
        #CONST = self.__SPEED_CONST_L2R if self.tracking_dict[vehicle_id]['direction'] == SpeedEstimator.directions['RIGHTWARDS'] else self.__SPEED_CONST_R2L
        #event_pairs = zip(self.tracking_dict[vehicle_id]["events"][:-1],self.tracking_dict[vehicle_id]["events"][1:]) # Get consequitive event pairs
        #intra_frame_speeds = [abs(a['bbox'][0]-b['bbox'][0])/(b['event_time']-a['event_time']) * CONST for a,b in event_pairs]
        #intra_frame_speeds = [i for i in intra_frame_speeds if i <= self.__MAX_SPEED_FILTER]
        #####
        
        self.last_speed = np.mean(intra_frame_speeds) if intra_frame_speeds else None
        return self.last_speed

    def get_speed_estimate_calibration(self, vehicle_id, direction):
        """ Estimates speed of vehicle or describes error given its id and completed events. """
        # Filter incompatibiles
        if len(self.tracking_dict[vehicle_id]["events"]) <= 2: return None #"Invalid event count."
        
        # Check for calibration data
        if direction == self.directions["RIGHTWARDS"] and len(self.rightwards_calibrations) == 0: return None #"No rightwards calibration data."
        elif direction == self.directions["LEFTWARDS"] and len(self.leftwards_calibrations) == 0: return None #"No leftwards calibration data."

        # Filter w/ left and right crops
        if direction == self.directions["RIGHTWARDS"]:
            self.tracking_dict[vehicle_id]["events"] = [
                k for k in self.tracking_dict[vehicle_id]["events"] if
                    k['bbox'][0] > self.config.LEFT_CROP_l2r and k['bbox'][0]+k['bbox'][2] < self.video.width-self.config.RIGHT_CROP_l2r
            ]
        elif direction == self.directions["LEFTWARDS"]:
            self.tracking_dict[vehicle_id]["events"] = [
                k for k in self.tracking_dict[vehicle_id]["events"] if
                    k['bbox'][0] > self.config.LEFT_CROP_r2l and k['bbox'][0]+k['bbox'][2] < self.video.width-self.config.RIGHT_CROP_r2l
            ]
        
        calibrations = self.rightwards_calibrations if self.tracking_dict[vehicle_id]['direction'] == self.directions['RIGHTWARDS'] else self.leftwards_calibrations
        
        if len(self.tracking_dict[vehicle_id]["events"]) <= 2: return None #"Invalid event count after filtering."
        
        # Compute speed
        event_pairs = zip(self.tracking_dict[vehicle_id]["events"][:-1],self.tracking_dict[vehicle_id]["events"][1:]) # Get consequitive event pairs
        intra_frame_speeds = []
        for a, b in event_pairs:
            # Compute distance in pixels
            distance_pixels = abs(a['bbox'][0] - b['bbox'][0])
            
            # Compute time difference in seconds
            time_diff_ms = b['event_time'] - a['event_time']

            # Check if time difference is zero (which might be the case if frame rate is very high)
            if time_diff_ms == 0: continue
            
            # Find closest calibration width for the first & following tracking box (left hand side)
            closest_calibration_a = min(calibrations, key=lambda x: abs(x[0] - a['bbox'][0]))
            closest_calibration_b = min(calibrations, key=lambda x: abs(x[0] - b['bbox'][0]))
            
            # Estimate distance based on first and second calibration
            a_distance_feet = self.config.calibration_vehicle_length * (distance_pixels/closest_calibration_a[1])/12
            b_distance_feet = self.config.calibration_vehicle_length * (distance_pixels/closest_calibration_b[1])/12
            
            # Compute speed in feet per second
            b_speed_fps = b_distance_feet / (time_diff_ms/1000)
            a_speed_fps = a_distance_feet / (time_diff_ms/1000)

            # Convert speed to miles per hour
            a_speed_mph = a_speed_fps * 0.68181818181818181818181818181818
            b_speed_mph = b_speed_fps * 0.68181818181818181818181818181818
            
            # Take average
            speed_mph = (a_speed_mph + b_speed_mph) / 2
            
            # Filter out speeds that exceed maximum speed
            if speed_mph <= self.__MAX_SPEED_FILTER:
                intra_frame_speeds.append(speed_mph)
                
        speed = np.mean(intra_frame_speeds) if intra_frame_speeds else None
        speed *= self.rightwards_calibrations_constant if direction == self.directions["RIGHTWARDS"] else self.leftwards_calibrations_constant
        
        self.last_speed = np.mean(intra_frame_speeds) if intra_frame_speeds else None
        print("Speed: ", self.last_speed)
        return self.last_speed

    def estimate_vehicle_speed(bounding_boxes, frames, fps, calibration_units):
        """Estimate the speed of a vehicle based on its bounding boxes and the calibration units."""
        
        def get_closest_calibration_unit(calibration_units, x):
            """Find the calibration unit with the closest x position to the given x."""
            return min(calibration_units, key=lambda unit: abs(unit[0] - x))
        
        # Initialize the previous x position and the previous frame
        prev_x = bounding_boxes[0][0]
        prev_frame = frames[0]

        # Initialize the list of speed estimates
        speeds = []

        for box, frame in zip(bounding_boxes[1:], frames[1:]):
            # Find the closest calibration unit to the x position of the bounding box
            calib_unit = get_closest_calibration_unit(calibration_units, box[0])

            # Compute the calibration factor (inches per pixel)
            calib_factor = calib_unit[2] / calib_unit[1]

            # Compute the displacement of the vehicle in inches
            displacement = (box[0] - prev_x) * calib_factor

            # Compute the time elapsed in seconds
            time_elapsed = (frame - prev_frame) / fps

            # Compute the speed in inches per second and convert it to miles per hour
            speed = displacement / time_elapsed * 0.05714  # 1 inch/sec equals approximately 0.05714 mph

            # Append the speed estimate to the list of speeds
            speeds.append(speed)

            # Update the previous x position and the previous frame
            prev_x = box[0]
            prev_frame = frame

        # Compute and return the average speed
        return sum(speeds) / len(speeds)
    
    def get_direction_estimate(self, events, allow_none=True):
        if len(events) < 2:
            raise ValueError("Not enough events to calculate direction")

        dx_sum = sum(events[i + 1]["bbox"][0] + events[i + 1]["bbox"][2] / 2 - (events[i]["bbox"][0] + events[i]["bbox"][2] / 2) for i in range(len(events) - 1))

        # Calculate average change in x
        dx_avg = dx_sum / (len(events) - 1)

        # Find lateral direction
        direction = None
        if dx_avg > 0: direction = self.directions["RIGHTWARDS"]
        elif dx_avg < 0: direction = self.directions["LEFTWARDS"]
        elif (not allow_none): raise ValueError("Direction could not be determined")

        return direction
    
    def get_filter_new_events(self):
        """ Extract events from bboxes via filtering according to configured parameters. 
        
        config:
            MIN_WIDTH (int): Minimum width of bbox to be considered an event.
            MAX_WIDTH (int): Maximum width of bbox to be considered an event.
            MIN_HEIGHT (int): Minimum height of bbox to be considered an event.
            MAX_HEIGHT (int): Maximum height of bbox to be considered an event.
        Returns: [event, event, ...]
        """
       
        
        events = []
        for i, bbox in self.bboxes.items():
            # width filter
            if not (bbox[2] > self.config.MIN_BBOX_WIDTH and bbox[2] < self.config.MAX_BBOX_WIDTH): continue
            # height filter
            if not (bbox[3] > self.config.MIN_BBOX_HEIGHT and bbox[3] < self.config.MAX_BBOX_HEIGHT): continue
            # botom filter
            if not (bbox[1]+bbox[3] > self.config.MIN_BBOX_Y): continue
            # create event
            new_event = copy.deepcopy(self.tracked_vehicle_event_template)
            new_event["frame_number"] = self.video.frames_read
            new_event["event_time"] = self.video.get_time_ms()
            new_event["bbox"] = bbox
            events.append(new_event)
        
        return events
    
    def correlate_events_and_vehicles(self, events):
        """ Correlates events to ongoing or new tracked vehicles. 
        
        config: 
            MAX_Y_DELTA: Maximum y-delta between event and tracked vehicle to be considered a match.
            MAX_MS_DELTA: Maximum video ms delta between event and tracked vehicle to be considered a match.
            X_DELTA_WEIGHT: Weight of x_delta in correlations calculations where y_delta weight is 1.
            NEW_EVENT_THRESH: Minimum correlation value for an event to be considered a new vehicle.
        Returns: 
            [(id, old_vehicle_new_event), ..., (None, new_vehicle_event)]
        """
        results = []
        
        candidate_vehicles = copy.deepcopy(self.tracking_dict)
        correlation_mat = np.zeros((len(events), max(len(events),len(candidate_vehicles))))+1e9 # [event, candidate] 
        overlap_rewards = [] # Reward for a new event overlapping with other existing events such that likely bad detections remain grouped.
        
        # Penalize for new events overlapping other new events.
        for event_idx, event in enumerate(events):
            overlap_reward = 0
            # Calculate overlap with other events
            for other_event in events:
                if event is other_event:
                    continue  # skip comparison with itself

                # Calculate overlap
                bbox1 = event["bbox"]
                bbox2 = other_event["bbox"]

                # Get coordinates for the intersection rectangle
                x_left = max(bbox1[0], bbox2[0])
                y_top = max(bbox1[1], bbox2[1])
                x_right = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
                y_bottom = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

                if x_right < x_left or y_bottom < y_top:
                    # No overlap, no penalty
                    continue

                # Compute the area of the intersection rectangle
                intersection_area = (x_right - x_left) * (y_bottom - y_top)

                # Compute the area of both bounding boxes
                bbox1_area = bbox1[2] * bbox1[3]
                bbox2_area = bbox2[2] * bbox2[3]

                # Compute overlap penalty based on the intersection area and the total area
                overlap_reward += intersection_area / (bbox1_area + bbox2_area - intersection_area)
            
            overlap_rewards.append(overlap_reward)
        
        for event_idx, event in enumerate(events):
            time_deltas = [] # Filtering measurement
            direction_matches = [] # Filtering measurement for events already containing 2+ detections.
            y_deltas = [] # Primary correlation measurement
            x_deltas = [] # Secondary correlation measurement
            predicted_directions = [] # Direction to use if correlated two single event vehicles.
        
            for vehicle_id, vehicle in candidate_vehicles.items():
                # Y Delta Calculation
                y_deltas.append(
                    abs((event["bbox"][1] + event["bbox"][3]) / 2 - (vehicle["events"][-1]["bbox"][1] + vehicle["events"][-1]["bbox"][3]) / 2)
                )
                
                # X Delta Calculation
                x_deltas.append(
                    (event["bbox"][0] + event["bbox"][2]) / 2 - (vehicle["events"][-1]["bbox"][0] + vehicle["events"][-1]["bbox"][2]) / 2
                )
                
                # Time Delta Calculation
                time_deltas.append(event["event_time"] - vehicle["events"][-1]["event_time"])
                
                # Direction Calculation
                direction_est = self.get_direction_estimate(vehicle["events"] + [event]) 
                direction_est = self.directions["RIGHTWARDS"] if (direction_est == 0 or direction_est == None) else direction_est
                predicted_directions.append(direction_est)
                
                # Direction Match Calculation
                direction_matches.append(direction_est == vehicle["direction"] if len(vehicle["events"]) > 1 else None)
                
            # Find best correlation, or new vehicle if no correlation is good enough  
            discarded = [False for i in candidate_vehicles]
            if len(candidate_vehicles) == 0: 
                correlation_mat[event_idx] = 1e9
                continue
            
            for i, time_delta in enumerate(time_deltas): 
                if time_delta > self.config.MAX_MS_DELTA: discarded[i] = True # filter by time delta
            
            for i, y_delta in enumerate(y_deltas):
                if y_delta > self.config.MAX_Y_DELTA: discarded[i] = True # filter by y delta
            
            for i, status in enumerate(discarded):
                if status: continue # already removed
                if direction_matches[i] is None: continue # target doesn't have enough events
                elif not direction_matches[i]: discarded[i] = True # filter by non matched directions
            
            # Compute weighted correlation based on xdelta, ydelta and if discarded for all candidates. Impossible candidates are assigned 1e9.
            correlations = [
                    x_delta * self.config.X_DELTA_WEIGHT \
                    + y_delta  *self.config.Y_DELTA_WEIGHT \
                    - overlap_rewards[event_idx] * self.config.OVERLAP_REWARD_WEIGHT
                if not discarded[i] else 1e9 
                for i,(x_delta,y_delta) in enumerate(zip(x_deltas,y_deltas))
            ]
            
            # Set 1e9 for correlations with nonexistent candidates
            correlations.extend([1e9 for i in range(len(correlation_mat[event_idx]) - len(correlations))])
             
            correlation_mat[event_idx] = np.array(correlations)
        
        # Use the Hungarian algorithm to find the optimal assignment
        event_indices, candidate_indices = linear_sum_assignment(correlation_mat)
        
        # Check the assignment to see if any event was assigned to a new candidate
        results = []

        # Iterate through the assignments to generate the results list
        for event_idx, candidate_idx in zip(event_indices, candidate_indices):
            if correlation_mat[event_idx, candidate_idx] >= 1e9:
                # The event needs a new candidate
                if np.min(correlation_mat[event_idx, :]) > self.config.NEW_EVENT_THRESH:
                    # If the best correlation is above the threshold,
                    results.append((None, events[event_idx]))
                else:
                    # Event is likely a bad detection
                    continue
            else:
                # Assign the best candidate for the event
                results.append((list(candidate_vehicles.keys())[candidate_idx], events[event_idx]))

        
        return results
    
    def check_finished_and_stale_events(self, final=False,
                                        MAX_MS_SINCE_LAST_EVENT=1000, MAX_VEHILCE_TOTAL_ELPASED_MS=3000):
        """ Finds ongoing tracked vehicles that have exited the frame. Finishes all open events if final is true. 
        
        config:
            MAX_MS_SINCE_LAST_EVENT: Maximum time since the vehicle was last tracked / appeared in a frame for event to be ongoing
            MAX_VEHILCE_TOTAL_ELPASED_MS: Maximum time a vehicle can be considered still ongoing from its first detection.
        """
        # If data complete, move all tracked vehicles to finalized
        finished_ids = []
        if final: finished_ids = list(self.tracking_dict.keys())
        # If data ongoing, compute completed tracked vehicles and move them to finalized
        else:
            for vehicle_id,tracking_data in self.tracking_dict.items():
                finished = False
                # Get overall event time
                total_elapsed_time = self.video.get_time_ms() - tracking_data['start_time']
                # Get last detected time delta
                last_event_delta = self.video.get_time_ms() - tracking_data['events'][-1]["event_time"]
                # Compute likelihood
                if total_elapsed_time > MAX_VEHILCE_TOTAL_ELPASED_MS: finished = True
                if last_event_delta > MAX_MS_SINCE_LAST_EVENT: finished = True
                # Finish
                if finished: finished_ids.append(vehicle_id)
                
        for vehicle_id in finished_ids: self.finish_tracked_vehicle(vehicle_id)
        self.update_gui()
        pass

    ##### DEWARP UTILITIES ######
    def setup_camera_dewarp(self):
        assert os.path.exists(self.config.dewarp_grids_dir), "Dewarp grids directory does not exist"
        imgs = glob.glob(self.config.dewarp_grids_dir+"/*.jpg")
        assert len(imgs) > 0, "No dewarp grids found. Please put grid jpg images in the directory."
        assert len(imgs) > 6, "Not enough dewarp grids found. Please put at least 6 grid jpg images in the directory."
        
        # Define the dimensions of checkerboard
        CHECKERBOARD = self.config.dewarp_grid_shape #(5, 8) #(4,6)
        
        # Set iteration early stopping minimium error or max iterations
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Vector for 3D points, 2D points
        # Vector for 2D points
        twodpoints, threedpoints = [],[]
        
        #  3D points real world coordinates
        objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        
        for filename in imgs:
            image = cv2.imread(filename)
            grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Threshold with in range funciton
            #grayColor = cv2.inRange(grayColor, 0, 100)
            #cv2.imshow('imgA', grayColor)
            #cv2.waitKey(0)
            # Find the chess board corners (ret=True if found)
            ret, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD, None)
            #cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
            # If desired number of corners can be detected then, refine pixel coordinates and display them
            if ret:
                # Refind pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria)
                twodpoints.append(corners2)
                threedpoints.append(objectp3d)
                
                # Draw and display the corners
                image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)
            
            else: 
                print("Failed to find corners in image: "+filename)
                continue

            if self.config.show_dewarp_grids:
                cv2.imshow('img', image)
                cv2.waitKey(0)
        
        cv2.destroyAllWindows()
        
        # Perform camera calibration by passing 3D points and corresponding pixel coordinates of detected corners
        ret, self.matrix, self.distortion, self.r_vecs, self.t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None, None)
        assert ret, "Failed to calibrate camera dewarp with provided grid images."
        
        print("Dewarping Calibration Complete")
        
        # Undistort Test
        if self.config.dewarp_test_image is not None and self.config.show_dewarp_grids:
            img = cv2.imread(self.config.dewarp_test_image)
            h, w = img.shape[:2]
            newcameramtx, roi=  cv2.getOptimalNewCameraMatrix(self.matrix,self.distortion,(w,h),1,(w,h))

            # Apply undistortion
            dst = cv2.undistort(img, self.matrix, self.distortion, None, newcameramtx)
            
            # Crop the result
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            cv2.imshow('Dewarped', dst)
            cv2.waitKey(0)

            # Estimate error
            error = 0
            for i in range(len(threedpoints)):
                imgpoints2, _ = cv2.projectPoints(threedpoints[i], self.r_vecs[i], self.t_vecs[i], self.matrix, self.distortion)
                error += cv2.norm(twodpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            
            print(f"Total Error: {error/len(threedpoints)}")
    
    def apply_camera_dewarp(self, img):
        h, w = img.shape[:2]
        newcameramtx, roi=  cv2.getOptimalNewCameraMatrix(self.matrix,self.distortion,(w,h),1,(w,h))

        # Apply undistortion and Crop to ROI
        dst = cv2.undistort(img, self.matrix, self.distortion, None, newcameramtx)
        #x, y, w, h = roi
        #dst = dst[y:y+h, x:x+w]
        return dst
    
    #### DISTANCE CALIBRATION UTILITIES ####
    def setup_distance_calibration(self):
        if not self.config.detector == 'YOLO': print("Distance calibration does not verify detections are correct. Please use YOLO detector for the best results.")
        assert os.path.exists(self.config.calibration_clips_path)
        assert len(glob.glob(self.config.calibration_clips_path+"/*.mp4")) > 0, "No calibration clips found. Please put mp4 videos in the directory."
        
        clips = glob.glob(self.config.calibration_clips_path+"/*.mp4")
        self.target_imgs = [None,None]
        trackings = {} # Clip: [(bbox_x, width), ...]
        for clipnum,clip in enumerate(clips):
            # Setup video
            trackings[clip] = {"direction": None, "detections": []}
            self.video = VideoStream(clip, WEBCAM_HFLIP=self.config.flip_calibration_clips_h)
            self.video.start(); self.target_imgs[0] = self.video.frame #img1
            self.video.read(); self.target_imgs[1] = self.video.frame #img2
            
            # Process Frames
            while not self.video.stopped:
                # Get detections
                self.bboxes = self.detector(self.target_imgs[0], self.target_imgs[1])
                if len(self.bboxes) > 0:
                    # Assume first bbox is correct
                    x = self.bboxes[0][0]
                    # Get bbox width
                    width = self.bboxes[0][2]
                    # Add to trackings
                    trackings[clip]['detections'].append((x, width))
                
                if self.config.show_gui_calibration: self.update_gui(text="")
                # Get next frame
                self.video.read(); self.target_imgs[0] = self.target_imgs[1]; self.target_imgs[1] = self.video.frame
            
            # Remove off screen detections
            detections = []
            for i, (x, width) in enumerate(trackings[clip]['detections']):
                if x <= 5 or x+width >= self.video.width-15: continue
                detections.append((x, width))
            trackings[clip]['detections'] = detections
            
            # Add direction
            if len(detections) < 2: del trackings[clip]; continue
            dx_sum = sum(detections[i+1][0] + detections[i+1][1]/2 - (detections[i][0] + detections[i][1]/2) for i in range(len(detections)-1))
            dx_avg = dx_sum / (len(detections) - 1)
            if dx_avg > 0: trackings[clip]['direction'] = self.directions["RIGHTWARDS"]
            elif dx_avg < 0: trackings[clip]['direction'] = self.directions["LEFTWARDS"]
            else: del trackings[clip]; continue
            
            self.bbox = []
            self.video = None
            print(f"\tProcessed calibration clip #{clipnum+1}/{len(clips)}")
        
        # Merge all trackings with same direction
        self.rightwards_calibrations, self.leftwards_calibrations = [], []
        for clip,data in trackings.items():
            if data['direction'] == self.directions["RIGHTWARDS"]: self.rightwards_calibrations.extend(data['detections'])
            elif data['direction'] == self.directions["LEFTWARDS"]: self.leftwards_calibrations.extend(data['detections'])
        
        # Sort detections by x
        self.rightwards_calibrations.sort(key=lambda x: x[0])
        self.leftwards_calibrations.sort(key=lambda x: x[0])
        
        # Comment out to plot detections
        if False:
            import matplotlib.pyplot as plt
            x_positions = [t[0] for t in self.leftwards_calibrations]
            widths = [t[1] for t in self.leftwards_calibrations]

            plt.figure(figsize=(10, 6))

            plt.scatter(x_positions, widths, label='Vehicle Width')
            plt.title('Vehicle Width for Leftward Movements')
            plt.xlabel('X-Position in Frame')
            plt.ylabel('Width (in pixels)')
            plt.legend()

            plt.show()
        
        # Calculate constants #FILL IN HERE...
        self.rightwards_calibrations_constant = 1.0
        self.leftwards_calibrations_constant = 1.0
        self.tune_distance_calibrations_constants()
        
    def tune_distance_calibrations_constants(self):
        # Get speed estimates:
        old_show_gui = self.config.show_gui
        self.config.show_gui = self.config.show_gui_calibration
        
        clips = {k:{} for k in glob.glob(self.config.calibration_clips_path+"/*.mp4")}
        for clipnum,clip in enumerate(clips.keys()):
            # Get target stream
            self.video = VideoStream(clip, WEBCAM_HFLIP=self.config.calibration_clips_hflip, WEBCAM_VFLIP=self.config.calibration_clips_vflip)
            self.video.start()
            while self.target_imgs[1] is None: self.target_imgs = [self.video.read(), self.video.read()]
            assert self.target_imgs[0] is not None, "Error loading target video"
            self.start(writeout=False)
            # Get speed
            assert len(self.finalized_tracking_dict) > 0, f"Error getting speed estimate. No vehicles in clip {clip}"
            assert len(self.finalized_tracking_dict) == 1, f"Error getting speed estimate. More than one vehicle in clip {clip}"
            clips[clip]['direction'] = self.finalized_tracking_dict[0]["direction"]
            clips[clip]['speed_estimate'] = self.finalized_tracking_dict[0]["speed_estimate"]
            print(f"\tTuned calibration clip #{clipnum+1}/{len(clips)}")
            
        # Actual speeds in calibration clips
        actual_speeds = self.config.calibration_clip_speeds  # [30,30, 30, 30, 30] #(mph)
        
        # Sort clips into directions
        leftwards_clips = [clip for clip in clips.values() if clip['direction'] == self.directions["LEFTWARDS"]]
        rightwards_clips = [clip for clip in clips.values() if clip['direction'] == self.directions["RIGHTWARDS"]]

        # Compute constants
        if leftwards_clips:
            leftwards_estimates = [clip['speed_estimate'] for clip in leftwards_clips]
            # Calibrate estimates using actual speeds
            calibration_ratios_leftwards = [actual / estimate for actual, estimate in zip(actual_speeds, leftwards_estimates) if estimate is not None]
            calibration_ratios_leftwards = calibration_ratios_leftwards if len(calibration_ratios_leftwards) > 0 else [1.0]
            self.leftwards_calibrations_constant = np.median(calibration_ratios_leftwards)

        if rightwards_clips:
            rightwards_estimates = [clip['speed_estimate'] for clip in rightwards_clips]
            # Calibrate estimates using actual speeds
            calibration_ratios_rightwards = [actual / estimate for actual, estimate in zip(actual_speeds, rightwards_estimates) if estimate is not None]
            calibration_ratios_rightwards = calibration_ratios_rightwards if len(calibration_ratios_rightwards) > 0 else [1.0]
            
            self.rightwards_calibrations_constant = np.median(calibration_ratios_rightwards)
            
        self.config.show_gui = old_show_gui
        
    ##### PROCESS UTILITIES ######
    def add_vehicle_event(self, bbox, vehicle_id):
        # Add new vehicle if id is None
        if vehicle_id is None or vehicle_id not in self.tracking_dict:
            vehicle_id = self.add_tracked_vehicle()
        
        # Add new event to vehicle
        self.tracking_dict[vehicle_id]["events"].append(copy.deepcopy(self.tracked_vehicle_event_template))
        
        # Populate frame number frame time and bbox
        self.tracking_dict[vehicle_id]["events"][-1].update({
            "frame_number": self.video.frames_read,
            "event_time": self.video.get_time_ms(),
            "bbox": bbox
        })
        
        # If events > 1, compute direction
        if len(self.tracking_dict[vehicle_id]['events']) > 1:
            self.tracking_dict[vehicle_id]["direction"] = self.get_direction_estimate(self.tracking_dict[vehicle_id]['events'])

    def finish_tracked_vehicle(self, vehicle_id,
                               MIN_EVENTS_FOR_VALID=20):
        if len(self.tracking_dict[vehicle_id]["events"]) < MIN_EVENTS_FOR_VALID: 
            del self.tracking_dict[vehicle_id]
            return
        
        direction = self.get_direction_estimate(self.tracking_dict[vehicle_id]["events"], allow_none=True)
        if direction == None: 
            print("Error: direction is None")
            del self.tracking_dict[vehicle_id]
            return

        time = self.tracking_dict[vehicle_id]["events"][-1]["event_time"] - self.tracking_dict[vehicle_id]["start_time"]
        
        if self.config.speed_estimate_method == "CONSTANT":
            speed = self.get_speed_estimate_constant(vehicle_id, direction)
        else:
            speed = self.get_speed_estimate_calibration(vehicle_id, direction)
        
        self.tracking_dict[vehicle_id].update({
            # Get direction
            "direction": direction,
            # Use last tracked frame time to calculate elapsed time
            "elapsed_time": time,
            # Calculate speed estimate
            "speed_estimate": speed
        })
        self.finalized_tracking_dict[vehicle_id] = self.tracking_dict[vehicle_id]
        del self.tracking_dict[vehicle_id]
    
    def add_tracked_vehicle(self):
        # Get new vehicle ID
        vehicle_id = max(self.tracking_dict.keys()) + 1 if len(self.tracking_dict) > 0 else 0
        # Add new vehicle to tracking dict
        self.tracking_dict[vehicle_id] = copy.deepcopy(self.tracked_vehicle_template)
        # Populate vehicle id and start_time
        self.tracking_dict[vehicle_id].update({
            "vehicle_id": vehicle_id,
            "start_time": self.video.get_time_ms()
        })
        
        return vehicle_id

    def process_results(self, writeout=True):
        """ Processes results and saves to file. """
        # Fix IDs
        old_ids = copy.deepcopy(self.finalized_tracking_dict)
        self.finalized_tracking_dict = {}
        for new_id, old_id in enumerate(old_ids.keys()):
            self.finalized_tracking_dict[new_id] = old_ids[old_id]
        
        # Create output data
        data = {}
        data['results'] = self.finalized_tracking_dict
        data['config'] = vars(self.config)
        
        # Write output data
        if writeout:
            with open(self.config.output_file, 'w') as outfile:
                json.dump(self.finalized_tracking_dict, outfile)
    
    def get_next_image(self, num_to_read=1):
        """ Gets next target and label image. """
        self.target_imgs[0] = self.target_imgs[1]
        for i in range(num_to_read): 
            self.target_imgs[1] = self.video.read()
        
        # Apply dewarping if enabled
        if self.target_imgs[1] is not None and self.config.should_dewarp:
            self.target_imgs[1] = self.apply_camera_dewarp(self.target_imgs[1])
        return not self.target_imgs[1] is None
    
    ##### GUI #####
    def update_gui(self, text="", use_saved=False):
        """ Displays target on LHS and label on RHS. """
        if not use_saved:
            # Construct Image   
            image_to_show = np.zeros((self.video.height+self.statusbar_height, self.video.width, 3), dtype=np.uint8)
            # Add Target Video 
            image_to_show[:self.video.height, :self.video.width] = self.target_imgs[1] if self.target_imgs[1] is not None else np.zeros((self.video.height, self.video.width, 3), dtype=np.uint8)
            # Add BBoxes to target image with IDs
            image_to_show = self.draw_bboxes(image_to_show)
            
            # Add l2r crop
            image_to_show = cv2.rectangle(image_to_show, (self.config.LEFT_CROP_l2r, int(self.video.height/2)), (self.config.LEFT_CROP_l2r+5, self.video.height), (0, 255, 0), 2)
            image_to_show = cv2.rectangle(image_to_show, (self.video.width-self.config.RIGHT_CROP_l2r,  int(self.video.height/2)), (self.video.width-self.config.RIGHT_CROP_l2r+5, self.video.height), (0, 255, 0), 2)
            # Add r2l crop
            image_to_show = cv2.rectangle(image_to_show, (self.config.LEFT_CROP_r2l, 0), (self.config.LEFT_CROP_r2l+5, int(self.video.height/2)), (0, 255, 0), 2)
            image_to_show = cv2.rectangle(image_to_show, (self.video.width-self.config.RIGHT_CROP_r2l, 0), (self.video.width-self.config.RIGHT_CROP_r2l+5, int(self.video.height/2)), (0, 255, 0), 2)
            
            # Save for faster loading for text-only updates
            self.saved_gui = image_to_show.copy()
        else:
            image_to_show = self.saved_gui.copy()
        
        # Add prompt text
        image_to_show = self.draw_text(image_to_show, self.get_status_text(text))

        cv2.imshow("Speed Measurer", image_to_show)
        cv2.waitKey(1)
    
    def get_status_text(self, text): #TODO
        """ Returns tracking status text of ongoing vehicles for GUI. """
        #FPS
        text += f"  |  FPS: {self.video.proc_fps:.1f}"
        
        #Seconds into video
        time_hours = int(self.video.get_time_ms()/1000/60/60) % 24
        time_minutes = int(self.video.get_time_ms()/1000/60) % 60
        time_seconds = int(self.video.get_time_ms()/1000) % 60
        time = "%02d:%02d:%02d" % (time_hours, time_minutes, time_seconds)
        text += f"  |  Time: {time}"
        
        #Last speed to 0 decimals
        text += f"  |  Last Speed: {self.last_speed:.0f} mph"
        
        return text
    
    def draw_text(self, image, text):
        # Place text centered in statusbar (on the top, width is vid width hieght is statusbar_height)
        cv2.putText(image, text, (10, self.video.height+int(self.statusbar_height/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, self.__cvRED, 2)
        
        return image
    
    def draw_bboxes(self, image):
        """ Draws bboxes on target image. """
        for i, bbox in self.bboxes.items():
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 0, 0), 2)
            cv2.putText(image, str(i), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return image
