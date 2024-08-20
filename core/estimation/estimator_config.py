class EstimatorConfig:
    pass
    #### PROGRAM ####
    mode = 'speed'
    detector = 'YOLO'
    input_video = None
    output_file = 'labels.json'
    webcam_source = 0
    label_video = None
    flip_label_h = False
    flip_label_v = False
    flip_input_h = False
    flip_input_v = False
    alignment_offset = 0
    skip_seconds = 0
    show_gui = True
    
    #### LOGGING ####
    loggingToFile = False
    logFilePath = 'speed-cam.log'
    verbose = True
    
    #### SPEED ESTIMATE ####
    speed_estimate_method = 'CALIBRATION' #'CALIBRATION' # CONSTANT = Old Linear Method w/ below constants, CALIBRATION = New calibration method w/ calibration videos
    speed_const_l2r = 35
    speed_const_r2l = 55
    
    #### DISTANCE CALIBRATION ####
    use_calibration_clips = True
    show_gui_calibration = False
    calibration_clips_path = 'tests/test_data/test_cam_A/calibration_clips'
    calibration_vehicle_length = 178.2 # inches
    calibration_clip_speeds = [31, 30, 34, 33, 33, 30.5, 31, 31, 35, 30] # MPH, not yet used.
    calibration_clips_hflip = False
    calibration_clips_vflip = False
    flip_calibration_clips_h = True
    
    #### DEWARPING ####
    should_dewarp = True
    show_dewarp_grids = False
    dewarp_test_image = 'data/calibration/test_img.png'
    dewarp_grids_dir = 'data/calibration/SpeedCam Grids v3'
    dewarp_grid_shape = (6, 8)
    
    #### CORRELATION ####
    MAX_Y_DELTA = 350 # Maximum vertical distance between events to still be considered the same event.
    MAX_MS_DELTA = 800 # Maximum time between events to still be considered the same event. 
    X_DELTA_WEIGHT = 0.1 # Larger = more likely to be considered new vehicle (less likely to be considered same vehicle)
    Y_DELTA_WEIGHT = 10 # Larger = more likely to be considered new vehicle (less likely to be considered same vehicle)
    NEW_EVENT_THRESH = 500 # Larger = less likely to accept uncorrealted event as new vehicle
    OVERLAP_REWARD_WEIGHT = 2000 # Larger = less likely to be considered new vehicle.
    
    #### DETECTION ####
    BLUR_SIZE = 10
    MIN_BBOX_WIDTH = 150
    MIN_BBOX_HEIGHT = 10
    MAX_BBOX_WIDTH = 900
    MAX_BBOX_HEIGHT = 1e9
    MIN_BBOX_Y = 550
    THRESHOLD_SENSITIVITY = 3
    LEFT_CROP_l2r = 0 # Distance from LHS of frame where detection does not occur
    RIGHT_CROP_l2r = 0 # Distance from RHS of frame where detection does not occur
    LEFT_CROP_r2l = 200 # Distance from LHS of frame where detection does not occur
    RIGHT_CROP_r2l = 100 # Distance from RHS of frame where detection does not occur

    #### YOLO ####
    yolo_config = "data/yolo/yolov4-tiny.cfg"
    yolo_weights = "data/yolo/yolov4-tiny.weights"
    yolo_classes = "data/yolo/coco.names"
    YOLO_CONFIDENCE_THRESHOLD = 0.4
    
    #### CAMERA ####
    WEBCAM_HFLIP = False
    WEBCAM_VFLIP = False
    CAMERA_WIDTH = 320
    CAMERA_HEIGHT = 240