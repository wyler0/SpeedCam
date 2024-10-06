# © 2024 Wyler Zahm. All rights reserved.

import unittest
import os
import time
from datetime import datetime
import multiprocessing

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from database import set_database_url, get_default_session_factory
from core.detection.detector import SpeedDetector, DetectorConfig
from core.detection.detection_schemas import VehicleDirection
from models import Base, CameraCalibration, SpeedCalibration, LiveDetectionState, VehicleDetection
from schemas import (
    CameraCalibrationCreate,
    SpeedCalibrationCreate,
    LiveDetectionStateCreate,
    VehicleDetection as VehicleDetectionSchema
)

class TestSpeedEstimator(unittest.TestCase):
    state: LiveDetectionState = None
    config: DetectorConfig = None
    detector: SpeedDetector = None
    
    def setup_db(self, video_path: str) -> tuple[Session, LiveDetectionState]:
        # Create an in-memory SQLite database
        set_database_url('sqlite:///./test.db')
        
        # Create all tables in the engine
        get_default_session_factory().create_database()
        
        # Get the db
        with get_default_session_factory().get_db_with() as db:
            # Create a test configuration
            self.config = DetectorConfig()

            # Create some test data using Pydantic models
            camera_calibration_data = CameraCalibrationCreate(
                camera_name="Test Camera",
                calibration_date=datetime.now(),
                rows=6,
                cols=8,
                calibration_matrix=[
                    [1427.352671431912, 0, 966.7577494824117],
                    [0, 1429.8530099082234, 674.2828018929074],
                    [0, 0, 1]
                ],
                distortion_coefficients=[
                    [0.22586511176071725, -0.5346156599122224, 0.0003958370205917897,
                    -0.0010333285038387622, 0.3806859810070146]
                ],
                rotation_matrix=[
                    [
                        [
                        0.06574610556560583
                        ],
                        [
                        0.05382206317565781
                        ],
                        [
                        -1.561513502657803
                        ]
                    ],
                    [
                        [
                        -0.06528320915246726
                        ],
                        [
                        -0.012115815401895325
                        ],
                        [
                        1.5656411983358511
                        ]
                    ],
                    [
                        [
                        -0.2913393653404885
                        ],
                        [
                        -0.23573849880302467
                        ],
                        [
                        1.5759189505425433
                        ]
                    ],
                    [
                        [
                        -0.06610817408090185
                        ],
                        [
                        -0.01579961580080157
                        ],
                        [
                        1.5444006543840616
                        ]
                    ],
                    [
                        [
                        -0.028167799667798938
                        ],
                        [
                        -0.01924831698513234
                        ],
                        [
                        -1.5434720230491281
                        ]
                    ],
                    [
                        [
                        -0.0564678251366376
                        ],
                        [
                        0.4747963566566645
                        ],
                        [
                        1.447775342594215
                        ]
                    ],
                    [
                        [
                        -0.23126101745531458
                        ],
                        [
                        0.27134245404767354
                        ],
                        [
                        1.5178954967212965
                        ]
                    ],
                    [
                        [
                        -0.28213718811538274
                        ],
                        [
                        -0.2243035324752586
                        ],
                        [
                        1.551851022072824
                        ]
                    ],
                    [
                        [
                        0.6511459567077419
                        ],
                        [
                        0.0235729589983813
                        ],
                        [
                        1.3000087772537174
                        ]
                    ],
                    [
                        [
                        -0.019309971034963982
                        ],
                        [
                        0.018296719257803764
                        ],
                        [
                        -1.5248333066897906
                        ]
                    ]
                ],
                translation_vector=[
                    [
                        [
                        -2.917448731643787
                        ],
                        [
                        3.158289962125551
                        ],
                        [
                        17.485211783717098
                        ]
                    ],
                    [
                        [
                        7.790557456226198
                        ],
                        [
                        -5.07113934135698
                        ],
                        [
                        13.828018275821819
                        ]
                    ],
                    [
                        [
                        -0.32800616733383336
                        ],
                        [
                        -3.844972700243291
                        ],
                        [
                        17.25115685718428
                        ]
                    ],
                    [
                        [
                        7.8807983645893875
                        ],
                        [
                        -3.4145499993805917
                        ],
                        [
                        14.179025604859516
                        ]
                    ],
                    [
                        [
                        -8.846137028455352
                        ],
                        [
                        0.6823531320424631
                        ],
                        [
                        15.821903116891223
                        ]
                    ],
                    [
                        [
                        4.833762979097013
                        ],
                        [
                        -3.04191653255552
                        ],
                        [
                        12.33405711710908
                        ]
                    ],
                    [
                        [
                        4.051757975337433
                        ],
                        [
                        -2.225384364752566
                        ],
                        [
                        13.77242393947707
                        ]
                    ],
                    [
                        [
                        -0.07088128312266802
                        ],
                        [
                        -3.6456382874551494
                        ],
                        [
                        14.968426354876256
                        ]
                    ],
                    [
                        [
                        4.75886190350652
                        ],
                        [
                        -3.2226557236889004
                        ],
                        [
                        11.58058491015783
                        ]
                    ],
                    [
                        [
                        -9.64994892681292
                        ],
                        [
                        1.7856482647276364
                        ],
                        [
                        17.42345664848074
                        ]
                    ]
                ],
                valid=True
            )
            camera_calibration = CameraCalibration(**camera_calibration_data.model_dump())
            db.add(camera_calibration)
            db.flush()  # This will populate the id field

            speed_calibration_data = SpeedCalibrationCreate(
                name="Test Speed Calibration",
                calibration_date=datetime.now(),
                camera_calibration_id=camera_calibration.id,  # Set this explicitly
                valid=False,
                left_to_right_constant=None,
                right_to_left_constant=None,
            )
            speed_calibration = SpeedCalibration(**speed_calibration_data.model_dump())
            db.add(speed_calibration)
            db.flush()

            state_data = LiveDetectionStateCreate(
                is_calibrating=True,
                video_path=video_path,
                speed_calibration_id=speed_calibration.id,
                started_at=None,
                running=False,
                error=None
            )
            state = LiveDetectionState(**state_data.model_dump())
            db.add(state)

            db.commit()

        return state
    
    def tearDown(self):
        set_database_url('sqlite:///./test.db')
        Base.metadata.drop_all(get_default_session_factory().engine)
        with get_default_session_factory().get_db_with() as db:
            db.close()
        
        if os.path.exists("test.db"): 
            os.remove("test.db")

    def start_detector(self, profile=False) -> multiprocessing.Event:
        with get_default_session_factory().get_db_with() as db:
            self.state = db.query(LiveDetectionState).first()
            if self.state:
                self.state.started_at = datetime.now()
                self.state.running = True
                db.commit()
            else:
                raise Exception("Live detection state not initialized.")
        return self.detector.start(profile=profile)
    
    def stop_detector(self, stop_signal: multiprocessing.Event):
        stop_signal.set()
        self.await_detector()
        
    def await_detector(self):
        self.detector.await_completion()
        self.state.running = False
        with get_default_session_factory().get_db_with() as db:
            db.commit()
    
    def Xtest_speed_estimation_one_vehicle(self):
        # Setup DB and Config and Estimator
        self.state = self.setup_db("data/test_data/videos/07_08_2024/split_1.mp4")
        self.config = DetectorConfig()
        length = 6.133008
        self.detector = SpeedDetector(self.config)
        
        # Start
        timer = time.time()
        _ = self.start_detector()
        
        # Wait for thread to finish
        self.await_detector()
        
        print(f"Time taken: {time.time() - timer}, length: {length}, slow down: {length / (time.time() - timer)}")
        
        # Check if detection was successful
        with get_default_session_factory().get_db_with() as db:
            detections = db.query(VehicleDetection).all()
        
        # Verify that the detections match the schema
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].direction, VehicleDirection.LEFT_TO_RIGHT)
    
    def test_speed_estimation_two_vehicles(self):
        # Setup DB and Config and Estimator
        self.state = self.setup_db("data/test_data/videos/07_08_2024/split_two_vehicles.mp4")
        self.config = DetectorConfig()
        self.detector = SpeedDetector(self.config)
        length = 4.100000
        
        # Start
        timer = time.time()
        stop_signal = self.start_detector(profile=True)
        
        # Wait for thread to finish
        self.await_detector()
        print(f"Time taken: {time.time() - timer}, length: {length}, slow down: {length / (time.time() - timer)}")
        # Check if detection was successful
        with get_default_session_factory().get_db_with() as db:
            detections = db.query(VehicleDetection).all()
        
        # Verify that the detections match the schema
        self.assertEqual(len(detections), 2)
        directions = [d.direction for d in detections]
        self.assertIn(VehicleDirection.LEFT_TO_RIGHT, directions)
        self.assertIn(VehicleDirection.RIGHT_TO_LEFT, directions)
        
    def Xtest_speed_estimation_early_stop(self):
        # Setup DB and Config and Estimator
        self.state = self.setup_db("data/test_data/videos/07_08_2024/split_additional_frames.mp4")
        self.config = DetectorConfig()
        self.detector = SpeedDetector(self.config)
        length = 11.000000
        
        # Run
        timer = time.time()
        signal = self.start_detector()
        time.sleep(1) # Cut off early
        self.stop_detector(signal)
        print(f"Time taken: {time.time() - timer}, length: {length}, slow down: {length / (time.time() - timer)}")
        
        # Check if detection was successful
        with get_default_session_factory().get_db_with() as db:
            detections = db.query(VehicleDetection).all()
        
        # Verify that the no vehicles were detected due to early stop
        self.assertEqual(len(detections), 0)

if __name__ == "__main__":
    unittest.main()