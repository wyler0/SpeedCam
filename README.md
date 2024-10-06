# SpeedCam: Real-Time Vehicle Speed Detection System

## Overview

SpeedCam is an advanced vehicle speed detection system that utilizes computer vision and machine learning techniques to accurately measure and record vehicle speeds in real-time. This system is designed for traffic monitoring, law enforcement, and road safety applications.

## How It Works

1. **Object Detection**: The system uses a YOLO-based object detection algorithm to identify vehicles in real-time video streams.

2. **Speed Estimation**: Once vehicles are detected, their speed is estimated using Farneback optical flow. This estimation process runs in a separate thread to ensure that real-time processing is not impacted.

3. **Calibration**: The system requires both camera calibration (dewapring of the captured image) and speed calibration (pixel to world speed conversion) to accurately convert pixel movements to real-world speeds.

4. **User Interface**: A web-based frontend allows users to monitor detections, manage calibrations, and control the detection process.

## Key Features

- Real-time vehicle detection and speed estimation
- Separate processing threads for detection and estimation
- Camera calibration for lens distortion correction
- Speed calibration for accurate speed measurements
- Web-based user interface for easy monitoring and control

## System Requirements

- Macbook Pro M Series (for MLX and GPU capabilities)
- Python 3.11 or higher
- Node.js

## Installation

1. Install Homebrew (for macOS users):
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Install Python 3.11 or higher:
   ```
   brew install python@3.11
   ```

3. Install Node.js:
   ```
   brew install node
   ```

4. Clone the repository:
   ```
   git clone https://github.com/your-repo/speedcam.git
   cd speedcam
   ```

5. Start the application:
   ```
   npm run dev
   ```
   This command will:
   - Create a Python virtual environment
   - Install Python requirements
   - Start the backend server
   - Start the frontend development server

6. Navigate to http://localhost:3000 (or whatever port was used.)

## Setup and Usage

### 1. Camera Calibration

Before using the system, you need to create a camera calibration:

1. Capture at least 8 images of a checkerboard grid pattern from different angles.
2. Use the "Create Camera Calibration" functionality in the frontend.
3. Upload the grid images and process the calibration.

This step corrects for lens distortion and ensures accurate measurements.

### 2. Speed Calibration

After camera calibration, you need to create a speed calibration:

1. Record video of vehicles traveling at known speeds in both lanes.
2. Use the "Create Speed Calibration" feature in the frontend.
3. Upload the calibration video and enter the known speeds.
4. The system will process the video and calculate the necessary constants to convert pixel movement to real-world speeds.

### 3. Running the Detector

Once calibrations are complete, you can start detecting vehicle speeds:

1. Navigate to the top of the frontend UI.
2. Use the status selector to choose your camera input and speed calibration.
3. Optionally define a speed limit in your unit of speed which is used to compute the high level statistics. Hit enter to submit a new speed limit.
3. Click "Start Detector" to begin real-time speed detection.

### 4. Monitoring and Using Results

Once the detector is running, you can monitor and analyze the results in real-time:

1. **Live Detection Feed**: At the top of the UI, you'll see a live image feed showing the current detections. This allows you to visually confirm that the system is working correctly.

2. **Detection Statistics**: Below the live feed, you'll find a summary of detection statistics for the last seven days:
   - Total number of vehicles detected
   - Average speed of detected vehicles
   - Number of speeding violations (based on the currently selected speed limit)

3. **Speed Graph**: The speed graph displays detected speeds for the last seven days, providing a quick overview of speed trends and patterns.

4. **Detection Table**: Further down, you'll find a scrollable table showing individual detections. Each row in the table represents a single vehicle detection, including details such as timestamp, detected speed, and lane information.

5. **Data Export**: The detection table includes an export feature that allows you to download your detection data:
   - Click the "Export" button to save the detection data as a CSV file.
   - You have the option to include images of each detection in the export.

6. **Custom Filters**: In the vehicle chart section, you can customize the filters to view historical results for any desired time range:
   - Click on the "Filters" option in the vehicle chart section.
   - Set your preferred time range and any other relevant filters.
   - The chart and detection table will update to show results based on your selected filters.

This comprehensive view allows you to monitor real-time detections, analyze speed trends over time, export detailed data, and customize your view of historical data for in-depth analysis or reporting.

## Contributing

[Instructions for contributors, if applicable]

## License

[Specify the license under which this project is released]

## Support

[Provide contact information or links for support]