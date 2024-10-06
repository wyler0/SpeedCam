// Base URL for the API
export const BASE_URL = 'http://localhost:8000';
export const API_BASE_URL = `${BASE_URL}/api/v1`;

// API endpoints
export const API_ENDPOINTS = {
  // Camera calibrations
  CAMERA_CALIBRATIONS: `${API_BASE_URL}/camera-calibrations`,
  UPLOAD_CALIBRATION_IMAGE: `${API_BASE_URL}/camera-calibrations/{calibration_id}/upload-image`,
  PROCESS_CALIBRATION: `${API_BASE_URL}/camera-calibrations/{calibration_id}/process`,

  // Vehicle detections
  VEHICLE_DETECTIONS: `${API_BASE_URL}/vehicle-detections`,
  
  // Speed calibrations
  SPEED_CALIBRATIONS: `${API_BASE_URL}/speed-calibrations`,
  SPEED_CALIBRATIONS_SUBMIT: `${API_BASE_URL}/speed-calibrations/{calibration_id}/submit`,
  SPEED_CALIBRATIONS_UPDATE: `${API_BASE_URL}/speed-calibrations/{calibration_id}`,
  UPDATE_SPEED_CALIBRATION_CROP: `${API_BASE_URL}/speed-calibrations/{calibration_id}/crop`,
  
  // Speed limit
  SPEED_LIMIT: `${API_BASE_URL}/speed-calibrations/speed-limit`,
  
  // Live detection
  LIVE_DETECTION: `${API_BASE_URL}/live-detection`,
  AVAILABLE_CAMERAS: `${API_BASE_URL}/live-detection/available_cameras`,
  START_DETECTION: `${API_BASE_URL}/live-detection/start`,
  STOP_DETECTION: `${API_BASE_URL}/live-detection/stop`,
  DETECTION_STATUS: `${API_BASE_URL}/live-detection/status`,
  LATEST_IMAGE_STATUS: `${API_BASE_URL}/live-detection/latest_image_status`,
  LATEST_IMAGE_URL: `${BASE_URL}/static/latest_detection_image.png`,
  UPLOAD_SPEED_CALIBRATION_VIDEO: `${API_BASE_URL}/live-detection/{calibration_id}/upload-video`,

  // Validate calibration image
  VALIDATE_CALIBRATION_IMAGE: `${API_BASE_URL}/camera-calibrations/validate`,

  // Add more endpoints as needed
};

// Export a function to get a specific endpoint
export const getEndpoint = (key: keyof typeof API_ENDPOINTS) => API_ENDPOINTS[key];