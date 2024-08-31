import { useState, useEffect, useCallback } from 'react';
import { getEndpoint } from '@/api/endpoints';
import toast from 'react-hot-toast';

export interface DetectionStatus {
  running: boolean;
  speed_calibration_id: string | null;
  camera_source: number | null;
  is_calibrating: boolean;
  processing_video: boolean;
}

export interface CameraInfo {
  id: string;
  name: string;
}

export function useSharedDetectionStatusService() {
  const [isDetectionOn, setIsDetectionOn] = useState(false);
  const [availableCameras, setAvailableCameras] = useState<CameraInfo[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);
  const [processingVideo, setProcessingVideo] = useState(false);
  const [isCalibrationMode, setIsCalibrationMode] = useState(false);

  const fetchDetectionStatus = useCallback(async () => {
    try {
      const response = await fetch(getEndpoint('DETECTION_STATUS'));
      const data: DetectionStatus = await response.json();
      setIsDetectionOn(data.running);
      setSelectedCamera(data.camera_source?.toString() || null);
      setProcessingVideo(data.processing_video);
      setIsCalibrationMode(data.is_calibrating);
    } catch (error) {
      console.error("Error fetching detection status:", error);
    }
  }, []);

  const fetchAvailableCameras = useCallback(async () => {
    try {
      const response = await fetch(getEndpoint('AVAILABLE_CAMERAS'));
      const data = await response.json();
      if (typeof data === 'object' && data !== null) {
        const cameras: CameraInfo[] = Object.entries(data.available_cameras).map(([id, name]) => ({
          id,
          name: name as string,
        }));
        setAvailableCameras(cameras);
      } else {
        console.error("Unexpected data format for available cameras:", data);
      }
    } catch (error) {
      console.error("Error fetching available cameras:", error);
    }
  }, []);

  useEffect(() => {
    fetchDetectionStatus();
    fetchAvailableCameras();

    const intervalId = setInterval(fetchDetectionStatus, 5000);

    return () => clearInterval(intervalId);
  }, [fetchDetectionStatus, fetchAvailableCameras]);

  const updateSelectedCamera = async (id: string) => {
    setSelectedCamera(id);
    try {
      const response = await fetch(`${getEndpoint('LIVE_DETECTION')}/update?camera_source=${encodeURIComponent(id)}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to update selected camera');
      }

      await fetchDetectionStatus();
    } catch (error) {
      console.error('Error updating selected camera:', error);
      setSelectedCamera(null);
      toast.error('Failed to update selected camera. Please try again.', {
        position: 'bottom-right',
        duration: 4000,
      });
    }
  };

  return {
    isDetectionOn,
    availableCameras,
    selectedCamera,
    processingVideo,
    isCalibrationMode,
    fetchDetectionStatus,
    updateSelectedCamera,
  };
}