import { useState, useEffect, useCallback } from 'react';
import { getEndpoint } from '@/api/endpoints';
import toast from 'react-hot-toast';

interface DetectionStatus {
  running: boolean;
  speed_calibration_id: string | null;
  camera_id: number | null;
}

interface CameraInfo {
  id: string;
  name: string;
}

export function useDetectionStatusService() {
  const [isDetectionOn, setIsDetectionOn] = useState(false);
  const [speedCalibrationId, setSpeedCalibrationId] = useState<string | null>(null);
  const [calibrationIds, setCalibrationIds] = useState<string[]>([]);
  const [availableCameras, setAvailableCameras] = useState<CameraInfo[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);

  const fetchDetectionStatus = useCallback(async () => {
    try {
      const response = await fetch(getEndpoint('DETECTION_STATUS'));
      const data: DetectionStatus = await response.json();
      setIsDetectionOn(data.running);
      setSpeedCalibrationId(data.speed_calibration_id);
      setSelectedCamera(data.camera_id?.toString() || null);
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

  const fetchCalibrationIds = useCallback(async () => {
    try {
      const response = await fetch(getEndpoint('SPEED_CALIBRATIONS'), {
        method: 'GET',
      });
      const data = await response.json();
      if (Array.isArray(data)) {
        setCalibrationIds(data);
      } else {
        console.error("Unexpected data format for calibration_ids:", data);
      }
    } catch (error) {
      console.error("Error fetching calibration IDs:", error);
    }
  }, []);

  useEffect(() => {
    fetchDetectionStatus();
    fetchCalibrationIds();
    fetchAvailableCameras();

    const intervalId = setInterval(fetchDetectionStatus, 5000);

    return () => clearInterval(intervalId);
  }, [fetchDetectionStatus, fetchCalibrationIds, fetchAvailableCameras]);

  const toggleDetection = async () => {
    const newStatus = !isDetectionOn;

    if (newStatus && (!speedCalibrationId || selectedCamera === null)) {
      toast.error('Please select both a calibration and a camera before starting detection.', {
        position: 'bottom-right',
        duration: 4000,
      });
      return;
    }

    try {
      const endpoint = newStatus ? 'start' : 'stop';
      const response = await fetch(`${getEndpoint('LIVE_DETECTION')}/${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ camera_id: selectedCamera }),
      });

      if (!response.ok) {
        throw new Error(`Failed to ${endpoint} detection`);
      }

      await fetchDetectionStatus();
      
      toast.success(`Detection ${newStatus ? 'started' : 'stopped'} successfully`, {
        position: 'bottom-right',
        duration: 3000,
      });
    } catch (error) {
      console.error(`Error toggling detection: ${error}`);
      toast.error(`Failed to ${newStatus ? 'start' : 'stop'} detection. Please try again.`, {
        position: 'bottom-right',
        duration: 4000,
      });
    }
  };

  const updateSpeedCalibration = async (id: string) => {
    setSpeedCalibrationId(id);
    try {
      const response = await fetch(`${getEndpoint('LIVE_DETECTION')}/update?speed_calibration_id=${encodeURIComponent(id)}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to update speed calibration');
      }

      await fetchDetectionStatus();
    } catch (error) {
      console.error('Error updating speed calibration:', error);
      setSpeedCalibrationId(null);
      toast.error('Failed to update speed calibration. Please try again.', {
        position: 'bottom-right',
        duration: 4000,
      });
    }
  };

  const updateSelectedCamera = async (id: string) => {
    setSelectedCamera(id);
    try {
      const response = await fetch(`${getEndpoint('LIVE_DETECTION')}/update?camera_id=${encodeURIComponent(id)}`, {
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
    speedCalibrationId,
    calibrationIds,
    availableCameras,
    selectedCamera,
    toggleDetection,
    updateSpeedCalibration,
    updateSelectedCamera,
  };
}