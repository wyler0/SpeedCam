import { useState, useEffect, useCallback } from 'react';
import { getEndpoint } from '@/api/endpoints';
import toast from 'react-hot-toast';

interface DetectionStatus {
  running: boolean;
  speed_calibration_id: string | null;
}

export function useDetectionStatusService() {
  const [isDetectionOn, setIsDetectionOn] = useState(false);
  const [speedCalibrationId, setSpeedCalibrationId] = useState<string | null>(null);
  const [calibrationIds, setCalibrationIds] = useState<string[]>([]);

  const fetchDetectionStatus = useCallback(async () => {
    try {
      const response = await fetch(`${getEndpoint('LIVE_DETECTION')}/status`);
      const data: DetectionStatus = await response.json();
      setIsDetectionOn(data.running);
      setSpeedCalibrationId(data.speed_calibration_id);
    } catch (error) {
      console.error("Error fetching detection status:", error);
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

    const intervalId = setInterval(fetchDetectionStatus, 5000);

    return () => clearInterval(intervalId);
  }, [fetchDetectionStatus, fetchCalibrationIds]);

  const toggleDetection = async () => {
    const newStatus = !isDetectionOn;

    if (newStatus && !speedCalibrationId) {
      toast.error('Please select a calibration before starting detection.', {
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
      const response = await fetch(`${getEndpoint('LIVE_DETECTION')}/update?speedCalibrationId=${id}`, {
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

  return {
    isDetectionOn,
    speedCalibrationId,
    calibrationIds,
    toggleDetection,
    updateSpeedCalibration,
  };
}