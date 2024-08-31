import { useState, useEffect, useCallback } from 'react';
import { getEndpoint } from '@/api/endpoints';
import toast from 'react-hot-toast';
import { useSharedDetectionStatusService } from './sharedDetectionStatusService';

export interface SpeedCalibration {
  name: string;
  description: string;
  camera_calibration_id: number;
  calibration_date: string;
  valid: boolean;
  left_to_right_constant: number;
  right_to_left_constant: number;
  id: number;
  vehicle_detections: number;
}


export function useDetectionStatusService() {
  const shared = useSharedDetectionStatusService();
  const [speedCalibrationId, setSpeedCalibrationId] = useState<string | null>(null);
  const [calibrations, setCalibrations] = useState<SpeedCalibration[]>([]);

  const fetchCalibrationIds = useCallback(async () => {
    try {
      const response = await fetch(getEndpoint('SPEED_CALIBRATIONS'), {
        method: 'GET',
      });
      const data = await response.json();
      if (Array.isArray(data)) {
        setCalibrations(data);
      } else {
        console.error("Unexpected data format for calibration_ids:", data);
      }
    } catch (error) {
      console.error("Error fetching calibration IDs:", error);
    }
  }, []);

  const toggleDetection = async () => {
    if (shared.processingVideo) {
      toast.error('Cannot toggle detection while video is processing.', {
        position: 'bottom-right',
        duration: 4000,
      });
      return;
    }

    const newStatus = !shared.isDetectionOn;

    if (newStatus && (!speedCalibrationId || shared.selectedCamera === null)) {
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
        body: JSON.stringify({ camera_source: shared.selectedCamera }),
      });

      if (!response.ok) {
        throw new Error(`Failed to ${endpoint} detection`);
      }

      await shared.fetchDetectionStatus();
      
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

      await shared.fetchDetectionStatus();
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
    ...shared,
    speedCalibrationId,
    calibrations,
    toggleDetection,
    updateSpeedCalibration,
    fetchCalibrationIds,
  };
}