import { useState } from 'react';
import { getEndpoint } from '@/api/endpoints';
import toast from 'react-hot-toast';
import { useSharedDetectionStatusService } from './sharedDetectionStatusService';

export function useCalibrationStatusService() {
  const shared = useSharedDetectionStatusService();
  const [cameraCalibrationId, setCameraCalibrationId] = useState<string | null>(null);

  const toggleCalibrationMode = async (mode: boolean) => {
    try {
      const response = await fetch(`${getEndpoint('LIVE_DETECTION')}/calibration_mode?is_calibrating=${encodeURIComponent(mode)}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      if (!response.ok) {
        throw new Error('Failed to update calibration mode');
      }

      const data = await response.json();
      await shared.fetchDetectionStatus();
      
      toast.success(`Calibration mode ${mode ? 'enabled' : 'disabled'}`, {
        position: 'bottom-right',
        duration: 3000,
      });

      return mode ? data.speed_calibration_id : null;
    } catch (error) {
      console.error('Error updating calibration mode:', error);
      toast.error('Failed to update calibration mode. Please try again.', {
        position: 'bottom-right',
        duration: 4000,
      });
      throw error;
    }
  };

  const toggleDetectionCalibration = async () => {
    if (shared.processingVideo) {
      toast.error('Cannot toggle detection while video is processing.', {
        position: 'bottom-right',
        duration: 4000,
      });
      return;
    }

    const newStatus = !shared.isDetectionOn;

    if (newStatus && (!cameraCalibrationId || shared.selectedCamera === null)) {
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
  
  const submitSpeedCalibration = async (speedCalibrationId: number, calibrationName: string) => {
    try {
      await updateSpeedCalibration(speedCalibrationId.toString(), { name: calibrationName });
      const response = await fetch(`${getEndpoint('SPEED_CALIBRATIONS_SUBMIT').replace('{calibration_id}', speedCalibrationId.toString())}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to submit speed calibration');
      }

      toast.success('Speed calibration submitted successfully', {
        position: 'bottom-right',
        duration: 3000,
      });
      return true;
    }
    catch (error) {
      console.error('Error submitting speed calibration:', error);
      toast.error('Failed to submit speed calibration. Please try again.', {
        position: 'bottom-right',
        duration: 4000,
      });
      return false;
    }
  };

  const updateSpeedCalibration = async (id: string, calibrationData: object) => {
    try {
      const response = await fetch(`${getEndpoint('SPEED_CALIBRATIONS_UPDATE').replace('{calibration_id}', id)}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(calibrationData),
      });

      if (!response.ok) {
        throw new Error('Failed to update speed calibration');
      }

      toast.success('Speed calibration updated successfully', {
        position: 'bottom-right',
        duration: 3000,
      });
      return true;
    } catch (error) {
      console.error('Error updating speed calibration:', error);
      toast.error('Failed to update speed calibration. Please try again.', {
        position: 'bottom-right',
        duration: 4000,
      });
      return false;
    }
  }

  const updateCameraCalibration = async (id: string) => {
    setCameraCalibrationId(id);
    try {
      const response = await fetch(`${getEndpoint('LIVE_DETECTION')}/update?camera_calibration_id=${encodeURIComponent(id)}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to update camera calibration');
      }

      await shared.fetchDetectionStatus();
    } catch (error) {
      console.error('Error updating camera calibration:', error);
      setCameraCalibrationId(null);
      toast.error('Failed to update camera calibration. Please try again.', {
        position: 'bottom-right',
        duration: 4000,
      });
    }
  };

  const uploadCalibrationVideo = async (calibrationId: number, videoFile: File) => {
    try {
      const formData = new FormData();
      formData.append('video', videoFile);

      const response = await fetch(getEndpoint('UPLOAD_SPEED_CALIBRATION_VIDEO').replace('{calibration_id}', calibrationId.toString()), {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        if (response.status === 400) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Bad request');
        }
        throw new Error('Failed to upload calibration video');
      }

      toast.success('Calibration video uploaded successfully', {
        position: 'bottom-right',
        duration: 3000,
      });
    } catch (err) {
      if (err instanceof Error) {
        toast.error(`Failed to upload calibration video: ${err.message}`, {
          position: 'bottom-right',
          duration: 4000,
        });
      }
      console.error('Error uploading calibration video:', err);
      throw err;
    }
  };

  return {
    ...shared,
    cameraCalibrationId,
    toggleCalibrationMode,
    updateCameraCalibration,
    uploadCalibrationVideo,
    submitSpeedCalibration,
    updateSpeedCalibration,
    toggleDetectionCalibration
  };
}