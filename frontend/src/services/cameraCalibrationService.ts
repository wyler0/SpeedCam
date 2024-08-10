import { useState, useEffect, useCallback } from 'react';
import { getEndpoint } from '@/api/endpoints';

export interface CameraCalibration {
  id: number;
  camera_name: string;
  calibration_date: string;
  description: string;
}

export function useCameraCalibrationService() {
  const [calibrations, setCalibrations] = useState<CameraCalibration[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchCalibrations = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(getEndpoint('CAMERA_CALIBRATIONS'));
      if (!response.ok) {
        throw new Error('Failed to fetch camera calibrations');
      }
      const data: CameraCalibration[] = await response.json();
      setCalibrations(data);
    } catch (err) {
      console.error('Error fetching camera calibrations:', err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchCalibrations();
  }, [fetchCalibrations]);

  const addCalibration = useCallback(async (newCalibration: Omit<CameraCalibration, 'id'>) => {
    try {
      const response = await fetch(getEndpoint('CAMERA_CALIBRATIONS'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newCalibration),
      });
      if (!response.ok) {
        throw new Error('Failed to add camera calibration');
      }
      await fetchCalibrations();
    } catch (err) {
      console.error('Error adding camera calibration:', err);
      throw err;
    }
  }, [fetchCalibrations]);

  const updateCalibration = useCallback(async (id: number, updatedCalibration: Partial<CameraCalibration>) => {
    try {
      const response = await fetch(`${getEndpoint('CAMERA_CALIBRATIONS')}/${id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updatedCalibration),
      });
      if (!response.ok) {
        throw new Error('Failed to update camera calibration');
      }
      await fetchCalibrations();
    } catch (err) {
      console.error('Error updating camera calibration:', err);
      throw err;
    }
  }, [fetchCalibrations]);

  const deleteCalibration = useCallback(async (id: number) => {
    try {
      const response = await fetch(`${getEndpoint('CAMERA_CALIBRATIONS')}/${id}`, {
        method: 'DELETE',
      });
      if (!response.ok) {
        throw new Error('Failed to delete camera calibration');
      }
      await fetchCalibrations();
    } catch (err) {
      console.error('Error deleting camera calibration:', err);
      throw err;
    }
  }, [fetchCalibrations]);

  return {
    calibrations,
    loading,
    error,
    fetchCalibrations,
    addCalibration,
    updateCalibration,
    deleteCalibration,
  };
}