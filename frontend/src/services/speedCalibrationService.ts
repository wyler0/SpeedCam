import { useState, useEffect, useCallback } from 'react';
import { getEndpoint } from '@/api/endpoints';

export interface SpeedCalibration {
  id: number;
  name: string;
  description: string;
}

export function useSpeedCalibrationService() {
  const [calibrations, setCalibrations] = useState<SpeedCalibration[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchCalibrations = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(getEndpoint('SPEED_CALIBRATIONS'));
      if (!response.ok) {
        throw new Error('Failed to fetch calibrations');
      }
      const data: SpeedCalibration[] = await response.json();
      setCalibrations(data);
    } catch (err) {
      console.error('Error fetching calibrations:', err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchCalibrations();
  }, [fetchCalibrations]);

  const addCalibration = useCallback(async (newCalibration: Omit<SpeedCalibration, 'id'>) => {
    try {
      const response = await fetch(getEndpoint('SPEED_CALIBRATIONS'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newCalibration),
      });
      if (!response.ok) {
        throw new Error('Failed to add calibration');
      }
      await fetchCalibrations();
    } catch (err) {
      console.error('Error adding calibration:', err);
      throw err;
    }
  }, [fetchCalibrations]);

  const updateCalibration = useCallback(async (id: number, updatedCalibration: Partial<SpeedCalibration>) => {
    try {
      const response = await fetch(`${getEndpoint('SPEED_CALIBRATIONS')}/${id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updatedCalibration),
      });
      if (!response.ok) {
        throw new Error('Failed to update calibration');
      }
      await fetchCalibrations();
    } catch (err) {
      console.error('Error updating calibration:', err);
      throw err;
    }
  }, [fetchCalibrations]);

  const deleteCalibration = useCallback(async (id: number) => {
    try {
      const response = await fetch(`${getEndpoint('SPEED_CALIBRATIONS')}/${id}`, {
        method: 'DELETE',
      });
      if (!response.ok) {
        throw new Error('Failed to delete calibration');
      }
      await fetchCalibrations();
    } catch (err) {
      console.error('Error deleting calibration:', err);
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