import { useState, useEffect, useCallback } from 'react';
import { getEndpoint } from '@/api/endpoints';
import { toast } from 'react-hot-toast';

export interface CameraCalibration {
  id: number;
  camera_name: string;
  calibration_date: string;
  rows: number;
  cols: number;
  description: string;
  images_path: string;
  thumbnail?: string;
  images?: string[];
  valid: boolean;
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
      toast.success('Camera calibration deleted successfully');
      await fetchCalibrations();
    } catch (err) {
      console.error('Error deleting camera calibration:', err);
      toast.error('Failed to delete camera calibration');
      throw err;
    }
  }, [fetchCalibrations]);

  const fetchThumbnail = useCallback(async (calibration: CameraCalibration): Promise<string | undefined> => {
    try {
      const response = await fetch(`${getEndpoint('CAMERA_CALIBRATIONS')}/${calibration.id}/images`);
      if (!response.ok) {
        throw new Error('Failed to fetch image list');
      }
      const { images } = await response.json();
      if (images.length === 0) {
        throw new Error('No images found for this calibration');
      }
      
      const thumbnailUrl = images.find((image: string) => image.includes("grid")); // Get the first image with "grid" in the name
      const thumbnailResponse = await fetch(thumbnailUrl);
      if (!thumbnailResponse.ok) {
        throw new Error('Failed to fetch thumbnail image');
      }
      const blob = await thumbnailResponse.blob();
      return URL.createObjectURL(blob);
    } catch (err) {
      console.error('Error fetching thumbnail:', err);
      return undefined;
    }
  }, []);
  
  const fetchAllImages = useCallback(async (calibration: CameraCalibration) => {
    try {
      const response = await fetch(`${getEndpoint('CAMERA_CALIBRATIONS')}/${calibration.id}/images`);
      if (!response.ok) {
        throw new Error('Failed to fetch image list');
      }
      const { images } = await response.json();
      const imagePromises = images.filter((imageUrl: string) => imageUrl.includes("grid")).map(async (imageUrl: string) => {
        const imageResponse = await fetch(imageUrl);
        if (!imageResponse.ok) {
          throw new Error(`Failed to fetch image: ${imageUrl}`);
        }
        const blob = await imageResponse.blob();
        return URL.createObjectURL(blob);
      });
      return await Promise.all(imagePromises);
    } catch (err) {
      console.error('Error fetching all images:', err);
      return [];
    }
  }, []);

  return {
    calibrations,
    loading,
    error,
    fetchCalibrations,
    addCalibration,
    updateCalibration,
    deleteCalibration,
    fetchThumbnail,
    fetchAllImages,
  };
}