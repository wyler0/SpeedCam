import { useState, useEffect, useCallback, useRef } from 'react';
import { getEndpoint } from '@/api/endpoints';

export enum Direction {
  leftToRight = 0,
  rightToLeft = 1,
}

export interface Detection {
  id: number;
  detection_date: string;
  direction: Direction | null;
  thumbnail_path: string | null;
  pixel_speed_estimate: number | null;
  real_world_speed_estimate: number | null;
  real_world_speed: number | null;
  speed_calibration_id: number | null;
  //optical_flow_path: string | null;
  error: string | null;
}

export const PredefinedFilters = {
  TODAY: 'TODAY',
  LAST_7_DAYS: 'LAST_7_DAYS',
  LAST_30_DAYS: 'LAST_30_DAYS',
  // SPEEDING: 'SPEEDING',
  // KNOWN_SPEED: 'KNOWN_SPEED',
} as const;

export type PredefinedFilterType = typeof PredefinedFilters[keyof typeof PredefinedFilters];

export interface VehicleDetectionFilters {
  speed_calibration_id?: number;
  startDate?: string;
  endDate?: string;
  minSpeed?: number;
  maxSpeed?: number;
  direction?: string;
  knownSpeedOnly?: boolean;
  predefinedFilter?: PredefinedFilterType;
}

export function useVehicleDetectionService(initialFilters: VehicleDetectionFilters = {}, pollingInterval = 5000, speedLimit: number) {
  const [filters, setFilters] = useState<VehicleDetectionFilters>(initialFilters);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollingTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const fetchDetections = useCallback(async () => {
    if (!filters.speed_calibration_id) {
      setError("A speed calibration ID is required");
      return;
    }

    setLoading(true);
    setError(null);
    const queryParams = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== null && value !== undefined) {
        queryParams.append(key, value.toString());
      }
    });

    try {
      const response = await fetch(`${getEndpoint('VEHICLE_DETECTIONS')}?${queryParams}`);
      if (!response.ok) {
        throw new Error('Failed to fetch vehicle detections');
      }
      const data: Detection[] = await response.json();
      setDetections(prevDetections => {
        // Only update if the data has changed
        if (JSON.stringify(prevDetections) !== JSON.stringify(data)) {
          return data;
        }
        return prevDetections;
      });
    } catch (error) {
      console.error('Error fetching vehicle detections:', error);
      setError(error instanceof Error ? error.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  }, [filters]);

  const startPolling = useCallback(() => {
    if (pollingTimeoutRef.current) {
      clearTimeout(pollingTimeoutRef.current);
    }
    
    const poll = () => {
      fetchDetections().then(() => {
        pollingTimeoutRef.current = setTimeout(poll, pollingInterval);
      });
    };

    poll();
  }, [fetchDetections, pollingInterval]);

  useEffect(() => {
    if (filters.speed_calibration_id) {
      fetchDetections();
      startPolling();
    } else {
      setDetections([]);
      if (pollingTimeoutRef.current) {
        clearTimeout(pollingTimeoutRef.current);
      }
    }

    return () => {
      if (pollingTimeoutRef.current) {
        clearTimeout(pollingTimeoutRef.current);
      }
    };
  }, [filters, fetchDetections, startPolling]);

  const updateFilters = useCallback((newFilters: Partial<VehicleDetectionFilters>) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
  }, []);

  // New effect to handle filter changes
  useEffect(() => {
    if (filters.speed_calibration_id) {
      fetchDetections();
    }
  }, [filters, fetchDetections]);

  const getStatistics = useCallback(() => {
    const vehiclesDetected = detections.filter(d => d.real_world_speed_estimate !== null).length;
    const averageSpeed = detections.reduce((sum, d) => d.real_world_speed_estimate !== null ? sum + d.real_world_speed_estimate : sum, 0) / detections.filter(d => d.real_world_speed_estimate !== null).length || 0;
    const speedingViolations = detections.filter(d => d.real_world_speed_estimate !== null && d.real_world_speed_estimate > speedLimit).length;

    return {
      vehiclesDetected,
      averageSpeed,
      speedingViolations,
    };
  }, [detections, speedLimit]);

  const updateDetection = async (id: number, updates: Partial<Detection>) => {
    try {
      const response = await fetch(`${getEndpoint('VEHICLE_DETECTIONS')}/${id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updates),
      });

      if (!response.ok) {
        throw new Error('Failed to update detection');
      }

      await fetchDetections();
    } catch (error) {
      console.error('Error updating detection:', error);
      throw error;
    }
  };

  const deleteDetection = async (id: number) => {
    try {
      const response = await fetch(`${getEndpoint('VEHICLE_DETECTIONS')}/${id}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error('Failed to delete detection');
      }

      await fetchDetections();
    } catch (error) {
      console.error('Error deleting detection:', error);
      throw error;
    }
  };

  return { 
    detections, 
    loading, 
    error, 
    filters, 
    updateFilters, 
    getStatistics, 
    PredefinedFilters, 
    updateDetection, 
    deleteDetection, 
    startPolling 
  };
}