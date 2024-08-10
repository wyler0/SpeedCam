import { useState, useEffect, useCallback } from 'react';
import { getEndpoint } from '@/api/endpoints';

export enum Direction {
  leftToRight = 0,
  rightToLeft = 1,
}

export interface Detection {
  id: number;
  detection_date: string;
  video_clip_path: string | null;
  speed_calibration_id: number | null;
  estimated_speed: number | null;
  true_speed: number | null;
  direction: Direction | null;
  confidence: number | null;
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
  speedCalibrationId?: number;
  startDate?: string;
  endDate?: string;
  minSpeed?: number;
  maxSpeed?: number;
  direction?: string;
  knownSpeedOnly?: boolean;
  predefinedFilter?: PredefinedFilterType;
}

export function useVehicleDetectionService(initialFilters: VehicleDetectionFilters = {}) {
  const [filters, setFilters] = useState<VehicleDetectionFilters>(initialFilters);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchDetections = useCallback(async () => {
    if (!filters.speedCalibrationId) {
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
      setDetections(data);
    } catch (error) {
      console.error('Error fetching vehicle detections:', error);
      setError(error instanceof Error ? error.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  }, [filters]);

  useEffect(() => {
    if (filters.speedCalibrationId) {
      fetchDetections();
    }
  }, [fetchDetections]);

  const updateFilters = useCallback((newFilters: Partial<VehicleDetectionFilters>) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
  }, []);

  const getStatistics = () => {
    const vehiclesDetected = detections.length;
    const averageSpeed = detections.reduce((sum, d) => d.estimated_speed !== null ? sum + d.estimated_speed : sum, 0) / detections.filter(d => d.estimated_speed !== null).length || 0;
    const speedingViolations = detections.filter(d => d.estimated_speed !== null && d.estimated_speed > 30).length;

    return {
      vehiclesDetected,
      averageSpeed,
      speedingViolations,
    };
  };

  return { detections, loading, error, filters, updateFilters, getStatistics, PredefinedFilters };
}