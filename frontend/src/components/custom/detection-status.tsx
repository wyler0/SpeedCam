// © 2024 Wyler Zahm. All rights reserved.

import React from "react";
import { toast } from "react-hot-toast";

import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel } from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { useState, useEffect } from "react";
import { getEndpoint } from "@/api/endpoints";

import { CameraInfo } from "@/services/sharedDetectionStatusService";

interface DetectionStatusToggleProps {
  isDetectionOn: boolean;
  speedCalibrationId: string | null;
  calibrations: Array<{ id: number; name: string; valid: boolean }>;
  availableCameras: CameraInfo[];
  selectedCamera: number | null;
  processingVideo: boolean;
  toggleDetection: () => void;
  updateSpeedCalibration: (id: string) => void;
  updateSelectedCamera: (id: string) => Promise<void>;
  initialSpeedLimit: number;
  onSpeedLimitChange: (newSpeedLimit: number) => void;
}

export function DetectionStatusToggle({
  isDetectionOn,
  speedCalibrationId,
  calibrations,
  availableCameras,
  selectedCamera,
  processingVideo,
  toggleDetection,
  updateSpeedCalibration,
  updateSelectedCamera,
  initialSpeedLimit,
  onSpeedLimitChange,
}: DetectionStatusToggleProps) {
  const [speedLimit, setSpeedLimit] = useState(initialSpeedLimit);
  const [isEditingSpeedLimit, setIsEditingSpeedLimit] = useState(false);

  useEffect(() => {
    fetchSpeedLimit();
  }, []);

  const fetchSpeedLimit = async () => {
    try {
      const response = await fetch(getEndpoint('SPEED_LIMIT'));
      if (!response.ok) {
        throw new Error('Failed to fetch speed limit');
      }
      const data = await response.json();
      setSpeedLimit(data.speed_limit);
      onSpeedLimitChange(data.speed_limit);
    } catch (error) {
      console.error("Failed to fetch speed limit:", error);
    }
  };

  const updateSpeedLimit = async () => {
    try {
      const response = await fetch(`${getEndpoint('SPEED_LIMIT')}?speed_limit=${speedLimit}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      if (!response.ok) {
        toast.error('Failed to update speed limit');
        throw new Error('Failed to update speed limit');
      }
      toast.success('Speed limit updated successfully');
      setIsEditingSpeedLimit(false);
      onSpeedLimitChange(speedLimit);
    } catch (error) {
      console.error("Failed to update speed limit:", error);
    }
  };

  return (
    <div className="flex items-center gap-4">
      <span className="text-lg font-bold" style={{ color: isDetectionOn ? 'black' : 'red' }}>
        Detection Status: {isDetectionOn ? 'ON' : 'OFF'}
      </span>
      
      {/* Camera Selection Dropdown */}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" size="sm">
            {selectedCamera !== null 
              ? availableCameras.find(cam => cam.id === selectedCamera.toString())?.name || 'Unknown Camera'
              : 'Select Camera...'}
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuLabel>Select Camera</DropdownMenuLabel>
          {availableCameras.length > 0 ? (
            availableCameras.map((camera) => (
              <DropdownMenuItem key={camera.id} onClick={() => updateSelectedCamera(camera.id.toString())}>
                {camera.name}
              </DropdownMenuItem>
            ))
          ) : (
            <DropdownMenuItem disabled>No cameras available</DropdownMenuItem>
          )}
        </DropdownMenuContent>
      </DropdownMenu>

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" size="sm">
            {speedCalibrationId ? speedCalibrationId : 'Select Speed Calibration...'}
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuLabel>Speed Calibration</DropdownMenuLabel>
          {calibrations.filter(calibration => calibration.valid).length > 0 ? (
            calibrations.filter(calibration => calibration.valid).map((calibration) => (
              <DropdownMenuItem key={calibration.id} onClick={() => updateSpeedCalibration(calibration.id.toString())}>
                {calibration.name}
              </DropdownMenuItem>
            ))
          ) : (
            <DropdownMenuItem disabled>No valid speed calibrations available</DropdownMenuItem>
          )}
        </DropdownMenuContent>
      </DropdownMenu>

      <Button 
        onClick={toggleDetection} 
        className={`px-4 py-2 rounded ${isDetectionOn ? 'bg-red-500' : 'bg-green-500'} text-white`}
        disabled={processingVideo || (!isDetectionOn && (!speedCalibrationId || selectedCamera === null))}
      >
        {isDetectionOn ? 'Turn Off' : 'Turn On'}
      </Button>

      {/* Speed Limit Input */}
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium">Speed Limit (mph):</span>
        {isEditingSpeedLimit ? (
          <Input
            type="number"
            value={speedLimit}
            onChange={(e) => setSpeedLimit(Number(e.target.value))}
            className="w-20"
            onBlur={updateSpeedLimit}
            onKeyPress={(e) => e.key === 'Enter' && updateSpeedLimit()}
          />
        ) : (
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsEditingSpeedLimit(true)}
          >
            {speedLimit} mph
          </Button>
        )}
      </div>
    </div>
  );
}