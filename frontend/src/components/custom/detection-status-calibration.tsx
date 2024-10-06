// © 2024 Wyler Zahm. All rights reserved.

import React from "react";

import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel } from "@/components/ui/dropdown-menu";

import { useCalibrationStatusService } from '@/services/detectionStatusCalibrationService';
import { useCameraCalibrationService } from '@/services/cameraCalibrationService';

export function DetectionStatusCalibrationToggle() {
  const { 
    isDetectionOn, 
    cameraCalibrationId,
    availableCameras,
    selectedCamera,
    processingVideo,
    toggleCalibrationMode, 
    toggleDetectionCalibration,
    updateCameraCalibration,
    updateSelectedCamera,
  } = useCalibrationStatusService();

  const { calibrations } = useCameraCalibrationService();

  return (
    <div className="flex items-center gap-4">
      <span className="text-lg font-bold" style={{ color: isDetectionOn ? 'black' : 'red' }}>
        Calibration Status: {isDetectionOn ? 'ON' : 'OFF'}
      </span>
      
      {/* Camera Selection Dropdown */}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" size="sm" disabled={processingVideo || isDetectionOn}>
            {selectedCamera !== null 
              ? availableCameras.find(cam => cam.id === selectedCamera)?.name || 'Unknown Camera'
              : 'Select Camera...'}
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuLabel>Select Camera</DropdownMenuLabel>
          {availableCameras.length > 0 ? (
            availableCameras.map((camera) => (
              <DropdownMenuItem key={camera.id} onClick={() => updateSelectedCamera(camera.id)}>
                {camera.name}
              </DropdownMenuItem>
            ))
          ) : (
            <DropdownMenuItem disabled>No cameras available</DropdownMenuItem>
          )}
        </DropdownMenuContent>
      </DropdownMenu>

      <DropdownMenu>
        <DropdownMenuTrigger asChild disabled={processingVideo || isDetectionOn}>
          <Button variant="outline" size="sm">
            {cameraCalibrationId 
              ? calibrations.find(cal => cal.id.toString() === cameraCalibrationId)?.camera_name || 'Unknown Calibration'
              : 'Select Camera Calibration...'}
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuLabel>Camera Calibration</DropdownMenuLabel>
          {calibrations.length > 0 ? (
            calibrations.map((calibration) => (
              <DropdownMenuItem key={calibration.id} onClick={() => updateCameraCalibration(calibration.id.toString())}>
                {calibration.camera_name}
              </DropdownMenuItem>
            ))
          ) : (
            <DropdownMenuItem disabled>No calibrations available</DropdownMenuItem>
          )}
        </DropdownMenuContent>
      </DropdownMenu>

      <Button
        onClick={() => toggleDetectionCalibration()}
        variant={isDetectionOn ? "destructive" : "default"}
        disabled={processingVideo || (!isDetectionOn && (!cameraCalibrationId || selectedCamera === null))}
      >
        {isDetectionOn ? 'Stop Calibration' : 'Start Calibration'}
      </Button>
    </div>
  );
}