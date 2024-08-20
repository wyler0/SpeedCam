import React from "react";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel } from "@/components/ui/dropdown-menu";
import { useDetectionStatusService } from '@/services/detectionStatusService';

export function DetectionStatusToggle() {
  const { 
    isDetectionOn, 
    speedCalibrationId, 
    calibrationIds,
    availableCameras,
    selectedCamera,
    toggleDetection, 
    updateSpeedCalibration,
    updateSelectedCamera
  } = useDetectionStatusService();

  return (
    <div className="flex items-center gap-4">
      <span className="text-lg font-bold" style={{ color: isDetectionOn ? 'black' : 'red' }}>
        {isDetectionOn ? 'Detection Status: ON' : 'Detection Status: OFF'}
      </span>
      
      {/* Camera Selection Dropdown */}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" size="sm">
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
        <DropdownMenuTrigger asChild>
          <Button variant="outline" size="sm">
            {speedCalibrationId ? speedCalibrationId : 'Select Calibration...'}
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuLabel>Select Calibration</DropdownMenuLabel>
          {calibrationIds.length > 0 ? (
            calibrationIds.map((id) => (
              <DropdownMenuItem key={id} onClick={() => updateSpeedCalibration(id)}>
                {id}
              </DropdownMenuItem>
            ))
          ) : (
            <DropdownMenuItem disabled>No calibrations available</DropdownMenuItem>
          )}
        </DropdownMenuContent>
      </DropdownMenu>

      <Button 
        onClick={toggleDetection} 
        className={`px-4 py-2 rounded ${isDetectionOn ? 'bg-red-500' : 'bg-green-500'} text-white`}
        disabled={!isDetectionOn && (!speedCalibrationId || selectedCamera === null)}
      >
        {isDetectionOn ? 'Turn Off' : 'Turn On'}
      </Button>
    </div>
  );
}