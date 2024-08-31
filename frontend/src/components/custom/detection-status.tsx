import React from "react";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel } from "@/components/ui/dropdown-menu";
import { useDetectionStatusService } from '@/services/detectionStatusService';

export function DetectionStatusToggle() {
  const { 
    isDetectionOn, 
    speedCalibrationId, 
    calibrations,
    availableCameras,
    selectedCamera,
    processingVideo,
    toggleDetection, 
    updateSpeedCalibration,
    updateSelectedCamera,
    fetchCalibrationIds,
  } = useDetectionStatusService();

  React.useEffect(() => {
    fetchCalibrationIds();
  }, [fetchCalibrationIds]);

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
    </div>
  );
}