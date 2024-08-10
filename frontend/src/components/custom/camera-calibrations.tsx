import React from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { TrashIcon } from "@/components/custom/icons";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertTriangle } from "lucide-react";
import { useCameraCalibrationService } from '@/services/cameraCalibrationService';

export function CameraCalibrations() {
  const {
    calibrations,
    loading,
    error,
    addCalibration,
    updateCalibration,
    deleteCalibration
  } = useCameraCalibrationService();

  const handleAddCalibration = () => {
    // Implement add calibration logic
    console.log("Add camera calibration");
  };

  const handleUpdateCalibration = (id: number) => {
    // Implement update calibration logic
    console.log("Update camera calibration", id);
  };

  const handleDeleteCalibration = async (id: number) => {
    try {
      await deleteCalibration(id);
    } catch (err) {
      console.error("Failed to delete camera calibration:", err);
      // Optionally, show an error message to the user
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Camera Calibrations</CardTitle>
        <CardDescription>Manage your camera calibration settings</CardDescription>
        <div className="flex justify-end">
          <Button onClick={handleAddCalibration}>Add Calibration</Button>
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div>Loading...</div>
        ) : error ? (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {calibrations.map((calibration) => (
              <Card key={calibration.id}>
                <CardHeader>
                  <CardTitle>{calibration.camera_name}</CardTitle>
                  <CardDescription>{calibration.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <img
                    //src={calibration.imageUrl}
                    alt={`Calibration for ${calibration.camera_name}`}
                    className="rounded-lg object-cover w-full aspect-video"
                  />
                  <p className="mt-2 text-sm text-gray-500">
                    Calibration Date: {new Date(calibration.calibration_date).toLocaleDateString()}
                  </p>
                </CardContent>
                <CardFooter className="flex items-center justify-between">
                  <Button variant="outline" size="sm" onClick={() => handleUpdateCalibration(calibration.id)}>
                    Update
                  </Button>
                  <Button variant="ghost" size="icon" className="rounded-full" onClick={() => handleDeleteCalibration(calibration.id)}>
                    <TrashIcon className="w-5 h-5" />
                    <span className="sr-only">Delete</span>
                  </Button>
                </CardFooter>
              </Card>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}