import React from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { TrashIcon } from "@/components/custom/icons";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertTriangle } from "lucide-react";
import { useSpeedCalibrationService } from '@/services/speedCalibrationService';

export function SpeedCalibrations() {
  const {
    calibrations,
    loading,
    error,
    addCalibration,
    updateCalibration,
    deleteCalibration
  } = useSpeedCalibrationService();

  const handleAddCalibration = () => {
    // Implement add calibration logic
    console.log("Add calibration");
  };

  const handleUpdateCalibration = (id: number) => {
    // Implement update calibration logic
    console.log("Update calibration", id);
  };

  const handleDeleteCalibration = async (id: number) => {
    try {
      await deleteCalibration(id);
    } catch (err) {
      console.error("Failed to delete calibration:", err);
      // Optionally, show an error message to the user
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Speed Calibrations</CardTitle>
        <CardDescription>Manage your speed calibration settings</CardDescription>
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
                  <CardTitle>{calibration.title}</CardTitle>
                  <CardDescription>{calibration.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <video className="rounded-lg object-cover w-full aspect-video" controls>
                    <source src={calibration.videoUrl} type="video/mp4" />
                  </video>
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