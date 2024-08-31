import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { TrashIcon } from "@/components/custom/icons";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertTriangle } from "lucide-react";
import { useSpeedCalibrationService } from '@/services/speedCalibrationService';
import { SpeedCalibrationAdd } from "./speed-calibration-add";
import { useCalibrationStatusService } from '@/services/detectionStatusCalibrationService';
import toast from 'react-hot-toast';

export function SpeedCalibrations() {
  const {
    calibrations,
    loading,
    error,
    addCalibration,
    updateCalibration,
    deleteCalibration
  } = useSpeedCalibrationService();

  const { toggleCalibrationMode, isDetectionOn } = useCalibrationStatusService();

  const [isAddCalibrationOpen, setIsAddCalibrationOpen] = useState(false);
  const [isWarningDialogOpen, setIsWarningDialogOpen] = useState(false);
  const [newSpeedCalibrationId, setNewSpeedCalibrationId] = useState<number | null>(null);

  const handleAddCalibration = () => {
    if (isDetectionOn) {
      setIsWarningDialogOpen(true);
    } else {
      openAddCalibrationModal();
    }
  };

  const openAddCalibrationModal = async () => {
    try {
      const speed_calibration_id = await toggleCalibrationMode(true);
      if (speed_calibration_id) {
        setNewSpeedCalibrationId(speed_calibration_id);
        setIsAddCalibrationOpen(true);
        setIsWarningDialogOpen(false);
      } else {
        await handleCloseAddCalibration();
        toast.error('Failed to get new speed calibration ID');
      }
    } catch (error) {
      console.error('Error starting calibration mode:', error);
      toast.error('Failed to start calibration mode. Please try again.');
    }
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
      toast.error('Failed to delete calibration. Please try again.');
    }
  };

  const handleCloseAddCalibration = async () => {
    try {
      await toggleCalibrationMode(false);
      setIsAddCalibrationOpen(false);
      setNewSpeedCalibrationId(null);
    } catch (error) {
      console.error('Error disabling calibration mode:', error);
      toast.error('Failed to disable calibration mode. Please try again.');
    }
  };

  return (
    <>
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

      <Dialog open={isWarningDialogOpen} onOpenChange={setIsWarningDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Warning</DialogTitle>
            <DialogDescription>
              Active speed camera detection will be stopped if a calibration is added. Do you want to proceed?
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsWarningDialogOpen(false)}>Cancel</Button>
            <Button onClick={openAddCalibrationModal}>Proceed</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {isAddCalibrationOpen && newSpeedCalibrationId && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <SpeedCalibrationAdd 
            onClose={handleCloseAddCalibration}
            onCalibrationAdded={() => {
              // Implement logic to refresh calibrations
              setIsAddCalibrationOpen(false);
              setNewSpeedCalibrationId(null);
            }}
            speedCalibrationId={newSpeedCalibrationId}
          />
        </div>
      )}
    </>
  );
}

/*
STOPPED HERE.

- Fixing the ned for a speed calibration id, which should be populated after the calibration is created, but the code wants it to be populated before (see 119 above)
- Seems the latest change makes the calibration status change rapidly, need to fix that.
*/
