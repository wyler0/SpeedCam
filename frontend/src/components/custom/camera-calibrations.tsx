// © 2024 Wyler Zahm. All rights reserved.

import React, { useState, useEffect, useCallback } from 'react';

import { AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { TrashIcon } from "@/components/custom/icons";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";

import { CameraCalibrationAdd } from './camera-calibration-add';

import { useCameraCalibrationService, CameraCalibration } from '@/services/cameraCalibrationService';

export function CameraCalibrations() {
  const [isAddCalibrationOpen, setIsAddCalibrationOpen] = useState(false);
  const [selectedCalibration, setSelectedCalibration] = useState<CameraCalibration | null>(null);
  const {
    calibrations,
    loading,
    error,
    deleteCalibration,
    fetchThumbnail,
    fetchAllImages,
    fetchCalibrations
  } = useCameraCalibrationService();

  const [calibrationsWithThumbnails, setCalibrationsWithThumbnails] = useState<(CameraCalibration & { thumbnail?: string | null })[]>([]);

  useEffect(() => {
    const fetchThumbnails = async () => {
      const updatedCalibrations = await Promise.all(
        calibrations.map(async (calibration) => {
          const thumbnail = await fetchThumbnail(calibration);
          return { ...calibration, thumbnail: thumbnail || undefined };
        })
      );
      setCalibrationsWithThumbnails(updatedCalibrations);
    };

    fetchThumbnails();
  }, [calibrations, fetchThumbnail]);

  const handleCalibrationAdded = useCallback(async () => {
    await fetchCalibrations();
  }, [fetchCalibrations]);

  const handleUpdateCalibration = (id: number) => {
    // Implement update calibration logic
    console.log("Update camera calibration", id);
  };

  const handleDeleteCalibration = async (id: number) => {
    if (window.confirm('Are you sure you want to delete this camera calibration?')) {
      try {
        await deleteCalibration(id);
      } catch (err) {
        console.error("Failed to delete camera calibration:", err);
        // Error is already handled in the service with a toast
      }
    }
  };

  const handleCalibrationClick = async (calibration: CameraCalibration) => {
    const images = await fetchAllImages(calibration);
    setSelectedCalibration({ ...calibration, images });
  };

  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle>Camera Calibrations</CardTitle>
          <CardDescription>Manage your camera calibration settings</CardDescription>
          <div className="flex justify-end">
            <Button onClick={() => setIsAddCalibrationOpen(true)}>Add Calibration</Button>
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
              {calibrationsWithThumbnails.map((calibration) => (
                <Card key={calibration.id} className="cursor-pointer" onClick={() => handleCalibrationClick(calibration)}>
                  <CardHeader className="flex flex-row items-center justify-between">
                    <CardTitle>{calibration.camera_name}</CardTitle>
                    <Button variant="ghost" size="icon" className="rounded-full" onClick={(e) => { e.stopPropagation(); handleDeleteCalibration(calibration.id); }}>
                      <TrashIcon className="w-5 h-5" />
                      <span className="sr-only">Delete</span>
                    </Button>
                  </CardHeader>
                  <CardContent>
                    <img
                      src={calibration.thumbnail || '/placeholder-image.jpg'}
                      alt={`Calibration for ${calibration.camera_name}`}
                      className="rounded-lg object-cover w-full aspect-video"
                    />
                    <p className="mt-2 text-sm text-gray-500">
                      Calibration Date: {new Date(calibration.calibration_date).toLocaleDateString()}
                    </p>
                  </CardContent>
                  <CardFooter className="flex items-center justify-between">
                    {/* <Button variant="outline" size="sm" onClick={(e) => { e.stopPropagation(); handleUpdateCalibration(calibration.id); }}>
                      Update
                    </Button> */}
                  </CardFooter>
                </Card>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {isAddCalibrationOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <CameraCalibrationAdd 
            onClose={() => setIsAddCalibrationOpen(false)} 
            onCalibrationAdded={handleCalibrationAdded}
          />
        </div>
      )}

      <Dialog open={!!selectedCalibration} onOpenChange={() => setSelectedCalibration(null)}>
        <DialogContent className="max-w-3xl">
          <DialogHeader>
            <DialogTitle>{selectedCalibration?.camera_name}</DialogTitle>
            <DialogDescription>Calibration Details</DialogDescription>
          </DialogHeader>
          {selectedCalibration && (
            <div className="mt-4">
              <p>Rows: {selectedCalibration.rows}</p>
              <p>Columns: {selectedCalibration.cols}</p>
              <p>Calibration Date: {new Date(selectedCalibration.calibration_date).toLocaleString()}</p>
              <p>Valid: {selectedCalibration.valid ? 'Yes' : 'No'}</p>
              <div className="mt-4 grid grid-cols-3 gap-4">
                {selectedCalibration.images?.map((image, index) => (
                  <img key={index} src={image} alt={`Calibration image ${index + 1}`} className="rounded-lg object-cover w-full aspect-video" />
                ))}
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}