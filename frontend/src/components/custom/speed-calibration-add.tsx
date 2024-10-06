// © 2024 Wyler Zahm. All rights reserved.

"use client"

import React, { useState, useEffect, useCallback } from "react"
import toast from 'react-hot-toast'

import { Button } from "@/components/ui/button"
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Separator } from "@/components/ui/separator"
import { TrashIcon, ChevronLeftIcon, ChevronRightIcon } from "@/components/custom/icons"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"

import { LatestDetectionImage } from "@/components/custom/latest-detection-image"
import { DetectionStatusCalibrationToggle } from "@/components/custom/detection-status-calibration"
import { useCalibrationStatusService } from '@/services/detectionStatusCalibrationService'
import { Direction, Detection, VehicleDetectionFilters } from '@/services/vehicleDetectionService'
import { getEndpoint } from '@/api/endpoints'

import { BASE_URL } from '@/api/endpoints'

export interface SpeedCalibrationAddProps {
  detections: Detection[];
  updateFilters: (newFilters: Partial<VehicleDetectionFilters>) => void;
  updateDetection: (id: number, updates: Partial<Detection>) => Promise<void>;
  deleteDetection: (id: number) => Promise<void>;
}

interface DetectionImages {
  [detectionId: number]: string[];
}

export function SpeedCalibrationAdd({
  detections,
  updateFilters,
  updateDetection,
  deleteDetection,
  onClose,
  onCalibrationAdded,
  speedCalibrationId,
}: SpeedCalibrationAddProps & {
  onClose: () => void;
  onCalibrationAdded: () => void;
  speedCalibrationId: number;
}) {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isCloseWarningOpen, setIsCloseWarningOpen] = useState(false)
  const [detectionImages, setDetectionImages] = useState<DetectionImages>({})
  const [videoSubmitted, setVideoSubmitted] = useState(false)
  const [currentImageIndex, setCurrentImageIndex] = useState<{ [detectionId: number]: number }>({})
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [calibrationName, setCalibrationName] = useState("Enter Calibration Name Here")
  const [cropValues, setCropValues] = useState({
    left_crop_l2r: 25,
    right_crop_l2r: 75,
    left_crop_r2l: 25,
    right_crop_r2l: 75,
  });
  
  const { uploadCalibrationVideo, processingVideo, submitSpeedCalibration } = useCalibrationStatusService();

  useEffect(() => {
    if (speedCalibrationId) { // Added check to ensure speedCalibrationId is defined
      updateFilters({ speed_calibration_id: speedCalibrationId, knownSpeedOnly: false });
    }
  }, [speedCalibrationId, updateFilters]);

  useEffect(() => {
    const fetchDetectionImages = async () => {
      for (const detection of detections) {
        if (!detectionImages[detection.id] && detection.thumbnail_path) {
          try {
            // Step 2: Query the thumbnail path to get a list of image paths
            const thumbnailResponse = await fetch(`${BASE_URL}/${detection.thumbnail_path}`);
            if (!thumbnailResponse.ok) throw new Error('Failed to fetch image paths');
            const imagePaths: string[] = await thumbnailResponse.json();

            // Step 3: Fetch all image paths
            const fetchedImages = await Promise.all(
              imagePaths.map(async (path) => {
                const imageResponse = await fetch(`${BASE_URL}${path}`);
                if (!imageResponse.ok) throw new Error(`Failed to fetch image: ${path}`);
                const blob = await imageResponse.blob();
                return { url: URL.createObjectURL(blob), name: path }; // Store both URL and name
              })
            );
            // Step 4: Sort images by detection date using the names from imagePaths
            fetchedImages.sort((a, b) => {
              const timestampA = Number(a.name.split('/')[5].split('_')[0]); //        "name": "/detections/2/2/bboxes/104_3.433.jpg"
              const timestampB = Number(b.name.split('/')[5].split('_')[0]);
              return timestampA - timestampB;
            });

            setDetectionImages(prev => ({ ...prev, [detection.id]: fetchedImages.map(image => image.url) }));
            setCurrentImageIndex(prev => ({ ...prev, [detection.id]: 0 }));
          } catch (error) {
            console.error('Error fetching detection images:', error);
          }
        }
      }
    };

    fetchDetectionImages();
  }, [detections]);

  const handleSpeedChange = async (id: number, speed: number) => {
    try {
      await updateDetection(id, { real_world_speed: speed });
      toast.success('Speed updated successfully');
    } catch (error) {
      console.error('Error updating speed:', error);
      toast.error('Failed to update speed. Please try again.');
    }
  }

  const handleDelete = async (id: number) => {
    try {
      await deleteDetection(id);
      toast.success('Detection deleted successfully');
    } catch (error) {
      console.error('Error deleting detection:', error);
      toast.error('Failed to delete detection. Please try again.');
    }
  }

  const handleSubmit = async () => {
    console.log("Submitting calibration data:", detections);

    // Check if at least 2 calibrations exist for each direction
    const leftToRightCalibrations = detections.filter(detection => detection.direction === Direction.leftToRight);
    const rightToLeftCalibrations = detections.filter(detection => detection.direction === Direction.rightToLeft);

    if (leftToRightCalibrations.length < 2 || rightToLeftCalibrations.length < 2) {
      toast.error('At least 2 calibrations are required for each direction before submitting. Please add more calibrations.');
      return;
    }

    const isSuccess = await submitSpeedCalibration(speedCalibrationId, calibrationName);
    if (isSuccess) {
      onCalibrationAdded();
      onClose();
    }
  }

  const handleClose = () => {
    setIsCloseWarningOpen(true)
  }

  const confirmClose = () => {
    setIsCloseWarningOpen(false)
    onClose()
  }

  const handlePrevImage = (detectionId: number) => {
    setCurrentImageIndex(prev => ({
      ...prev,
      [detectionId]: (prev[detectionId] - 1 + detectionImages[detectionId].length) % detectionImages[detectionId].length
    }));
  }

  const handleNextImage = (detectionId: number) => {
    setCurrentImageIndex(prev => ({
      ...prev,
      [detectionId]: (prev[detectionId] + 1) % detectionImages[detectionId].length
    }));
  }

  const handleVideoUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (processingVideo) {
      toast.error('Video is already being processed. Please wait for the current video to finish processing before uploading a new one.');
      return;
    }

    setVideoSubmitted(true)
    const file = event.target.files?.[0];
    if (file) {
      setVideoFile(file);
      try {
        await uploadCalibrationVideo(speedCalibrationId, file);
        toast.success('Video uploaded successfully');
      } catch (error) {
        console.error('Error uploading video:', error);
        //toast.error('Failed to upload video. Please try again.');
      }
    }
    setVideoFile(null);
    
    setVideoSubmitted(false);
    // Clear the input value
    event.target.value = '';
  }

  const totalVehicles = detections.length
  const vehiclesWithSpeed = detections.filter((vehicle) => vehicle.real_world_speed !== null).length

  const handleCropChange = useCallback((newCropValues: typeof cropValues) => {
    setCropValues(newCropValues);
    updateCropValuesOnBackend(newCropValues);
  }, []);

  const updateCropValuesOnBackend = async (newCropValues: typeof cropValues) => {
    try {
      const response = await fetch(getEndpoint('UPDATE_SPEED_CALIBRATION_CROP').replace('{calibration_id}', speedCalibrationId.toString()), {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          left_crop_l2r: Math.round(newCropValues.left_crop_l2r),
          right_crop_l2r: Math.round(newCropValues.right_crop_l2r),
          left_crop_r2l: Math.round(newCropValues.left_crop_r2l),
          right_crop_r2l: Math.round(newCropValues.right_crop_r2l),
        }),
      });
      if (!response.ok) {
        throw new Error('Failed to update crop values');
      }
    } catch (error) {
      console.error('Error updating crop values:', error);
      toast.error('Failed to update crop values');
    }
  };

  return (
    <>
      <Card className="w-[800px] max-h-[90vh] flex flex-col">
        <CardHeader className="pb-4">
          <div className="flex justify-between items-center">
            <CardTitle>Add Speed Calibration</CardTitle>
            <Button variant="ghost" onClick={handleClose}>x</Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4 flex-grow overflow-y-auto">
          <div className="space-y-2">
            <Label htmlFor="calibration-name">Calibration Name</Label>
            <Input
              id="calibration-name"
              placeholder="Enter calibration name"
              value={calibrationName}
              onChange={(e) => setCalibrationName(e.target.value)}
            />
          </div>
          <DetectionStatusCalibrationToggle />
          <div className="grid gap-4 md:grid-cols-2">
            <div className="col-span-2 md:col-span-1">
              <LatestDetectionImage
                allowCropAdjustment={true}
                onCropChange={handleCropChange}
                cropValues={cropValues}
              />
            </div>
            <div className="col-span-2 md:col-span-1">
              <h4 className="text-lg font-semibold mb-2">Crop Values</h4>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <p>Left to Right:</p>
                  <p>Left: {cropValues.left_crop_l2r.toFixed(2)}%</p>
                  <p>Right: {cropValues.right_crop_l2r.toFixed(2)}%</p>
                </div>
                <div>
                  <p>Right to Left:</p>
                  <p>Left: {cropValues.left_crop_r2l.toFixed(2)}%</p>
                  <p>Right: {cropValues.right_crop_r2l.toFixed(2)}%</p>
                </div>
              </div>
            </div>
          </div>
          <Separator />
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Detected Vehicles</h3>
          </div>
          <div className="grid gap-6 md:grid-cols-2">
            {isLoading ? (
              <div className="flex justify-center col-span-2">
                <div>Loading...</div>
              </div>
            ) : error ? (
              <div className="text-red-500 col-span-2">{error}</div>
            ) : (
              detections.map((detection) => (
                <Card key={detection.id} className="grid grid-cols-[1fr_auto] items-center gap-2">
                  <div className="relative">
                    {detectionImages[detection.id] && (
                      <>
                        <img
                          src={detectionImages[detection.id][currentImageIndex[detection.id] || 0]}
                          alt={`Vehicle detected at ${detection.detection_date}`}
                          className="w-full h-auto rounded-md"
                          style={{ aspectRatio: "16/9", objectFit: "cover" }}
                        />
                        <div className="absolute top-1/2 left-0 right-0 flex justify-between transform -translate-y-1/2">
                          <Button
                            size="icon"
                            variant="ghost"
                            onClick={() => handlePrevImage(detection.id)}
                            className="bg-black bg-opacity-50 text-white rounded-full p-1"
                          >
                            <ChevronLeftIcon className="w-6 h-6" />
                          </Button>
                          <Button
                            size="icon"
                            variant="ghost"
                            onClick={() => handleNextImage(detection.id)}
                            className="bg-black bg-opacity-50 text-white rounded-full p-1"
                          >
                            <ChevronRightIcon className="w-6 h-6" />
                          </Button>
                        </div>
                      </>
                    )}
                  </div>
                  <div className="flex flex-col items-end gap-2">
                    <div className="text-sm text-muted-foreground">{new Date(detection.detection_date).toLocaleString()}</div>
                    <div className="flex items-center gap-2">
                      <Input
                        type="number"
                        placeholder="Speed"
                        value={detection.real_world_speed || ""}
                        onChange={(e) => {
                          handleSpeedChange(detection.id, Number(e.target.value));
                        }}
                        className="w-24 h-8 text-sm"
                      />
                      <span>mph</span>
                    </div>
                    <Button size="icon" variant="ghost" onClick={() => handleDelete(detection.id)}>
                      <TrashIcon className="w-5 h-5" />
                    </Button>
                  </div>
                </Card>
              ))
            )}
          </div>
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Upload Calibration Video</h3>
          </div>
          <div className="flex items-center space-x-2">
            <Input
              type="file"
              accept="video/*"
              onChange={handleVideoUpload}
              disabled={processingVideo || videoSubmitted} // Disable input if processingVideo is true
            />
            {processingVideo && <span>Processing video...</span>}
            {videoFile && !processingVideo && <span>{videoFile.name}</span>}
          </div>
        </CardContent>
        <CardFooter>
          <Button onClick={handleSubmit} disabled={!calibrationName.trim()}>Submit Calibration</Button>
        </CardFooter>
      </Card>

      <Dialog open={isCloseWarningOpen} onOpenChange={setIsCloseWarningOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Warning</DialogTitle>
            <DialogDescription>
              Closing without submitting will not save any work. Are you sure you want to close?
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsCloseWarningOpen(false)}>Cancel</Button>
            <Button onClick={confirmClose}>Close</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}