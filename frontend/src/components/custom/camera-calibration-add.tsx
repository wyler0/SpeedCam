// © 2024 Wyler Zahm. All rights reserved.



import React, { useState, useRef, useCallback } from 'react';
import { Camera, CameraType } from 'react-camera-pro';
import { toast } from "react-hot-toast";

import { Card, CardHeader, CardContent, CardFooter, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { CameraIcon, XIcon, SpinnerIcon, CheckIcon, UploadIcon } from "@/components/custom/icons"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

import { getEndpoint } from "@/api/endpoints";

interface CameraCalibrationAddProps {
  onClose: () => void;
  onCalibrationAdded: () => void;
}

type CameraRef = {
  takePhoto: () => string;
};

type ImageStatus = 'loading' | 'success' | 'error' | null;

interface CapturedImage {
  src: string;
  status: ImageStatus;
  flipped: boolean;
}

export function CameraCalibrationAdd({ onClose, onCalibrationAdded }: CameraCalibrationAddProps) {
  const [cameraType, setCameraType] = useState<'user' | 'environment'>('environment');
  const [capturedImages, setCapturedImages] = useState<CapturedImage[]>([]);
  const [rows, setRows] = useState<number>(0);
  const [columns, setColumns] = useState<number>(0);
  const [cameraName, setCameraName] = useState<string>('');
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [nameError, setNameError] = useState<string | null>(null);
  const camera = useRef<CameraRef | null>(null);
  const [isCapturing, setIsCapturing] = useState<boolean>(true);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isCameraFlipped, setIsCameraFlipped] = useState<boolean>(false);

  const flipImage = async (imageSrc: string): Promise<string> => {
    const response = await fetch(imageSrc);
    const blob = await response.blob();
    const imageBitmap = await createImageBitmap(blob);
    const canvas = document.createElement('canvas');
    canvas.width = imageBitmap.width;
    canvas.height = imageBitmap.height;
    const ctx = canvas.getContext('2d');
    ctx?.scale(-1, 1);
    ctx?.drawImage(imageBitmap, -canvas.width, 0);
    return canvas.toDataURL('image/jpeg');
  };

  const handleCapture = async () => {
    if (camera.current && 'takePhoto' in camera.current) {
      let photo = camera.current.takePhoto();
      
      const newImage: CapturedImage = { src: photo, status: 'loading', flipped: isCameraFlipped };
      setCapturedImages(prev => [...prev, newImage]);

      try {
        // Convert base64 to blob
        const response = await fetch(photo);
        const blob = await response.blob();

        // Create FormData and append the image and grid dimensions
        const formData = new FormData();
        formData.append('file', blob, 'calibration_image.jpg');
        formData.append('rows', rows.toString());
        formData.append('columns', columns.toString());

        const validateResponse = await fetch(getEndpoint('VALIDATE_CALIBRATION_IMAGE'), {
          method: 'POST',
          body: formData,
        });

        if (validateResponse.ok) {
          const result = await validateResponse.json();
          if (result.corners_found) {
            setCapturedImages(prev => 
              prev.map(img => img.src === photo ? { ...img, status: 'success' } : img)
            );
          } else {
            throw new Error('Corners not found in the image');
          }
        } else {
          throw new Error('Image validation failed');
        }
      } catch (error) {
        console.error('Error validating image:', error);
        setCapturedImages(prev => 
          prev.map(img => img.src === photo ? { ...img, status: 'error' } : img)
        );
      }
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        let imageSrc = URL.createObjectURL(file);
        
        if (isCameraFlipped) {
          imageSrc = await flipImage(imageSrc);
        }

        const newImage: CapturedImage = { src: imageSrc, status: 'loading', flipped: isCameraFlipped };
        setCapturedImages(prev => [...prev, newImage]);

        try {
          const formData = new FormData();
          formData.append('file', file);
          formData.append('rows', rows.toString());
          formData.append('columns', columns.toString());

          const validateResponse = await fetch(getEndpoint('VALIDATE_CALIBRATION_IMAGE'), {
            method: 'POST',
            body: formData,
          });

          if (validateResponse.ok) {
            const result = await validateResponse.json();
            if (result.corners_found) {
              setCapturedImages(prev => 
                prev.map(img => img.src === newImage.src ? { ...img, status: 'success' } : img)
              );
            } else {
              throw new Error('Corners not found in the image');
            }
          } else {
            throw new Error('Image validation failed');
          }
        } catch (error) {
          console.error('Error validating image:', error);
          setCapturedImages(prev => 
            prev.map(img => img.src === newImage.src ? { ...img, status: 'error' } : img)
          );
        }
      }
    }
  };

  const handleDeleteImage = (index: number) => {
    setCapturedImages(prev => prev.filter((_, i) => i !== index));
  };

  const renderImageStatus = (status: ImageStatus) => {
    switch (status) {
      case 'loading':
        return <SpinnerIcon className="w-6 h-6 animate-spin text-blue-500" />;
      case 'success':
        return <CheckIcon className="w-6 h-6 text-green-500" />;
      case 'error':
        return <XIcon className="w-6 h-6 text-red-500" />;
      default:
        return null;
    }
  };

  const toggleCaptureMethod = () => {
    setIsCapturing(!isCapturing);
  };

  const handleFlipToggle = async (isFlipped: boolean) => {
    setIsCameraFlipped(isFlipped);
    const updatedImages = await Promise.all(capturedImages.map(async (img) => {
      if (img.flipped !== isFlipped) {
        const flippedSrc = await flipImage(img.src);
        return { ...img, src: flippedSrc, flipped: isFlipped };
      }
      return img;
    }));
    setCapturedImages(updatedImages);
  };

  const handleSubmit = async () => {
    // Validate form
    if (!cameraName.trim()) {
      toast.error("Camera name is required");
      return;
    }
    if (rows < 5 || rows > 10 || columns < 5 || columns > 10) {
      toast.error("Rows and columns must be between 5 and 10");
      return;
    }
    const successfulImages = capturedImages.filter(img => img.status === 'success');
    if (successfulImages.length < 8) {
      toast.error("At least 8 successfully processed images are required");
      return;
    }

    setIsSubmitting(true);
    setNameError(null);

    try {
      // Step 1: Create the calibration
      const calibrationData = {
        camera_name: cameraName,
        rows: rows,
        cols: columns,
        horizontal_flip: isCameraFlipped, // Include the flip information
      };

      const createResponse = await fetch(getEndpoint('CAMERA_CALIBRATIONS'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(calibrationData),
      });

      if (!createResponse.ok) {
        const errorData = await createResponse.json();
        if (errorData.detail === "A calibration with this camera name already exists") {
          setNameError("A calibration with this name already exists. Please choose a different name.");
          throw new Error(errorData.detail);
        }
        throw new Error('Failed to create calibration');
      }

      const createdCalibration = await createResponse.json();

      // Step 2: Upload images (in parallel)
      const uploadPromises = successfulImages.map(async (img) => {
        const blob = await fetch(img.src).then(r => r.blob());
        const formData = new FormData();
        formData.append('file', blob, 'calibration_image.jpg');

        const uploadResponse = await fetch(getEndpoint('UPLOAD_CALIBRATION_IMAGE').replace('{calibration_id}', createdCalibration.id.toString()), {
          method: 'POST',
          body: formData,
        });

        if (!uploadResponse.ok) {
          throw new Error(`Failed to upload image: ${uploadResponse.statusText}`);
        }
      });

      await Promise.all(uploadPromises);

      // Step 3: Process the calibration
      const processResponse = await fetch(getEndpoint('PROCESS_CALIBRATION').replace('{calibration_id}', createdCalibration.id.toString()), {
        method: 'POST',
      });

      if (!processResponse.ok) {
        throw new Error('Failed to process calibration');
      }

      const processResult = await processResponse.json();
      toast.success(processResult.message);

      onCalibrationAdded();
      onClose();
    } catch (error) {
      console.error('Error submitting calibration:', error);
      if (error instanceof Error && error.message !== "A calibration with this camera name already exists") {
        toast.error("Failed to create or process calibration");
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Card className="w-[500px] max-h-[90vh] flex flex-col">
      <CardHeader className="pb-4">
        <div className="flex justify-between items-center">
          <CardTitle>Add Camera Calibration</CardTitle>
          <Button variant="ghost" onClick={onClose}>x</Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 flex-grow overflow-y-auto">
        <div className="flex items-center gap-4">
          <Input 
            id="camera-name" 
            placeholder="Camera Name" 
            className={`flex-1 ${nameError ? 'border-red-500' : ''}`}
            value={cameraName}
            onChange={(e) => {
              setCameraName(e.target.value);
              setNameError(null);
            }}
          />
        </div>
        {nameError && <p className="text-red-500 text-sm">{nameError}</p>}
        <Select onValueChange={(value: 'user' | 'environment') => setCameraType(value)} value={cameraType}>
          <SelectTrigger>
            <SelectValue placeholder="Select camera type" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="environment">Back Camera</SelectItem>
            <SelectItem value="user">Front Camera</SelectItem>
          </SelectContent>
        </Select>
        <div className="flex items-center gap-4">
          <label htmlFor="rows" className="flex-none">Row Count:</label>
          <Input 
            type="number" 
            id="rows" 
            placeholder="Number of Rows" 
            className="flex-1"
            value={rows}
            onChange={(e) => setRows(Number(e.target.value))}
            min={5}
            max={10}
          />
          <label htmlFor="columns" className="flex-none">Column Count:</label>
          <Input 
            type="number" 
            id="columns" 
            placeholder="Number of Columns" 
            className="flex-1"
            value={columns}
            onChange={(e) => setColumns(Number(e.target.value))}
            min={5}
            max={10}
          />
        </div>
        <div className="flex items-center gap-4">
          <label htmlFor="flip-camera" className="flex-none">Flip Camera:</label>
          <input
            type="checkbox"
            id="flip-camera"
            checked={isCameraFlipped}
            onChange={(e) => handleFlipToggle(e.target.checked)}
          />
        </div>
        <div className="flex justify-between items-center">
          <Button onClick={toggleCaptureMethod}>
            {isCapturing ? 'Switch to Upload' : 'Switch to Capture'}
          </Button>
          {!isCapturing && (
            <Button onClick={() => fileInputRef.current?.click()}>
              <UploadIcon className="mr-2 h-4 w-4" /> Upload Images
            </Button>
          )}
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            accept="image/*"
            multiple
            onChange={handleFileUpload}
          />
        </div>

        {isCapturing ? (
          <div className="relative w-full aspect-video bg-muted rounded-md overflow-hidden">
            <div style={{ transform: isCameraFlipped ? 'scaleX(-1)' : 'none' }}>
              <Camera 
                ref={camera} 
                facingMode={cameraType} 
                aspectRatio={16 / 9} 
                errorMessages={{
                  noCameraAccessible: 'No camera device accessible. Please connect your camera or try a different browser.',
                  permissionDenied: 'Permission denied. Please refresh and give camera permission.',
                  switchCamera:
                    'It is not possible to switch camera to different one because there is only one video device accessible.',
                  canvas: 'Canvas is not supported.',
                }}
              />
            </div>
            <div className="absolute bottom-4 right-4">
              <Button onClick={handleCapture} disabled={rows === 0 || columns === 0}>Capture</Button>
            </div>
          </div>
        ) : (
          <div className="w-full aspect-video bg-muted flex items-center justify-center">
            <p>Upload images using the button above</p>
          </div>
        )}

        <div className="h-[calc(2*33.33vw)] overflow-y-auto">
          <div className="grid grid-cols-3 gap-4 content-start">
            {capturedImages.map((img, index) => (
              <div key={index} className="relative">
                <img
                  src={img.src}
                  alt={`Captured Image ${index + 1}`}
                  className={`rounded-md w-full h-auto ${img.status === 'error' ? 'opacity-50' : ''}`}
                  style={{ 
                    aspectRatio: "4/3", 
                    objectFit: "cover", 
                    transform: img.flipped ? 'scaleX(-1)' : 'none' 
                  }}
                />
                <div className="absolute top-1 right-1 flex gap-1">
                  {renderImageStatus(img.status)}
                  <Button
                    variant="destructive"
                    size="icon"
                    className="p-1"
                    onClick={() => handleDeleteImage(index)}
                  >
                    <XIcon className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            ))}
            <div
              className="rounded-md w-full h-auto bg-muted flex items-center justify-center cursor-pointer"
              style={{ aspectRatio: "4/3" }}
              onClick={() => isCapturing ? handleCapture() : fileInputRef.current?.click()}
            >
              <CameraIcon className="w-8 h-8 text-muted-foreground" />
            </div>
          </div>
        </div>
      </CardContent>
      <CardFooter>
        <Button 
          className="w-full" 
          onClick={handleSubmit} 
          disabled={isSubmitting || capturedImages.filter(img => img.status === 'success').length < 8}
        >
          {isSubmitting ? 'Submitting...' : 'Save Calibration'}
        </Button>
      </CardFooter>
    </Card>
  )
}