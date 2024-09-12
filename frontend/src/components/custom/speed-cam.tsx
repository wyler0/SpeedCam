// © 2024 Wyler Zahm. All rights reserved.

'use client'
import Link from "next/link"
import { useState } from "react";
import { DetectionStatistics } from "@/components/custom/detection-statistics"
import { DetectionStatusToggle } from "@/components/custom/detection-status"
import { DetectedVehicles } from "@/components/custom/detected-vehicles"
import { SpeedCalibrations } from "@/components/custom/speed-calibrations"
import { CameraCalibrations } from "@/components/custom/camera-calibrations"
import { LatestDetectionImage } from "@/components/custom/latest-detection-image"

// Lift shared state into this root component
import { useVehicleDetectionService } from '@/services/vehicleDetectionService'; 
import { useDetectionStatusService } from '@/services/detectionStatusService'; 



export function SpeedCam() {

  // Setup the state for the SpeedCam component

  const [speedLimit, setSpeedLimit] = useState(35); // Default speed limit

  const {
    detections, 
    loading, 
    error, 
    filters, 
    updateFilters, 
    getStatistics, 
    PredefinedFilters, 
    updateDetection, 
    deleteDetection, 
  } = useVehicleDetectionService({}, 5000, speedLimit); // Pass speedLimit to useVehicleDetectionService

  const {
    isDetectionOn,
    availableCameras,
    selectedCamera,
    processingVideo,
    updateSelectedCamera,
    speedCalibrationId,
    calibrations,
    toggleDetection,
    updateSpeedCalibration,
    fetchCalibrationIds,
  } = useDetectionStatusService();


  const handleSmoothScroll = (e: React.MouseEvent<HTMLAnchorElement>, targetId: string) => {
    e.preventDefault();
    const target = document.getElementById(targetId);
    if (target) {
      const elementPosition = target.getBoundingClientRect().top; 
      const headerOffset = document.querySelector('header')?.offsetHeight || 0; 
      const offsetPosition = elementPosition + window.scrollY - headerOffset; 
      window.scrollTo({ top: offsetPosition, behavior: 'smooth' }); 
    }
  };

  const handleSpeedLimitChange = (newSpeedLimit: number) => {
    setSpeedLimit(newSpeedLimit);
  };

  return (
    <div className="flex flex-col min-h-screen">
      <header className="bg-primary text-primary-foreground py-4 px-6 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <h1 className="text-2xl font-bold">SpeedCam</h1>
          <nav className="flex items-center gap-4">
            <Link href="#-time-detectirealon" className="hover:text-muted-foreground" prefetch={false} scroll={false} onClick={(e) => handleSmoothScroll(e, 'real-time-detection')}>
              Real-Time Detection
            </Link>
            <Link href="#detected-vehicles" className="hover:text-muted-foreground" prefetch={false} scroll={false} onClick={(e) => handleSmoothScroll(e, 'detected-vehicles')}>
              Detected Vehicles
            </Link>
            <Link href="#speed-calibration" className="hover:text-muted-foreground" prefetch={false} scroll={false} onClick={(e) => handleSmoothScroll(e, 'speed-calibration')}>
              Speed Calibration
            </Link>
            <Link href="#camera-calibration" className="hover:text-muted-foreground" prefetch={false} scroll={false} onClick={(e) => handleSmoothScroll(e, 'camera-calibration')}>
              Camera Calibration
            </Link>
          </nav>
        </div>
      </header>
      <main className="flex-1 bg-muted/40 p-6 md:p-10">
        <div className="max-w-6xl mx-auto grid gap-8">
          <section id="real-time-detection">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold">Real-Time Detection</h2>
              <DetectionStatusToggle 
                isDetectionOn={isDetectionOn}
                speedCalibrationId={speedCalibrationId}
                calibrations={calibrations}
                availableCameras={availableCameras}
                selectedCamera={selectedCamera ? parseInt(selectedCamera) : null}
                processingVideo={processingVideo}
                toggleDetection={toggleDetection}
                updateSpeedCalibration={updateSpeedCalibration}
                updateSelectedCamera={updateSelectedCamera}
                initialSpeedLimit={speedLimit}
                onSpeedLimitChange={handleSpeedLimitChange}
              />
            </div>
            <div className="grid gap-6 mt-6">
              <LatestDetectionImage />
            </div>
            <div className="grid gap-6 mt-6">
              <DetectionStatistics 
                detections={detections}
                loading={loading}
                error={error}
                filters={filters}
                updateFilters={updateFilters}
                getStatistics={getStatistics}
                speedCalibrationId={speedCalibrationId}
                speedLimit={speedLimit}
              />
            </div>
          </section>
          <section id="detected-vehicles">
            <DetectedVehicles 
              detections={detections}
              loading={loading}
              error={error}
              filters={filters}
              updateFilters={updateFilters}
              PredefinedFilters={PredefinedFilters}
              speedCalibrationId={speedCalibrationId}
              calibrations={calibrations}
              fetchCalibrationIds={fetchCalibrationIds}
            />
          </section>
          <SpeedCalibrations 
            detections={detections}
            updateFilters={updateFilters}
            updateDetection={updateDetection}
            deleteDetection={deleteDetection}
          />
          <CameraCalibrations />
        </div>
      </main>
    </div>
  )
}