import { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card"
import { useVehicleDetectionService } from '@/services/vehicleDetectionService';
import { useDetectionStatusService } from '@/services/detectionStatusService';
import { Alert, AlertDescription } from "@/components/ui/alert"
import { AlertTriangle } from "lucide-react"

export function DetectionStatistics() {
  const { detections, loading, error, filters, updateFilters, getStatistics } = useVehicleDetectionService();
  const { speedCalibrationId } = useDetectionStatusService();

  useEffect(() => {
    if (speedCalibrationId) {
      const sevenDaysAgo = new Date();
      sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
      
      updateFilters({
        speed_calibration_id: parseInt(speedCalibrationId),
        startDate: sevenDaysAgo.toISOString(),
        endDate: new Date().toISOString(),
        predefinedFilter: 'LAST_7_DAYS'
      });
    }
  }, [speedCalibrationId, updateFilters]);

  const stats = getStatistics();

  return (
    <Card>
      <CardHeader>
        <CardTitle>Detection Statistics (Last 7 Days)</CardTitle>
        <CardDescription>Current real-time detection status and statistics.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="flex flex-col items-center gap-2">
            <div className="text-4xl font-bold">{speedCalibrationId ? stats.vehiclesDetected : 'N/A'}</div>
            <div className="text-muted-foreground">Vehicles Detected</div>
          </div>
          <div className="flex flex-col items-center gap-2">
            <div className="text-4xl font-bold">{speedCalibrationId ? `${stats.averageSpeed.toFixed(1)} mph` : 'N/A'}</div>
            <div className="text-muted-foreground">Average Speed</div>
          </div>
          <div className="flex flex-col items-center gap-2">
            <div className="text-4xl font-bold">{speedCalibrationId ? stats.speedingViolations : 'N/A'}</div>
            <div className="text-muted-foreground">Speeding Violations</div>
          </div>
        </div>
      </CardContent>
      {!speedCalibrationId && (
        <CardFooter>
          <Alert variant="default" className="w-full">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              No speed calibration selected. Statistics are not available.
            </AlertDescription>
          </Alert>
        </CardFooter>
      )}
      {error && (
        <CardFooter>
          <Alert variant="destructive" className="w-full">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              Error: {error}
            </AlertDescription>
          </Alert>
        </CardFooter>
      )}
    </Card>
  )
}