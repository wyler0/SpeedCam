// © 2024 Wyler Zahm. All rights reserved.

import React, { useEffect, useState, useMemo } from 'react';
import { addDays, format, parseISO } from 'date-fns';

import { Cell, XAxis, ScatterChart, Scatter, YAxis, ZAxis, Tooltip, Legend } from "recharts"
import { AlertTriangle } from "lucide-react"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuItem } from "@/components/ui/dropdown-menu";
import { Table, TableHeader, TableRow, TableHead, TableBody, TableCell } from "@/components/ui/table";
import { ChartContainer } from "@/components/ui/chart"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Calendar } from "@/components/ui/calendar"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog"

import { FilterIcon, CalendarIcon, TrashIcon } from '@/components/custom/icons';

import { Detection, Direction, PredefinedFilterType, VehicleDetectionFilters } from '@/services/vehicleDetectionService';
import { SpeedCalibration } from '@/services/detectionStatusService';
import { deleteVehicleDetection } from '@/services/vehicleDetectionService';
import { toast } from "react-hot-toast";

// Add these imports at the top of the file
import { saveAs } from 'file-saver';
import JSZip from 'jszip';
import { BASE_URL } from '@/api/endpoints';
import Image from 'next/image';
import { Pagination } from "@/components/ui/pagination";

interface DetectedVehiclesProps {
  detections: Detection[];
  loading: boolean;
  error: string | null;
  filters: VehicleDetectionFilters;
  updateFilters: (newFilters: Partial<VehicleDetectionFilters>) => void;
  PredefinedFilters: Record<string, PredefinedFilterType>;
  speedCalibrationId: string | null;
  calibrations: SpeedCalibration[];
  fetchCalibrationIds: () => Promise<void>;
}

// Add these utility functions after the imports and before the DetectedVehicles component

const convertToCSV = (data: Detection[]): string => {
  const headers = ['Time', 'Camera', 'Speed (MPH)', 'Speed (Pixels)', 'Direction', 'Image Directory'];
  const rows = data
    .filter(d => d.pixel_speed_estimate !== null)
    .map(d => [
      d.detection_date,
      d.speed_calibration_id,
      d.real_world_speed_estimate,
      d.pixel_speed_estimate,
      d.direction === Direction.leftToRight ? 'Left to Right' : 'Right to Left',
      `images/${d.id}`
    ]);
  
  return [
    headers.join(','),
    ...rows.map(r => r.join(','))
  ].join('\n');
};

const exportToCSV = (data: Detection[]) => {
  const csv = convertToCSV(data);
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const filename = `detected_vehicles_${new Date().toISOString()}.csv`;
  saveAs(blob, filename);
};

export function DetectedVehicles({
  detections,
  loading,
  error,
  filters,
  updateFilters,
  PredefinedFilters,
  speedCalibrationId,
  calibrations,
  fetchCalibrationIds,
}: DetectedVehiclesProps) {
  const [tempFilters, setTempFilters] = useState<VehicleDetectionFilters>({});
  const [isOpen, setIsOpen] = useState(false);
  const [detectionImages, setDetectionImages] = useState<{ [id: number]: string[] }>({});
  const [isExportDialogOpen, setIsExportDialogOpen] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10; // You can adjust this number as needed

  const memoizedDetections = useMemo(() => detections, [detections]);

  // Add this memoized sorted detections array
  const sortedDetections = useMemo(() => {
    return [...detections]
      .filter(detection => detection.real_world_speed_estimate != null)
      .sort((a, b) => parseISO(b.detection_date).getTime() - parseISO(a.detection_date).getTime());
  }, [detections]);

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
    else {
      fetchCalibrationIds();
      updateFilters({
        speed_calibration_id: undefined,
        startDate: undefined,
        endDate: undefined,
        predefinedFilter: undefined
      });
    }
  }, [speedCalibrationId, updateFilters, fetchCalibrationIds]);

  useEffect(() => {
    setTempFilters(filters);
  }, [filters]);

  useEffect(() => {
    const fetchDetectionImages = async () => {
      const newDetectionImages: { [id: number]: string[] } = {};
      for (const detection of memoizedDetections) {
        if (detection.thumbnail_path) {
          try {
            const thumbnailResponse = await fetch(`${BASE_URL}/${detection.thumbnail_path}`);
            if (!thumbnailResponse.ok) throw new Error('Failed to fetch image paths');
            const imagePaths: string[] = await thumbnailResponse.json();
            
            const fetchedImages = imagePaths.map(path => `${BASE_URL}${path}`);
            newDetectionImages[detection.id] = fetchedImages;
          } catch (error) {
            console.error('Error fetching detection images:', error);
          }
        }
      }
      setDetectionImages(newDetectionImages);
    };

    fetchDetectionImages();
  }, [memoizedDetections]);

  const handleFilterChange = (key: string, value: any) => {
    setTempFilters(prev => {
      const newFilters = { ...prev, [key]: value };
      
      if (key === 'predefinedFilter') {
        // Clear start and end dates when selecting a predefined filter
        newFilters.startDate = undefined;
        newFilters.endDate = undefined;
      } else if (key === 'startDate' || key === 'endDate') {
        // Clear predefined filter if dates are selected
        newFilters.predefinedFilter = undefined;
      }

      // Validate predefined filter
      if (key === 'predefinedFilter') {
        newFilters.predefinedFilter = value as PredefinedFilterType;
      }

      // Validate date range
      if (key === 'startDate' || key === 'endDate') {
        const startDate = newFilters.startDate ? new Date(newFilters.startDate) : undefined;
        const endDate = newFilters.endDate ? new Date(newFilters.endDate) : undefined;
        
        if (startDate && endDate && startDate > endDate) {
          if (key === 'startDate') {
            newFilters.endDate = addDays(startDate, 1).toISOString();
          } else {
            newFilters.startDate = addDays(endDate, -1).toISOString();
          }
        }
      }
      
      // Validate speed range
      if (key === 'minSpeed' || key === 'maxSpeed') {
        const minSpeed = typeof newFilters.minSpeed === 'number' ? newFilters.minSpeed : undefined;
        const maxSpeed = typeof newFilters.maxSpeed === 'number' ? newFilters.maxSpeed : undefined;
        
        if (minSpeed !== undefined && maxSpeed !== undefined) {
          if (minSpeed > maxSpeed) {
            if (key === 'minSpeed') {
              newFilters.maxSpeed = minSpeed;
            } else {
              newFilters.minSpeed = maxSpeed;
            }
          }
        }
      }
      return newFilters;
    });
  };

  const applyFilters = () => {
    updateFilters(tempFilters);
    setIsOpen(false);
  };

  const handleDropdownOpenChange = (open: boolean) => {
    setIsOpen(open);
    if (!open) {
      // Reset temporary filters when closing without applying
      setTempFilters(filters);
    }
  };

  // In the DetectedVehicles component, add this function:

  const handleExport = () => {
    if (memoizedDetections.length > 0) {
      setIsExportDialogOpen(true);
    } else {
      console.log('No data to export');
    }
  };

  const handleExportConfirm = async (includeImages: boolean) => {
    setIsExportDialogOpen(false);
    const zip = new JSZip();

    // Prepare CSV data
    const csv = convertToCSV(memoizedDetections);
    zip.file("detected_vehicles.csv", csv);

    if (includeImages) {
      for (const detection of memoizedDetections) {
        if (detectionImages[detection.id]) {
          const detectionFolder = zip.folder(`images/${detection.id}`);
          const sortedImages = [...detectionImages[detection.id]].sort((a, b) => {
            const timestampA = Number(a.split('/')[5].split('_')[0]);
            const timestampB = Number(b.split('/')[5].split('_')[0]);
            return timestampA - timestampB;
          });

          for (let i = 0; i < sortedImages.length; i++) {
            const imageUrl = sortedImages[i];
            const response = await fetch(imageUrl);
            const blob = await response.blob();
            const timestamp = imageUrl.split('/')[5].split('_')[0];
            detectionFolder?.file(`${timestamp}_${i}.jpg`, blob);
          }
        }
      }
    }

    const content = await zip.generateAsync({type: "blob"});
    const filename = `detected_vehicles_${includeImages ? 'with_images_' : ''}${new Date().toISOString()}.zip`;
    saveAs(content, filename);
  };

  const handleDelete = async (detectionId: number) => {
    try {
      await deleteVehicleDetection(detectionId);
      // Remove the deleted detection from the local state
      const updatedDetections = detections.filter(d => d.id !== detectionId);
      toast.success("The vehicle detection has been successfully deleted.");
    } catch (error) {
      console.error('Error deleting detection:', error);
      toast.error("Failed to delete the vehicle detection. Please try again.");
    }
  };

  const paginatedDetections = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    return sortedDetections.slice(startIndex, startIndex + itemsPerPage);
  }, [sortedDetections, currentPage, itemsPerPage]);

  const totalPages = Math.ceil(sortedDetections.length / itemsPerPage);

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Detected Vehicle Chart</CardTitle>
        <CardDescription>Filtered plot of detected vehicle speeds.</CardDescription>
        <div className="flex flex-col mb-4">
          <div className="flex items-center justify-between mb-2">
            {/* Active filters */}
            <div className="flex-grow flex flex-wrap gap-2">
              <span key='camera_calibration_id' className={`bg-gray-100 text-gray-800 text-sm font-medium px-2.5 py-0.5 rounded`}>
                Calibration: {filters.speed_calibration_id ? calibrations.find(c => c.id === filters.speed_calibration_id)?.name : 'Not Selected'}
              </span>
              {Object.entries(filters).map(([key, value]) => {
                if (value !== undefined && value !== '') {
                  let label = '';
                  let bgColor = 'bg-gray-100';
                  let textColor = 'text-gray-800';

                  switch (key) {
                    case 'speed_calibration_id':
                      break;
                    case 'startDate':
                      label = `From: ${format(new Date(value), 'PP')}`;
                      break;
                    case 'endDate':
                      label = `To: ${format(new Date(value), 'PP')}`;
                      break;
                    case 'direction':
                      label = `Direction: ${value === Direction.leftToRight.toString() ? "Left to Right" : "Right to Left"}`;
                      break;
                    case 'minSpeed':
                      label = `Min Speed: ${value} mph`;
                      break;
                    case 'maxSpeed':
                      label = `Max Speed: ${value} mph`;
                      break;
                  }

                  return (
                    <span key={key} className={`${bgColor} ${textColor} text-sm font-medium px-2.5 py-0.5 rounded`}>
                      {label}
                    </span>
                  );
                }
                return null;
              })}
            </div>
            {/* Buttons moved to the right */}
            <div className="flex items-center gap-2 ml-4">
              <DropdownMenu open={isOpen} onOpenChange={handleDropdownOpenChange}>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" size="sm" className="gap-2">
                    <FilterIcon className="w-4 h-4" />
                    Filters
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-80 max-h-[80vh] overflow-y-auto">
                  <DropdownMenuLabel>Filter by:</DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  
                  {/* Speed Calibration ID */}
                  <div className="p-2">
                    <Label htmlFor="speed_calibration_id">Speed Calibration ID</Label>
                    <Select onValueChange={(value) => handleFilterChange('speed_calibration_id', parseInt(value))}>
                      <SelectTrigger id="speed_calibration_id">
                        <SelectValue placeholder="Select calibration ID" />
                      </SelectTrigger>
                      <SelectContent>

                      {calibrations.filter(calibration => calibration.valid).length > 0 ? (
                          calibrations.filter(calibration => calibration.valid).map((calibration) => (
                            <SelectItem key={calibration.id} value={calibration.id.toString()}>{calibration.name}</SelectItem>
                          ))
                        ) : (
                          <SelectItem disabled value="No valid speed calibrations available">No valid speed calibrations available</SelectItem>
                        )}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Start Date */}
                  <div className="p-2">
                    <Popover>
                      <PopoverTrigger asChild>
                        <Button variant="outline" className="w-full justify-start text-left font-normal">
                          <CalendarIcon className="mr-2 h-4 w-4" />
                          {tempFilters.startDate ? format(new Date(tempFilters.startDate), 'PPP') : <span>Pick a date</span>}
                        </Button>
                      </PopoverTrigger>
                      <PopoverContent className="w-auto p-0">
                        <Calendar
                          mode="single"
                          selected={tempFilters.startDate ? new Date(tempFilters.startDate) : undefined}
                          onSelect={(date) => handleFilterChange('startDate', date?.toISOString())}
                          disabled={(date) => date > new Date(tempFilters.endDate || Date.now())}
                        />
                      </PopoverContent>
                    </Popover>
                  </div>

                  {/* End Date */}
                  <div className="p-2">
                    <Label>End Date</Label>
                    <Popover>
                      <PopoverTrigger asChild>
                        <Button variant="outline" className="w-full justify-start text-left font-normal">
                          <CalendarIcon className="mr-2 h-4 w-4" />
                          {tempFilters.endDate ? format(new Date(tempFilters.endDate), 'PPP') : <span>Pick a date</span>}
                        </Button>
                      </PopoverTrigger>
                      <PopoverContent className="w-auto p-0">
                        <Calendar
                          mode="single"
                          selected={tempFilters.endDate ? new Date(tempFilters.endDate) : undefined}
                          onSelect={(date) => handleFilterChange('endDate', date?.toISOString())}
                          disabled={(date) => date < new Date(tempFilters.startDate || 0)}
                        />
                      </PopoverContent>
                    </Popover>
                  </div>

                  {/* Min Speed */}
                  <div className="p-2">
                    <Label htmlFor="minSpeed">Min Speed (mph)</Label>
                    <Input
                      id="minSpeed"
                      type="number"
                      value={tempFilters.minSpeed || ''}
                      onChange={(e) => handleFilterChange('minSpeed', e.target.value ? parseInt(e.target.value) : undefined)}
                      min={0}
                    />
                  </div>

                  {/* Max Speed */}
                  <div className="p-2">
                    <Label htmlFor="maxSpeed">Max Speed (mph)</Label>
                    <Input
                      id="maxSpeed"
                      type="number"
                      value={tempFilters.maxSpeed || ''}
                      onChange={(e) => handleFilterChange('maxSpeed', e.target.value ? parseInt(e.target.value) : undefined)}
                      min={tempFilters.minSpeed || 0}
                    />
                  </div>

                  {/* Direction */}
                  <div className="p-2">
                    <Label htmlFor="direction">Direction</Label>
                    <Select onValueChange={(value) => handleFilterChange('direction', value)}>
                      <SelectTrigger id="direction">
                        <SelectValue placeholder="Select direction" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value={Direction.leftToRight.toString()}>Left to Right</SelectItem>
                        <SelectItem value={Direction.rightToLeft.toString()}>Right to Left</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="p-2">
                    <Button onClick={applyFilters} className="w-full">Apply Filters</Button>
                  </div>
                </DropdownMenuContent>
              </DropdownMenu>
              <Button size="sm" onClick={handleExport}>Export</Button>
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid gap-4 mt-4">
          {filters.speed_calibration_id != null ? (
            <Card>
              <CardContent className="p-0">
                <ScatterPlotChart data={memoizedDetections} />
              </CardContent>
            </Card>
          ) : (
            <Alert variant="default" className="full">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                No speed calibration selected. Scatter plot is not available.
              </AlertDescription>
            </Alert>
          )}
          {filters.speed_calibration_id != null ? (
            <Card>
              <CardHeader>
                <CardTitle>Detected Vehicles Table</CardTitle>
                <CardDescription>List of vehicles detected with speed, direction, and other details (most recent first).</CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div>Loading...</div>
                ) : error ? (
                  <Alert variant="destructive">
                    <AlertTriangle className="h-4 w-4" />
                    <AlertDescription>
                      Error: {error}
                    </AlertDescription>
                  </Alert>
                ) : (
                  <>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Thumbnail</TableHead>
                          <TableHead>Time</TableHead>
                          <TableHead>Camera</TableHead>
                          <TableHead>Speed</TableHead>
                          <TableHead>Direction</TableHead>
                          <TableHead>Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {paginatedDetections.map((detection) => (
                          <TableRow key={detection.id}>
                            <TableCell>
                              {detectionImages[detection.id] && detectionImages[detection.id].length > 0 && (
                                <div className="w-36 rounded-md">
                                  <img
                                    src={detectionImages[detection.id][Math.floor(detectionImages[detection.id].length / 2)]}
                                    alt={`Vehicle detected at ${detection.detection_date}`}
                                    className="rounded-md"
                                    style={{ 
                                      width: '100%', 
                                      height: 'auto', 
                                      objectFit: "cover", 
                                      clipPath: "inset(15% 0 15% 0)" // Cropping top and bottom 15%
                                    }}
                                  />
                                </div>
                              )}
                            </TableCell>
                            <TableCell>{format(parseISO(detection.detection_date), 'yyyy-MM-dd HH:mm:ss')}</TableCell>
                            <TableCell>{detection.speed_calibration_id}</TableCell>
                            <TableCell>{detection.real_world_speed_estimate} mph</TableCell>
                            <TableCell>{detection.direction}</TableCell>
                            <TableCell>
                              <Button 
                                variant="destructive" 
                                size="icon" 
                                className="rounded-full"
                                onClick={() => handleDelete(detection.id)}
                              >
                                <TrashIcon className="w-5 h-5" />
                                <span className="sr-only">Delete</span>
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                    <Pagination
                      currentPage={currentPage}
                      totalPages={totalPages}
                      onPageChange={handlePageChange}
                    />
                  </>
                )}
              </CardContent>
            </Card>
          ) : (
            <Alert variant="default" className="full">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                No speed calibration selected. Table is not available.
              </AlertDescription>
            </Alert>
          )}
          
        </div>
      </CardContent>
      <Dialog open={isExportDialogOpen} onOpenChange={setIsExportDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Export Options</DialogTitle>
            <DialogDescription>
              Do you want to include images in the export? This will increase the file size.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button onClick={() => handleExportConfirm(false)}>Export without Images</Button>
            <Button onClick={() => handleExportConfirm(true)}>Export with Images</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}

const CustomShape = (props: any) => {
  const { cx, cy, fill } = props;
  return <circle cx={cx} cy={cy} r={4} fill={fill} />;
};  

function ScatterPlotChart({ data }: { data: Detection[] }) {
  const chartConfig = {
    leftToRight: {
      label: "Left to Right",
      color: "hsl(0, 100%, 50%)", // Red
    },
    rightToLeft: {
      label: "Right to Left",
      color: "hsl(240, 100%, 50%)", // Blue
    },
  };

  const formattedData = useMemo(() => data.map(d => ({
    time: format(parseISO(d.detection_date), 'HH:mm:ss'),
    speed: d.real_world_speed_estimate,
    direction: d.direction
  })).filter(d => d.speed !== null), [data]);

  return (
    <div className="w-full">
      <ChartContainer
        config={chartConfig}
      >
        <ScatterChart
          width={800}
          height={400}
          margin={{ top: 20, right: 20, bottom: 20, left: 40 }}
        >
          <XAxis dataKey="time" name="Time" tickLine={false} axisLine={true} />
          <YAxis dataKey="speed" name="Speed" unit=" mph" />
          <ZAxis dataKey="direction" range={[0, 1]} name="Direction" />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
          <Legend 
            payload={[
              { value: chartConfig.leftToRight.label, type: 'square', color: chartConfig.leftToRight.color },
              { value: chartConfig.rightToLeft.label, type: 'square', color: chartConfig.rightToLeft.color },
            ]}
          />
          <Scatter
            name="Vehicles"
            data={formattedData}
            shape={<CustomShape />}
          >
            {formattedData.map((entry, index) => (
              <Cell 
                key={`cell-${index}`} 
                fill={entry.direction === 0 ? chartConfig.leftToRight.color : chartConfig.rightToLeft.color} 
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ChartContainer>
    </div>
  )
}
