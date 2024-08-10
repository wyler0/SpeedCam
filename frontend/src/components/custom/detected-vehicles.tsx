import React, { useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuItem, DropdownMenuSub, DropdownMenuSubTrigger, DropdownMenuSubContent, DropdownMenuRadioGroup, DropdownMenuRadioItem } from "@/components/ui/dropdown-menu";
import { Table, TableHeader, TableRow, TableHead, TableBody, TableCell } from "@/components/ui/table";
import { ChartContainer } from "@/components/ui/chart"
import { Cell, XAxis, ScatterChart, Scatter, YAxis, ZAxis, Tooltip, Legend } from "recharts"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { AlertTriangle } from "lucide-react"
import { Calendar } from "@/components/ui/calendar"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { addDays } from 'date-fns';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";

import { useVehicleDetectionService, Detection, Direction, PredefinedFilterType, VehicleDetectionFilters } from '@/services/vehicleDetectionService';
import { useDetectionStatusService } from '@/services/detectionStatusService';
import { EyeIcon, FilterIcon, CalendarIcon } from '@/components/custom/icons';
import { format } from 'date-fns';

export function DetectedVehicles() {
  const { detections, loading, error, filters, updateFilters, PredefinedFilters } = useVehicleDetectionService();
  const { speedCalibrationId, calibrationIds } = useDetectionStatusService();
  const [tempFilters, setTempFilters] = useState<VehicleDetectionFilters>({});
  const [isOpen, setIsOpen] = useState(false);
  const [isPredefinedFilterOpen, setIsPredefinedFilterOpen] = useState(false);

  useEffect(() => {
    if (speedCalibrationId) {
      const sevenDaysAgo = new Date();
      sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
      
      updateFilters({
        speedCalibrationId: parseInt(speedCalibrationId),
        startDate: sevenDaysAgo.toISOString(),
        endDate: new Date().toISOString(),
        predefinedFilter: 'LAST_7_DAYS'
      });
    }
  }, [speedCalibrationId, updateFilters]);

  useEffect(() => {
    setTempFilters(filters);
  }, [filters]);

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

  return (
    <Card>
      <CardHeader>
        <CardTitle>Detected Vehicle Chart</CardTitle>
        <CardDescription>Filtered plot of detected vehicle speeds.</CardDescription>
        <div className="flex flex-col mb-4">
          <div className="flex items-center justify-between mb-2">
            {/* Active filters */}
            <div className="flex-grow flex flex-wrap gap-2">
              <span key='speedCalibrationId' className={`bg-gray-100 text-gray-800 text-sm font-medium px-2.5 py-0.5 rounded`}>
                Calibration: {filters.speedCalibrationId || 'Not Selected'}
              </span>
              {Object.entries(filters).map(([key, value]) => {
                if (value !== undefined && value !== '') {
                  let label = '';
                  let bgColor = 'bg-gray-100';
                  let textColor = 'text-gray-800';

                  switch (key) {
                    case 'speedCalibrationId':
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
                    <Label htmlFor="speedCalibrationId">Speed Calibration ID</Label>
                    <Select onValueChange={(value) => handleFilterChange('speedCalibrationId', parseInt(value))}>
                      <SelectTrigger id="speedCalibrationId">
                        <SelectValue placeholder="Select calibration ID" />
                      </SelectTrigger>
                      <SelectContent>
                        {calibrationIds.map((id) => (
                          <SelectItem key={id} value={id.toString()}>{id}</SelectItem>
                        ))}
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
              <Button size="sm">Export</Button>
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid gap-4 mt-4">
          {speedCalibrationId ? (
            <Card>
              <CardContent className="p-0">
                <ScatterPlotChart data={detections} />
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
          {speedCalibrationId ? (
            <Card>
              <CardHeader>
                <CardTitle>Detected Vehicles Table</CardTitle>
              <CardDescription>List of vehicles detected with speed, direction, and other details.</CardDescription>
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
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Time</TableHead>
                        <TableHead>Camera</TableHead>
                        <TableHead>Speed</TableHead>
                        <TableHead>Direction</TableHead>
                        <TableHead>Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                      <TableBody>
                        {detections.map((detection) => (
                          <TableRow key={detection.id}>
                            <TableCell>{detection.detection_date}</TableCell>
                            <TableCell>{detection.speed_calibration_id}</TableCell>
                            <TableCell>{detection.estimated_speed} mph</TableCell>
                            <TableCell>{detection.direction}</TableCell>
                            <TableCell>
                              <Button variant="ghost" size="icon" className="rounded-full">
                                <EyeIcon className="w-5 h-5" />
                                <span className="sr-only">View</span>
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                  </Table>
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

  const formattedData = data.map(d => ({
    time: new Date(d.detection_date).toLocaleTimeString(),
    speed: d.estimated_speed,
    direction: d.direction
  }));

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