# © 2024 Wyler Zahm. All rights reserved.

from typing import List

from pydantic import BaseModel

import numpy as np
import cv2 as cv

from core.optical_flow.optical_flow_base import OpticalFlowDetector, register_flow

class OpenCVOpticalFlowParams(BaseModel):
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 25
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0
    
    MIN_FLOW_MAGNITUDE: float = 4 # Minimum magnitude of flow to be considered. Dependent on vehicle speed, flow compute params, etc.
    MIN_DISTANCE_BETWEEN_GROUPS: float = 150 # Minimum distance between groups to be considered separate events.
    MAX_DISTANCE_FROM_CENTER: float = 400 # Maximum distance from center to form an event. Related to vehicle size.
    MIN_FLOW_COUNT: int = 20 # Minimum number of flows to form an event.
    
@register_flow
class OpenCVOpticalFlowDetector(OpticalFlowDetector):
    name = 'opencv_farneback'
    params: OpenCVOpticalFlowParams = OpenCVOpticalFlowParams()
    
    def __init__(self, params: OpenCVOpticalFlowParams = None):
        self.set_parameters(params)

    def compute_flow(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute optical flow over a group of N frames.
        
        Args:
            frames (List[np.ndarray]): List of input frames as numpy arrays.
            
        """
        if len(frames) < 2:
            raise ValueError("At least 2 frames are required to compute optical flow.")

        prvs = cv.cvtColor(frames[0], cv.COLOR_BGR2GRAY)

        flows = []
        for i in range(1, len(frames)):
            next = cv.cvtColor(frames[i], cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(prvs, next, None, 
                                               self.params.pyr_scale, self.params.levels, self.params.winsize, 
                                               self.params.iterations, self.params.poly_n, self.params.poly_sigma, 
                                               self.params.flags)
            prvs = next
            
            # Extract motion vectors above threshold magnitude
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            mag_threshold = 5
            mask = mag > mag_threshold
            y, x = np.where(mask)
            
            # Store as vectors: [x, y, dx, dy]
            vectors = np.column_stack((x, y, flow[mask]))
            
            flows.append(vectors)
        
        return flows
    
    def group_flows(self, flows: List[np.ndarray]) -> List[List[np.ndarray]]:
        """
        Group flows into event groups, ensuring flows in different directions are never grouped together.
        """
        final_groups = []
        for frame_flows in flows:
            groups = []
            frame_flows = np.array(frame_flows)
            
            # 1. Magnitude-based filtering
            magnitudes = np.linalg.norm(frame_flows[:, 2:], axis=1)
            significant_flows = frame_flows[magnitudes > self.params.MIN_FLOW_MAGNITUDE]
            
            if len(significant_flows) > 0:
                # 2. Direction-based separation (left vs right)
                left_flows = significant_flows[significant_flows[:, 2] < 0]
                right_flows = significant_flows[significant_flows[:, 2] >= 0]
                
                for direction_flows in [left_flows, right_flows]:
                    if len(direction_flows) > self.params.MIN_FLOW_COUNT:
                        # 3. Spatial proximity clustering
                        clusters = self.cluster_by_proximity(direction_flows)
                        groups.extend(clusters)
            
            final_groups.append(groups)
        
        return final_groups
    
    def cluster_by_proximity(self, flows: np.ndarray) -> List[np.ndarray]:
        """
        Cluster flows based on spatial proximity.
        """
        clusters = []
        remaining_flows = flows.copy()
        
        while len(remaining_flows) > 0:
            center = remaining_flows[0, :2]
            distances = np.linalg.norm(remaining_flows[:, :2] - center, axis=1)
            close_flows = remaining_flows[distances < self.params.MAX_DISTANCE_FROM_CENTER]
            
            if len(close_flows) > self.params.MIN_FLOW_COUNT:
                clusters.append(close_flows)
            
            remaining_flows = remaining_flows[distances >= self.params.MAX_DISTANCE_FROM_CENTER]
        
        return clusters
    
    def visualize_flow_groups(self, flow_groups: List[List[np.ndarray]], frames: List[np.ndarray], output_path: str = None, opacity: float = 0.3) -> List[str]:
        """
        Generate a visual representation of the optical flow groups with reduced opacity.
        
        Args:
            flow_groups (List[List[np.ndarray]]): List of optical flow vector groups.
            frames (List[np.ndarray]): List of input frames.
            output_path (str, optional): Path to save the visualization. If None, don't save.
            opacity (float, optional): Opacity of the flow vectors (0.0 to 1.0). Default is 0.3.
        
        Returns:
            List[str]: List of paths to the saved visualization images.
        """
        colors = [
            (255, 0, 0),   # Red
            (0, 255, 0),   # Green
            (0, 0, 255),   # Blue
            (255, 255, 0), # Yellow
            (0, 255, 255), # Cyan
            (255, 0, 255), # Magenta
            (128, 0, 0),   # Maroon
            (0, 128, 0),   # Dark Green
            (0, 0, 128),   # Navy
            (128, 128, 0)  # Olive
        ]

        paths = []
        for i, frame_groups in enumerate(flow_groups):
            img = frames[i+1].copy()
            overlay = np.zeros_like(img)
            
            for c, flow in enumerate(frame_groups):    
                color = colors[c % len(colors)]  # Use modulo to cycle through colors if more groups than colors
                for x, y, dx, dy in flow:
                    x, y, dx, dy = map(int, [x, y, dx, dy])
                    cv.arrowedLine(overlay, (x, y), (x + dx, y + dy), color, 1, tipLength=0.5)

            # Blend the overlay with the original image
            result = cv.addWeighted(overlay, opacity, img, 1 - opacity, 0)

            # Write images
            if output_path:
                path = f'{output_path}_{i}.png'
                cv.imwrite(path, result)
                paths.append(path)

        return paths
    
    def visualize_flow(self, flow: List[np.ndarray], frames: List[np.ndarray], output_path: str = None) -> None:
        """
        Generate a visual representation of the optical flow.
        
        Args:
            output_path (str, optional): Path to save the visualization. If None, don't save.
        
        Returns:
            np.ndarray: Visualization of the optical flow.
        """
        assert len(flow) == len(frames) - 1, "Must have at least one more frames than flows. Please provide the input used for compute_flow and the output of compute_flow."
        
        for i, vectors in enumerate(flow):
            img = frames[i+1].copy()
            
            # Draw vectors on the image
            for x, y, dx, dy in vectors:
                x, y, dx, dy = map(int, [x, y, dx, dy])
                cv.arrowedLine(img, (x, y), (x + dx, y + dy), (0, 255, 0), 1, tipLength=0.5)

            # Write images
            cv.imwrite(f'{output_path}_{i}.png', img)
            # Show images
            # cv.imshow('Flow', img)
            # cv.waitKey(0)

    def set_parameters(self, params: OpenCVOpticalFlowParams = None):
        """
        Set or update algorithm-specific parameters.
        
        Args:
            params (OpenCVOpticalFlowParams, optional): Parameters to set. If None, use default.
        """
        if params: self.params = params
        else: self.params = OpenCVOpticalFlowParams()

    def save_flow_data(self, file_path: str, flows: List[np.ndarray]) -> None:
        """
        Save the computed flow data to a file.
        
        Args:
            file_path (str): Path to save the flow data.
            flows (List[np.ndarray]): List of computed flows.
        """
        np.save(file_path, flows)

    def load_flow_data(self, file_path: str) -> List[np.ndarray]:
        """
        Load previously saved flow data from a file.
        
        Args:
            file_path (str): Path to load the flow data from.
        """
        return np.load(file_path)