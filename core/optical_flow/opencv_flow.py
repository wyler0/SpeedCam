# © 2024 Wyler Zahm. All rights reserved.

from typing import List, Tuple
from pydantic import BaseModel
from sklearn.cluster import DBSCAN
import numpy as np
from scipy.spatial.distance import cdist

import numpy as np
import cv2 as cv

from core.optical_flow.optical_flow_base import OpticalFlowDetector, register_flow

class OpenCVOpticalFlowParamsFarnbeck(BaseModel):
    MIN_FLOW_MAGNITUDE: float = 4 # Minimum magnitude of flow to be considered. Dependent on vehicle speed, flow compute params, etc.
    MIN_DISTANCE_BETWEEN_GROUPS: float = 150 # Minimum distance between groups to be considered separate events.
    MAX_DISTANCE_FROM_CENTER: float = 400 # Maximum distance from center to form an event. Related to vehicle size.
    MIN_FLOW_COUNT: int = 20 # Minimum number of flows to form an event.

    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 25
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0
    
@register_flow
class OpenCVOpticalFlowDetectorFarnbeck(OpticalFlowDetector):
    name = 'opencv_farneback'
    params: OpenCVOpticalFlowParamsFarnbeck = OpenCVOpticalFlowParamsFarnbeck()
    
    def __init__(self, params: OpenCVOpticalFlowParamsFarnbeck = None):
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
            
            # 4. Filter groups based on direction and location
            groups = self.filter_groups_by_direction_and_location(groups)
        
            final_groups.append(groups)
        
        return final_groups
    
    def filter_groups_by_direction_and_location(self, groups):
        """
        Filter groups based on direction and location using efficient NumPy operations.
        """
        CENTROID_THRESHOLD = 50  # Adjust as needed
        LOWER_LINE_Y = 600  # Adjust as needed
        UPPER_LINE_Y = 600  # Adjust as needed
        
        left_groups = [g for g in groups if g[0][2] < 0]
        right_groups = [g for g in groups if g[0][2] >= 0]
        
        if not left_groups or not right_groups:
            return groups

        left_centroids = np.array([np.mean(g[:, :2], axis=0) for g in left_groups])
        right_centroids = np.array([np.mean(g[:, :2], axis=0) for g in right_groups])
        
        # Calculate pairwise distances between left and right centroids
        distances = cdist(left_centroids, right_centroids)
        
        # Find pairs of groups with centroids within the threshold
        close_pairs = np.argwhere(distances < CENTROID_THRESHOLD)
        
        left_to_remove = set()
        right_to_remove = set()
        
        for left_idx, right_idx in close_pairs:
            left_bottom = np.max(left_groups[left_idx][:, 1])
            right_bottom = np.max(right_groups[right_idx][:, 1])
            
            if left_bottom < UPPER_LINE_Y and right_bottom < UPPER_LINE_Y:
                left_to_remove.add(left_idx)
            elif right_bottom > LOWER_LINE_Y:
                right_to_remove.add(right_idx)
        
        filtered_left_groups = [g for i, g in enumerate(left_groups) if i not in left_to_remove]
        filtered_right_groups = [g for i, g in enumerate(right_groups) if i not in right_to_remove]
        
        return filtered_left_groups + filtered_right_groups

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
                    # Increase arrow size by multiplying dx and dy
                    cv.arrowedLine(overlay, (x, y), (x + dx*2, y + dy*2), color, 2, tipLength=0.3)

            # Reduce opacity for the overlay
            opacity = 0.4  # Adjust this value as needed (0.0 to 1.0)
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

    def set_parameters(self, params: OpenCVOpticalFlowParamsFarnbeck = None):
        """
        Set or update algorithm-specific parameters.
        
        Args:
            params (OpenCVOpticalFlowParams, optional): Parameters to set. If None, use default.
        """
        if params: self.params = params
        else: self.params = OpenCVOpticalFlowParamsFarnbeck()

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
    
  
class OpenCVOpticalFlowParamsLucasKanade(BaseModel):
    MIN_FLOW_MAGNITUDE: float = 1 # Minimum magnitude of flow to be considered. Dependent on vehicle speed, flow compute params, etc.
    MIN_DISTANCE_BETWEEN_GROUPS: float = 150 # Minimum distance between groups to be considered separate events.
    MAX_DISTANCE_FROM_CENTER: float = 300 # Maximum distance from center to form an event. Related to vehicle size.
    MIN_FLOW_COUNT: int = 5 # Minimum number of flows to form an event.
    
    # CENTROID_THRESHOLD = 100  # Adjust as needed
    # LOWER_LINE_Y = 600  # 
    # UPPER_LINE_Y = 600  # Adjust as needed
    
    winsize: int = 15
    maxLevel: int = 3
    criteria: Tuple[int, int, float] = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)

    X_WEIGHT: float = 0.3  # Weight for x-coordinate in distance calculation
    Y_WEIGHT: float = 0.7  # Weight for y-coordinate in distance calculation

    
    
@register_flow
class OpenCVOpticalFlowDetectorLucasKanade(OpticalFlowDetector):
    name = 'opencv_lucaskanade'
    params: OpenCVOpticalFlowParamsLucasKanade = OpenCVOpticalFlowParamsLucasKanade()
    
    def __init__(self, params: OpenCVOpticalFlowParamsFarnbeck = None):
        self.set_parameters(params)

    def compute_flow(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute optical flow over a group of N frames using Lucas-Kanade method.
        
        Args:
            frames (List[np.ndarray]): List of input frames as numpy arrays.
        """
        if len(frames) < 2:
            raise ValueError("At least 2 frames are required to compute optical flow.")

        prvs = cv.cvtColor(frames[0], cv.COLOR_BGR2GRAY)
        
        # Parameters for Lucas-Kanade optical flow
        lk_params = dict(winSize=(self.params.winsize, self.params.winsize),
                         maxLevel=self.params.maxLevel,
                         criteria=self.params.criteria)


        # Create some random points to track
        p0 = cv.goodFeaturesToTrack(prvs, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

        flows = []
        for i in range(1, len(frames)):
            next_frame = cv.cvtColor(frames[i], cv.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            p1, st, err = cv.calcOpticalFlowPyrLK(prvs, next_frame, p0, None, **lk_params)
            
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            
            # Calculate flow vectors
            flow_vectors = good_new - good_old
            
            # Store as vectors: [x, y, dx, dy]
            vectors = np.column_stack((good_old, flow_vectors))
            
            flows.append(vectors)
            
            # Update the previous frame and previous points
            prvs = next_frame.copy()
            p0 = good_new.reshape(-1, 1, 2)

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
                
                # X. Filter groups based on direction and location
                # groups = self.filter_groups_by_direction_and_location(groups)
            
            final_groups.append(groups)
        
        return final_groups
    
    # def filter_groups_by_direction_and_location(self, groups):
    #     """
    #     Filter groups based on direction and location to remove similar groups in opposite directions.
    #     """
        
    #     left_groups = [g for g in groups if g[0][2] < 0]
    #     right_groups = [g for g in groups if g[0][2] >= 0]
        
    #     if not left_groups or not right_groups:
    #         return groups

    #     left_centroids = np.array([np.mean(g[:, :2], axis=0) for g in left_groups])
    #     right_centroids = np.array([np.mean(g[:, :2], axis=0) for g in right_groups])
        
    #     # Calculate pairwise distances between left and right centroids
    #     distances = cdist(left_centroids, right_centroids)
        
    #     # Find pairs of groups with centroids within the threshold
    #     close_pairs = np.argwhere(distances < self.params.CENTROID_THRESHOLD)
        
    #     left_to_remove = set()
    #     right_to_remove = set()
        
    #     for left_idx, right_idx in close_pairs:
    #         left_bottom = np.min(left_groups[left_idx][:, 1])
    #         right_bottom = np.min(right_groups[right_idx][:, 1])
            
    #         if left_bottom < self.params.UPPER_LINE_Y and right_bottom < self.params.UPPER_LINE_Y:
    #             left_to_remove.add(left_idx)
    #         elif right_bottom > self.params.LOWER_LINE_Y:
    #             right_to_remove.add(right_idx)
        
    #     filtered_left_groups = [g for i, g in enumerate(left_groups) if i not in left_to_remove]
    #     filtered_right_groups = [g for i, g in enumerate(right_groups) if i not in right_to_remove]
        
    #     return filtered_left_groups + filtered_right_groups

    def cluster_by_proximity(self, flows: np.ndarray) -> List[np.ndarray]:
        """
        Cluster flows based on spatial proximity, with more flexibility on the x-axis.
        Ensures each flow is only added to one group.
        """
        if len(flows) == 0:
            return []

        # Calculate weighted coordinates
        weighted_coords = np.column_stack((
            flows[:, 0] * self.params.X_WEIGHT,
            flows[:, 1] * self.params.Y_WEIGHT
        ))

        # Use DBSCAN for clustering
        dbscan = DBSCAN(
            eps=self.params.MAX_DISTANCE_FROM_CENTER,
            min_samples=self.params.MIN_FLOW_COUNT,
            metric='euclidean'
        )
        cluster_labels = dbscan.fit_predict(weighted_coords)

        # Group flows by cluster label
        clusters = []
        for label in set(cluster_labels):
            if label != -1:  # -1 is the label for noise points
                cluster = flows[cluster_labels == label]
                clusters.append(cluster)

        return clusters
    
    def visualize_flow_groups(self, flow_groups: List[List[np.ndarray]], frames: List[np.ndarray], output_path: str = None, opacity: float = 0.8) -> List[str]:
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
                color = colors[c % len(colors)]
                for x, y, dx, dy in flow:
                    x, y, dx, dy = map(int, [x, y, dx, dy])
                    # Increase arrow size by multiplying dx and dy
                    cv.arrowedLine(overlay, (x, y), (x + dx*2, y + dy*2), color, 2, tipLength=0.3)

            # Reduce opacity for the overlay
            opacity = 0.4  # Adjust this value as needed (0.0 to 1.0)
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

    def set_parameters(self, params: OpenCVOpticalFlowParamsLucasKanade = None):
        """
        Set or update algorithm-specific parameters.
        
        Args:
            params (OpenCVOpticalFlowParams, optional): Parameters to set. If None, use default.
        """
        if params: self.params = params
        else: self.params = OpenCVOpticalFlowParamsLucasKanade()

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
    
  