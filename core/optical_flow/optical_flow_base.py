import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Any

flow_registry: dict[str, 'OpticalFlowDetector'] = {}

class OpticalFlowDetector(ABC):
    name = None
    
    def __init__(self):
        pass

    @abstractmethod
    def compute_flow(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute optical flow over a group of N frames.
        
        Args:
            frames (List[np.ndarray]): List of input frames as numpy arrays.
        """
        raise NotImplementedError("Flow computation method not implemented.")

    @abstractmethod
    def group_flows(self, flows: List[np.ndarray]) -> List[List[np.ndarray]]:
        """
        Group flows into event groups.
        """
        raise NotImplementedError("Group flows method not implemented.")
    
    @abstractmethod
    def visualize_flow_groups(self, flow: List[List[np.ndarray]], frames: List[np.ndarray], output_path: str = None) -> List[str]:
        """
        Generate a visual representation of the optical flow.
        
        Args:
            output_path (str, optional): Path to save the visualization. If None, don't save.
        
        Returns:
            np.ndarray: Visualization of the optical flow.
        """
        raise NotImplementedError("Visualization method not implemented.")
    
    @abstractmethod
    def visualize_flow(self, flow: List[np.ndarray], frames: List[np.ndarray], output_path: str = None) -> None:
        """
        Generate a visual representation of the optical flow.
        
        Args:
            output_path (str, optional): Path to save the visualization. If None, don't save.
        
        Returns:
            np.ndarray: Visualization of the optical flow.
        """
        raise NotImplementedError("Visualization method not implemented.")

    @abstractmethod
    def set_parameters(self, **kwargs):
        """
        Set or update algorithm-specific parameters.
        
        Args:
            **kwargs: Arbitrary keyword arguments for parameters.
        """
        raise NotImplementedError("Parameter setting method not implemented.")

    @abstractmethod
    def save_flow_data(self, file_path: str, flows: List[np.ndarray]) -> None:
        """
        Save the computed flow data to a file.
        
        Args:
            file_path (str): Path to save the flow data.
        """
        raise NotImplementedError("Save flow data method not implemented.")

    @abstractmethod
    def load_flow_data(self, file_path: str) -> List[np.ndarray]:
        """
        Load previously saved flow data from a file.
        
        Args:
            file_path (str): Path to load the flow data from.
        """
        raise NotImplementedError("Load flow data method not implemented.")


def register_flow(flow_class: 'OpticalFlowDetector') -> None:
    """Register a flow class with a given name.

    Args:
        flow_name (str): The name to associate with the flow class.
        flow_class (Any): The flow class to register.
    """
    assert issubclass(flow_class, OpticalFlowDetector), "Flow class must inherit from OpticalFlowDetector."
    assert flow_class.name is not None, "Flow class must have a name."
    
    flow_registry[flow_class.name] = flow_class
