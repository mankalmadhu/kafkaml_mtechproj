"""
Base dataset class for KafkaML E2E automation
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any
import numpy as np


class BaseDataset(ABC):
    """Abstract base class for datasets
    
    Datasets are responsible for:
    1. Loading training and test data
    2. Parsing model predictions
    
    All data configuration (types, shapes, restrictions) stays in YAML config.
    """
    
    @abstractmethod
    def load_training_data(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load training data
        
        Args:
            num_samples: Number of samples to load
            
        Returns:
            (x_train, y_train) - Training features and labels
        """
        pass
    
    @abstractmethod
    def load_test_data(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load test data for inference
        
        Args:
            num_samples: Number of samples to load
            
        Returns:
            (x_test, y_test) - Test features and labels
        """
        pass
    
    @abstractmethod
    def parse_prediction(self, prediction_obj: dict) -> Tuple[Any, float]:
        """Parse prediction output from inference
        
        Args:
            prediction_obj: JSON object from Kafka output topic
            
        Returns:
            (predicted_value, confidence) - Predicted class/value and confidence score
        """
        pass

