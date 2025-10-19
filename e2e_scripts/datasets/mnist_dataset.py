"""
MNIST dataset handler for KafkaML E2E automation
"""

import numpy as np
from typing import Tuple
from .base_dataset import BaseDataset


class MNISTDataset(BaseDataset):
    """MNIST handwritten digits dataset
    
    Loads data from TensorFlow datasets and parses softmax predictions.
    """
    
    def __init__(self):
        """Initialize MNIST dataset handler"""
        self._train_cache = None
        self._test_cache = None
    
    def load_training_data(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load MNIST training data from TensorFlow datasets
        
        Args:
            num_samples: Number of training samples to load
            
        Returns:
            (x_train, y_train) - Training images (28x28) and labels (0-9)
        """
        if self._train_cache is None:
            import tensorflow as tf
            (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
            self._train_cache = (x_train, y_train)
        
        x_train, y_train = self._train_cache
        return x_train[:num_samples], y_train[:num_samples]
    
    def load_test_data(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load MNIST test data from TensorFlow datasets
        
        Args:
            num_samples: Number of test samples to load
            
        Returns:
            (x_test, y_test) - Test images (28x28) and labels (0-9)
        """
        if self._test_cache is None:
            import tensorflow as tf
            _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            self._test_cache = (x_test, y_test)
        
        x_test, y_test = self._test_cache
        return x_test[:num_samples], y_test[:num_samples]
    
    def parse_prediction(self, prediction_obj: dict) -> Tuple[int, float]:
        """Parse MNIST softmax prediction (10 classes)
        
        Args:
            prediction_obj: JSON with 'values' key containing probability array
            
        Returns:
            (predicted_digit, confidence) - Digit 0-9 and max probability
        """
        probs = prediction_obj['values']
        predicted = int(np.argmax(probs))
        confidence = float(max(probs))
        return predicted, confidence

