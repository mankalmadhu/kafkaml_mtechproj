"""
Occupancy dataset handler for KafkaML E2E automation
"""

import os
import csv
import numpy as np
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler
from .base_dataset import BaseDataset


class OccupancyDataset(BaseDataset):
    """Occupancy detection dataset (binary classification)
    
    Features: 5 sensor readings (Temperature, Humidity, Light, CO2, HumidityRatio)
    Labels: Binary (0=not_occupied, 1=occupied)
    
    Data format:
    - CSV with columns: index, date, Temperature, Humidity, Light, CO2, HumidityRatio, Occupancy
    - Features are normalized using StandardScaler
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize Occupancy dataset handler
        
        Args:
            data_path: Path to occupancy data directory (optional, defaults to datasets/)
        """
        self.data_path = data_path
        self._train_cache = None
        self._test_cache = None
        self._scaler = None
    
    def _load_csv_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from CSV file
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            (features, labels) as numpy arrays
        """
        features = []
        labels = []
        
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            
            for row in reader:
                # Extract features: Temperature, Humidity, Light, CO2, HumidityRatio (columns 2-6)
                feature_values = [float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6])]
                # Extract label: Occupancy (column 7)
                label_value = float(row[7])
                
                features.append(feature_values)
                labels.append(label_value)
        
        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)
        
        return X, y
    
    def load_training_data(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load occupancy training data
        
        Args:
            num_samples: Number of training samples to load
            
        Returns:
            (x_train, y_train) - Normalized features (N, 5) and labels (N,)
        """
        if self._train_cache is None:
            # Determine data file path
            if self.data_path:
                train_file = os.path.join(self.data_path, 'datatraining.txt')
            else:
                # Default to datasets/ directory
                datasets_dir = os.path.dirname(os.path.abspath(__file__))
                train_file = os.path.join(datasets_dir, 'datatraining.txt')
            
            # Load raw data
            X, y = self._load_csv_data(train_file)
            
            # Normalize features using StandardScaler
            self._scaler = StandardScaler()
            X_normalized = self._scaler.fit_transform(X).astype(np.float32)
            
            self._train_cache = (X_normalized, y)
        
        x_train, y_train = self._train_cache
        return x_train[:num_samples], y_train[:num_samples]
    
    def load_test_data(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load occupancy test data
        
        Args:
            num_samples: Number of test samples to load
            
        Returns:
            (x_test, y_test) - Normalized features (N, 5) and labels (N,)
        """
        if self._test_cache is None:
            # Determine data file path
            if self.data_path:
                test_file = os.path.join(self.data_path, 'datatest.txt')
            else:
                # Default to datasets/ directory
                datasets_dir = os.path.dirname(os.path.abspath(__file__))
                test_file = os.path.join(datasets_dir, 'datatest.txt')
            
            # Load raw data
            X, y = self._load_csv_data(test_file)
            
            # Normalize features using the same scaler as training data
            # If scaler not fitted yet, load training data first
            if self._scaler is None:
                self.load_training_data(1)  # Load at least 1 sample to fit scaler
            
            X_normalized = self._scaler.transform(X).astype(np.float32)
            
            self._test_cache = (X_normalized, y)
        
        x_test, y_test = self._test_cache
        return x_test[:num_samples], y_test[:num_samples]
    
    def parse_prediction(self, prediction_obj: dict) -> Tuple[int, float]:
        """Parse occupancy prediction (binary classification with sigmoid)
        
        Args:
            prediction_obj: JSON with 'values' key containing probability
            
        Returns:
            (predicted_class, confidence) - 0 or 1, and confidence score
        
        For sigmoid output:
        - values[0] = probability of class 1 (occupied)
        - If prob > 0.5 -> class 1 (occupied), confidence = prob
        - If prob <= 0.5 -> class 0 (not occupied), confidence = 1 - prob
        """
        # Sigmoid outputs a single value (probability of positive class)
        prob_occupied = prediction_obj['values'][0]
        
        if prob_occupied > 0.5:
            predicted = 1  # Occupied
            confidence = prob_occupied
        else:
            predicted = 0  # Not occupied
            confidence = 1.0 - prob_occupied
        
        return int(predicted), float(confidence)
    
    def compute_label_weights(self, num_samples: int = None) -> Dict[int, float]:
        """Compute label weights based on inverse class frequency
        
        This method calculates weights to balance class distribution by giving
        higher weights to minority classes and lower weights to majority classes.
        
        Args:
            num_samples: Number of samples to use for weight calculation (None = use all)
            
        Returns:
            Dictionary mapping class labels to their weights
            e.g., {0: 2.5, 1: 0.8} means class 0 gets 2.5x weight, class 1 gets 0.8x weight
        """
        # Load training data to compute class frequencies
        if num_samples is None:
            # Use all available training data
            x_train, y_train = self.load_training_data(100000)  # Large number to get all data
        else:
            x_train, y_train = self.load_training_data(num_samples)
        
        # Count class frequencies
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        
        # Calculate total samples
        total_samples = len(y_train)
        
        # Compute inverse frequency weights
        # Weight = total_samples / (num_classes * class_count)
        num_classes = len(unique_classes)
        class_weights = {}
        
        for class_label, class_count in zip(unique_classes, class_counts):
            # Inverse frequency weight: more samples = lower weight
            weight = total_samples / (num_classes * class_count)
            class_weights[int(class_label)] = float(weight)
        
        return class_weights
    
    def get_class_distribution(self, num_samples: int = None) -> Dict[int, int]:
        """Get class distribution for analysis
        
        Args:
            num_samples: Number of samples to analyze (None = use all)
            
        Returns:
            Dictionary mapping class labels to their counts
        """
        # Load training data
        if num_samples is None:
            x_train, y_train = self.load_training_data(100000)  # Large number to get all data
        else:
            x_train, y_train = self.load_training_data(num_samples)
        
        # Count class frequencies
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        
        return {int(class_label): int(count) for class_label, count in zip(unique_classes, class_counts)}
