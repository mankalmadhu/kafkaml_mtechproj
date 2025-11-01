"""
Occupancy dataset handler for KafkaML E2E automation
"""

import os
import csv
import numpy as np
import logging
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class OccupancyDataset(BaseDataset):
    """Occupancy detection dataset (binary classification)
    
    Features: 5 sensor readings (Temperature, Humidity, Light, CO2, HumidityRatio)
    Labels: Binary (0=not_occupied, 1=occupied)
    
    Data format:
    - CSV with columns: index, date, Temperature, Humidity, Light, CO2, HumidityRatio, Occupancy
    - Features are normalized using StandardScaler
    """
    
    def __init__(self, data_path: str = None, inference_test_file: str = None):
        """
        Initialize Occupancy dataset handler
        
        Args:
            data_path: Path to occupancy data directory (optional, defaults to datasets/)
            inference_test_file: Name of test file for inference (optional, defaults to 'datatest2.txt')
        """
        self.data_path = data_path
        self.inference_test_file = inference_test_file or 'datatest.txt'
        self._train_cache = None
        self._test_cache = None
        self._inference_cache = None
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
    
    def load_training_data(self, num_samples: int, filename: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load occupancy training data
        
        Args:
            num_samples: Number of training samples to load (use -1 or None for all)
            filename: Optional filename to load from (instead of default datatraining.txt)
            
        Returns:
            (x_train, y_train) - Normalized features (N, 5) and labels (N,)
        """
        # Clear cache if loading from custom file
        should_use_cache = filename is None
        
        if should_use_cache and self._train_cache is not None:
            x_train, y_train = self._train_cache
            if num_samples and num_samples > 0:
                return x_train[:num_samples], y_train[:num_samples]
            return x_train, y_train
        
        # Determine data file path
        if filename:
            if self.data_path:
                train_file = os.path.join(self.data_path, filename)
            else:
                datasets_dir = os.path.dirname(os.path.abspath(__file__))
                train_file = os.path.join(datasets_dir, filename)
        else:
            if self.data_path:
                train_file = os.path.join(self.data_path, 'datatraining.txt')
            else:
                datasets_dir = os.path.dirname(os.path.abspath(__file__))
                train_file = os.path.join(datasets_dir, 'datatraining.txt')
        
        # Load raw data
        X, y = self._load_csv_data(train_file)
        
        # Normalize features using StandardScaler
        self._scaler = StandardScaler()
        X_normalized = self._scaler.fit_transform(X).astype(np.float32)
        
        # Cache only if using default file
        if should_use_cache:
            self._train_cache = (X_normalized, y)
        
        if num_samples and num_samples > 0:
            return X_normalized[:num_samples], y[:num_samples]
        return X_normalized, y
    
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
                test_file = os.path.join(self.data_path, 'datatest2.txt')
            else:
                # Default to datasets/ directory
                datasets_dir = os.path.dirname(os.path.abspath(__file__))
                test_file = os.path.join(datasets_dir, 'datatest2.txt')
            
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
    
    def load_inference_data(self, num_samples: int, filename: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load occupancy test data for inference predictions
        
        Args:
            num_samples: Number of test samples to load
            filename: Optional filename to load from
            
        Returns:
            (x_test, y_test) - Normalized features (N, 5) and labels (N,)
        """
        should_use_cache = filename is None
        
        if should_use_cache and self._inference_cache is not None:
            x_test, y_test = self._inference_cache
            return x_test[:num_samples], y_test[:num_samples]
        
        # Determine data file path
        if filename:
            if self.data_path:
                test_file_path = os.path.join(self.data_path, filename)
            else:
                datasets_dir = os.path.dirname(os.path.abspath(__file__))
                test_file_path = os.path.join(datasets_dir, filename)
        else:
            if self.data_path:
                test_file_path = os.path.join(self.data_path, self.inference_test_file)
            else:
                # Default to datasets/ directory
                datasets_dir = os.path.dirname(os.path.abspath(__file__))
                test_file_path = os.path.join(datasets_dir, self.inference_test_file)
        
        # Load raw data
        X, y = self._load_csv_data(test_file_path)
        
        # Normalize features using the same scaler as training data
        # If scaler not fitted yet, load training data first
        if self._scaler is None:
            self.load_training_data(1)  # Load at least 1 sample to fit scaler
        
        X_normalized = self._scaler.transform(X).astype(np.float32)
        
        if should_use_cache:
            self._inference_cache = (X_normalized, y)
        
        return X_normalized[:num_samples], y[:num_samples]
    
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
    
    def compute_label_weights_from_data(self, y_data: np.ndarray) -> Dict[int, float]:
        """Compute label weights from provided label data
        
        Args:
            y_data: Array of labels to compute weights from
            
        Returns:
            Dictionary mapping class labels to their weights
        """
        # Count class frequencies
        unique_classes, class_counts = np.unique(y_data, return_counts=True)
        
        # Calculate total samples
        total_samples = len(y_data)
        
        # Compute inverse frequency weights
        # Weight = total_samples / (num_classes * class_count)
        num_classes = len(unique_classes)
        class_weights = {}
        
        for class_label, class_count in zip(unique_classes, class_counts):
            weight = total_samples / (num_classes * class_count)
            class_weights[int(class_label)] = float(weight)
        
        return class_weights
    
    def get_class_distribution_from_data(self, y_data: np.ndarray) -> Dict[int, int]:
        """Get class distribution from provided label data
        
        Args:
            y_data: Array of labels to analyze
            
        Returns:
            Dictionary mapping class labels to their counts
        """
        unique_classes, class_counts = np.unique(y_data, return_counts=True)
        return {int(class_label): int(count) for class_label, count in zip(unique_classes, class_counts)}
    
    def load_faulty_train_dataset(self, num_samples: int = None, filename: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load faulty occupancy data from datatraining_faulty.txt and split 80/20 for train/validation
        
        Args:
            num_samples: Optional limit on total samples to load (before splitting)
            
        Returns:
            (x_train, y_train, x_val, y_val) - Training and validation splits (80/20)
        """
        if filename:          
            train_file = os.path.join(self.data_path, filename)
        else:
            train_file = os.path.join(self.data_path, 'datatraining_faulty.txt')
            
        # Check if dataset exists
        if not os.path.exists(train_file):
            logger.error(f"Faulty training dataset not found: {train_file}")
            raise FileNotFoundError(f"Dataset not found: {train_file}")
        
        # Load raw data
        X, y = self._load_csv_data(train_file)
        
        # Limit samples if specified
        if num_samples is not None:
            X, y = X[:num_samples], y[:num_samples]
        
        # Normalize features
        self._scaler = StandardScaler()
        X_normalized = self._scaler.fit_transform(X).astype(np.float32)
        
        
        return X_normalized, y
    
