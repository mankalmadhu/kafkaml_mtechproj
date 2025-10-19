"""
Dataset handlers for KafkaML E2E automation

Available datasets:
- MNISTDataset: MNIST handwritten digits
- OccupancyDataset: Occupancy detection (binary classification)
"""

from .base_dataset import BaseDataset
from .mnist_dataset import MNISTDataset
from .occupancy_dataset import OccupancyDataset

__all__ = ['BaseDataset', 'MNISTDataset', 'OccupancyDataset']

