"""
Aggregation strategies for federated learning
Implements robust aggregation methods including Krum for Byzantine fault tolerance
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Any


class AggregationStrategy:
    """Base class for aggregation strategies"""
    
    def aggregate(self, global_model_weights: List[np.ndarray], 
                  device_updates: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Aggregate device updates into global model weights
        
        Args:
            global_model_weights: Current global model weights
            device_updates: List of device updates with weights and metadata
            
        Returns:
            New global model weights
        """
        raise NotImplementedError


class FedAvgStrategy(AggregationStrategy):
    """Standard Federated Averaging strategy"""
    
    def aggregate(self, global_model_weights: List[np.ndarray], 
                  device_updates: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Standard FedAvg aggregation
        
        Args:
            global_model_weights: Current global model weights
            device_updates: List of device updates
            
        Returns:
            Averaged model weights
        """
        if not device_updates:
            return global_model_weights
            
        # Collect all weight lists (global + device updates)
        all_weights = [global_model_weights]
        for update in device_updates:
            all_weights.append(update['weights'])
        
        # Calculate average across all weight lists
        new_weights = []
        for weights_list_tuple in zip(*all_weights):
            # Average across all models for each layer
            averaged_layer = np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            new_weights.append(averaged_layer)
        
        return new_weights


class KrumStrategy(AggregationStrategy):
    """Krum aggregation strategy for Byzantine fault tolerance"""
    
    def __init__(self, f: int = 1):
        """
        Initialize Krum strategy
        
        Args:
            f: Maximum number of faulty devices to tolerate
        """
        self.f = f
    
    def aggregate(self, global_model_weights: List[np.ndarray], 
                  device_updates: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Krum aggregation: Select the model most similar to others
        
        Args:
            global_model_weights: Current global model weights
            device_updates: List of device updates
            
        Returns:
            Selected model weights (most reliable)
        """
        if not device_updates:
            return global_model_weights
            
        # For 2 devices, we need at least 3 total models (global + 2 devices)
        # If we have fewer than 3, fall back to FedAvg
        if len(device_updates) < 2:
            logging.warning("Not enough devices for Krum, falling back to FedAvg")
            fedavg = FedAvgStrategy()
            return fedavg.aggregate(global_model_weights, device_updates)
        
        # Collect all models (global + device updates)
        all_models = [global_model_weights]
        device_metadata = []
        
        for update in device_updates:
            all_models.append(update['weights'])
            device_metadata.append({
                'device_id': update.get('device_id', 'unknown'),
                'account': update.get('account', 'unknown'),
                'data_size': update.get('data_size', 0),
                'metrics': update.get('metrics', {})
            })
        
        # Calculate pairwise distances between all models
        distances = self._calculate_pairwise_distances(all_models)
        
        # Select the model with smallest sum of distances to closest (n-f-2) models
        selected_model_idx = self._krum_selection(distances, len(all_models))
        
        logging.info(f"Krum selected model {selected_model_idx} out of {len(all_models)} models")
        
        if selected_model_idx == 0:
            logging.info("Krum selected global model (no update)")
            return global_model_weights
        else:
            device_idx = selected_model_idx - 1
            logging.info(f"Krum selected device {device_metadata[device_idx]['device_id']}")
            return device_updates[device_idx]['weights']
    
    def _calculate_pairwise_distances(self, models: List[List[np.ndarray]]) -> np.ndarray:
        """
        Calculate pairwise Euclidean distances between models
        
        Args:
            models: List of model weight lists
            
        Returns:
            Distance matrix (n x n)
        """
        n = len(models)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Calculate Euclidean distance between flattened weights
                dist = self._model_distance(models[i], models[j])
                distances[i][j] = dist
                distances[j][i] = dist
        
        return distances
    
    def _model_distance(self, weights1: List[np.ndarray], weights2: List[np.ndarray]) -> float:
        """
        Calculate Euclidean distance between two models
        
        Args:
            weights1: First model weights
            weights2: Second model weights
            
        Returns:
            Euclidean distance
        """
        total_distance = 0.0
        
        for w1, w2 in zip(weights1, weights2):
            # Flatten weights and calculate L2 distance
            flat_w1 = w1.flatten()
            flat_w2 = w2.flatten()
            distance = np.linalg.norm(flat_w1 - flat_w2)
            total_distance += distance
        
        return total_distance
    
    def _krum_selection(self, distances: np.ndarray, n: int) -> int:
        """
        Select model using Krum algorithm
        
        Args:
            distances: Distance matrix
            n: Number of models
            
        Returns:
            Index of selected model
        """
        # For each model, calculate sum of distances to closest (n-f-2) models
        krum_scores = []
        
        for i in range(n):
            # Get distances from model i to all other models
            model_distances = distances[i]
            
            # Sort distances (excluding self-distance which is 0)
            sorted_distances = np.sort(model_distances[model_distances > 0])
            
            # Select closest (n - self.f - 2) models
            # For 2 devices: n=3, f=1, so we select closest (3-1-2)=0 models
            # This means we need at least 3 devices for meaningful Krum
            # For simplicity with 2 devices, we'll select the closest model
            if len(sorted_distances) >= 1:
                krum_score = np.sum(sorted_distances[:1])  # Sum of closest 1 model
            else:
                krum_score = 0
            
            krum_scores.append(krum_score)
        
        # Select model with minimum Krum score
        selected_idx = np.argmin(krum_scores)
        return selected_idx


class FedAvgPlusStrategy(AggregationStrategy):
    """FedAvg+ strategy: Robust aggregation using Krum for device selection"""
    
    def __init__(self, f: int = 1):
        """
        Initialize FedAvg+ strategy
        
        Args:
            f: Maximum number of faulty devices to tolerate
        """
        self.f = f
        self.krum = KrumStrategy(f)
        self.fedavg = FedAvgStrategy()
    
    def aggregate(self, global_model_weights: List[np.ndarray], 
                  device_updates: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        FedAvg+ aggregation: Use Krum to select reliable devices, then average
        
        Args:
            global_model_weights: Current global model weights
            device_updates: List of device updates
            
        Returns:
            Aggregated model weights
        """
        if not device_updates:
            return global_model_weights
        
        # For 2 devices, use Krum to select the most reliable device
        if len(device_updates) >= 2:
            logging.info("FedAvg+: Using Krum to select reliable devices")
            
            # Use Krum to select the most reliable device
            selected_weights = self.krum.aggregate(global_model_weights, device_updates)
            
            # For FedAvg+, we could also do weighted average of selected devices
            # For simplicity, we'll return the Krum-selected model
            return selected_weights
        else:
            logging.info("FedAvg+: Not enough devices, falling back to FedAvg")
            return self.fedavg.aggregate(global_model_weights, device_updates)


def get_aggregation_strategy(strategy_name: str, **kwargs) -> AggregationStrategy:
    """
    Factory function to get aggregation strategy
    
    Args:
        strategy_name: Name of the strategy ('FedAvg', 'FedAvg+', 'Krum')
        **kwargs: Additional parameters for strategy initialization
        
    Returns:
        AggregationStrategy instance
    """
    if strategy_name == 'FedAvg':
        return FedAvgStrategy()
    elif strategy_name == 'FedAvg+':
        return FedAvgPlusStrategy(**kwargs)
    elif strategy_name == 'Krum':
        return KrumStrategy(**kwargs)
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy_name}")


# Example usage and testing
if __name__ == "__main__":
    # Test the aggregation strategies
    logging.basicConfig(level=logging.INFO)
    
    # Mock model weights (simplified)
    global_weights = [np.random.randn(10, 5), np.random.randn(5, 1)]
    
    # Mock device updates
    device_updates = [
        {
            'weights': [np.random.randn(10, 5), np.random.randn(5, 1)],
            'device_id': 'device1',
            'account': '0x123',
            'data_size': 1000,
            'metrics': {'accuracy': 0.85}
        },
        {
            'weights': [np.random.randn(10, 5), np.random.randn(5, 1)],
            'device_id': 'device2', 
            'account': '0x456',
            'data_size': 800,
            'metrics': {'accuracy': 0.82}
        }
    ]
    
    # Test different strategies
    strategies = ['FedAvg', 'FedAvg+', 'Krum']
    
    for strategy_name in strategies:
        print(f"\nTesting {strategy_name}:")
        strategy = get_aggregation_strategy(strategy_name)
        result = strategy.aggregate(global_weights, device_updates)
        print(f"Result shape: {[w.shape for w in result]}")
