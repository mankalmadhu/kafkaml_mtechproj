"""
Aggregation strategies for federated learning
Bare minimum implementation focusing on core aggregation logic
"""

import numpy as np
import logging

class AggregationStrategy:
    """Base class for aggregation strategies"""
    
    def aggregate(self, global_weights: list, device_weights: list) -> list:
        """
        Aggregate device weights into global weights
        
        Args:
            global_weights: Current global model weights
            device_weights: List of device model weights (each is List[np.ndarray])
            
        Returns:
            New global model weights
        """
        raise NotImplementedError


class FedAvgStrategy(AggregationStrategy):
    """Standard Federated Averaging - simple average of all models"""
    
    def aggregate(self, global_weights: list, device_weights: list) -> list:
        """
        Average global model and all device models
        
        Args:
            global_weights: List of numpy arrays (global model)
            device_weights: List of model weights (each model is List[np.ndarray])
            
        Returns:
            Averaged weights (List[np.ndarray])
        """
        if not device_weights:
            return global_weights
        
        # Collect all models: [global, device1, device2, ...]
        all_models = [global_weights] + device_weights
        
        # Average each layer across all models
        new_weights = []
        for layer_idx in range(len(global_weights)):
            # Stack all models' weights for this layer
            layer_weights = [model[layer_idx] for model in all_models]
            # Average
            averaged = np.mean(layer_weights, axis=0)
            new_weights.append(averaged)
        
        logging.info(f"FedAvg: Averaged {len(all_models)} models")
        return new_weights


class KrumStrategy(AggregationStrategy):
    """Krum: Select most reliable model (closest to consensus)"""
    
    def aggregate(self, global_weights: list, device_weights: list) -> list:
        """
        Select the model most similar to others (most reliable)
        
        Args:
            global_weights: Global model weights
            device_weights: List of device model weights
            
        Returns:
            Selected model weights
        """
        if not device_weights:
            return global_weights
        
        # Fall back to FedAvg if not enough devices
        if len(device_weights) < 2:
            logging.warning("Not enough devices for Krum, using FedAvg")
            return FedAvgStrategy().aggregate(global_weights, device_weights)
        
        # Collect all models: [global, device1, device2, ...]
        all_models = [global_weights] + device_weights
        
        # Calculate distances
        distances = self._calculate_distances(all_models)
        
        # Select model with minimum sum of distances (most similar to others)
        selected_idx = np.argmin([np.sum(distances[i]) for i in range(len(all_models))])
        
        logging.info(f"Krum: Selected model {selected_idx} from {len(all_models)} models")
        return all_models[selected_idx]
    
    def _calculate_distances(self, models: list) -> np.ndarray:
        """Calculate pairwise distances between models"""
        n = len(models)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._model_distance(models[i], models[j])
                distances[i][j] = distances[j][i] = dist
        
        return distances
    
    def _model_distance(self, weights1: list, weights2: list) -> float:
        """Calculate Euclidean distance between two models"""
        total = 0.0
        for w1, w2 in zip(weights1, weights2):
            total += np.linalg.norm(w1.flatten() - w2.flatten())
        return total


class FedAvgPlusStrategy(AggregationStrategy):
    """FedAvg+: Same as Krum for now (can be enhanced later)"""
    
    def aggregate(self, global_weights: list, device_weights: list) -> list:
        """
        Use Krum to select most reliable model
        TODO: Enhance to weighted average of selected devices
        """
        logging.info("FedAvg+: Using Krum selection")
        return KrumStrategy().aggregate(global_weights, device_weights)


def get_aggregation_strategy(strategy_name: str) -> AggregationStrategy:
    """
    Factory function to get aggregation strategy
    
    Args:
        strategy_name: Name of the strategy ('FedAvg', 'FedAvg+', 'Krum')
        
    Returns:
        AggregationStrategy instance
    """
    if strategy_name == 'FedAvg':
        return FedAvgStrategy()
    elif strategy_name == 'FedAvg+':
        return FedAvgPlusStrategy()
    elif strategy_name == 'Krum':
        return KrumStrategy()
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy_name}")
