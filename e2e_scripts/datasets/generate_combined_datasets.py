#!/usr/bin/env python3
"""
Generate combined datasets for federated learning

Combines datatest.txt, datatest2.txt, and datatraining.txt
- Extracts 200 samples for inference
- Remaining data split 60/40 for device1 and device2
"""

import os
import csv
import random
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_csv_data(filepath):
    """Load data from CSV file"""
    features = []
    labels = []
    
    logger.info(f"Loading {filepath}...")
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
    
    logger.info(f"  Loaded {len(X)} samples")
    return X, y

def save_csv_data(filepath, X, y):
    """Save data to CSV file with header"""
    logger.info(f"Saving {len(X)} samples to {filepath}...")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['index', 'date', 'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'Occupancy'])
        
        for i, (features, label) in enumerate(zip(X, y)):
            row = [i, '', features[0], features[1], features[2], features[3], features[4], label]
            writer.writerow(row)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load all data files
    datatest_file = os.path.join(script_dir, 'datatest.txt')
    datatest2_file = os.path.join(script_dir, 'datatest2.txt')
    datatraining_file = os.path.join(script_dir, 'datatraining.txt')
    
    logger.info("=" * 80)
    logger.info("Loading datasets...")
    logger.info("=" * 80)
    
    X_test, y_test = load_csv_data(datatest_file)
    X_test2, y_test2 = load_csv_data(datatest2_file)
    X_train, y_train = load_csv_data(datatraining_file)
    
    # Combine all datasets
    X_all = np.concatenate([X_test, X_test2, X_train])
    y_all = np.concatenate([y_test, y_test2, y_train])
    
    logger.info(f"\nTotal combined samples: {len(X_all)}")
    
    # Shuffle the data
    logger.info("\nShuffling data...")
    indices = np.random.permutation(len(X_all))
    X_shuffled = X_all[indices]
    y_shuffled = y_all[indices]
    
    # Extract 200 samples for inference
    X_inference = X_shuffled[:200]
    y_inference = y_shuffled[:200]
    
    logger.info(f"\nInference samples: {len(X_inference)}")
    
    # Remaining data
    X_remaining = X_shuffled[200:]
    y_remaining = y_shuffled[200:]
    
    logger.info(f"Remaining samples: {len(X_remaining)}")
    
    # Split remaining data 60/40 for device1 and device2
    split_idx = int(len(X_remaining) * 0.6)
    
    X_device1 = X_remaining[:split_idx]
    y_device1 = y_remaining[:split_idx]
    
    X_device2 = X_remaining[split_idx:]
    y_device2 = y_remaining[split_idx:]
    
    logger.info(f"\nDevice 1 samples: {len(X_device1)}")
    logger.info(f"Device 2 samples: {len(X_device2)}")
    
    # Save datasets
    logger.info("\n" + "=" * 80)
    logger.info("Saving datasets...")
    logger.info("=" * 80)
    
    save_csv_data(os.path.join(script_dir, 'device1_data.txt'), X_device1, y_device1)
    save_csv_data(os.path.join(script_dir, 'device2_data.txt'), X_device2, y_device2)
    save_csv_data(os.path.join(script_dir, 'inference_data.txt'), X_inference, y_inference)
    
    # Print statistics
    logger.info("\n" + "=" * 80)
    logger.info("Dataset Statistics")
    logger.info("=" * 80)
    
    # Device 1 stats
    device1_occupied = np.sum(y_device1 == 1.0)
    logger.info(f"\nDevice 1:")
    logger.info(f"  Total: {len(y_device1)}")
    logger.info(f"  Occupied: {device1_occupied} ({device1_occupied/len(y_device1)*100:.1f}%)")
    logger.info(f"  Not occupied: {len(y_device1)-device1_occupied} ({100-device1_occupied/len(y_device1)*100:.1f}%)")
    
    # Device 2 stats
    device2_occupied = np.sum(y_device2 == 1.0)
    logger.info(f"\nDevice 2:")
    logger.info(f"  Total: {len(y_device2)}")
    logger.info(f"  Occupied: {device2_occupied} ({device2_occupied/len(y_device2)*100:.1f}%)")
    logger.info(f"  Not occupied: {len(y_device2)-device2_occupied} ({100-device2_occupied/len(y_device2)*100:.1f}%)")
    
    # Inference stats
    inference_occupied = np.sum(y_inference == 1.0)
    logger.info(f"\nInference:")
    logger.info(f"  Total: {len(y_inference)}")
    logger.info(f"  Occupied: {inference_occupied} ({inference_occupied/len(y_inference)*100:.1f}%)")
    logger.info(f"  Not occupied: {len(y_inference)-inference_occupied} ({100-inference_occupied/len(y_inference)*100:.1f}%)")
    
    logger.info("\n✓ Dataset generation completed successfully!")

if __name__ == '__main__':
    main()

