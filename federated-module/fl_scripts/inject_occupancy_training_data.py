#!/usr/bin/env python3

import json
import numpy as np
from kafka import KafkaProducer
import logging
import csv
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = ['localhost:31162']  # Port-forwarded Kafka
TOPIC = 'FED-OCCUPANCY-data_topic'

def inject_occupancy_training_data():
    try:
        logging.info("Creating Kafka producer...")
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS
            # No serializer needed for raw binary data
        )
        logging.info("✅ Kafka producer created successfully")
        
        # Load occupancy dataset
        logging.info("Loading occupancy dataset...")
        data_file = '/Users/madhuahobalan/workspace/kafkaml/kafka-ml/federated-module/fl_scripts/datatraining.txt'
        
        # Read CSV data
        features = []
        labels = []
        
        with open(data_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            
            for row in reader:
                # Extract features: Temperature, Humidity, Light, CO2, HumidityRatio
                # Skip first two columns (index and date)
                feature_values = [float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6])]
                label_value = int(row[7])  # Occupancy (0 or 1)
                
                features.append(feature_values)
                labels.append(label_value)
        
        # Convert to numpy arrays
        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)
        
        logging.info(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
        logging.info(f"Feature ranges: Temperature={X[:, 0].min():.2f}-{X[:, 0].max():.2f}, "
                    f"Humidity={X[:, 1].min():.2f}-{X[:, 1].max():.2f}, "
                    f"Light={X[:, 2].min():.2f}-{X[:, 2].max():.2f}, "
                    f"CO2={X[:, 3].min():.2f}-{X[:, 3].max():.2f}, "
                    f"HumidityRatio={X[:, 4].min():.6f}-{X[:, 4].max():.6f}")
        logging.info(f"Label distribution: Unoccupied={np.sum(y == 0)}, Occupied={np.sum(y == 1)}")
        
        # Normalize features using StandardScaler
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X).astype(np.float32)
        
        logging.info("Features normalized using StandardScaler")
        
        # Create subset for testing (smaller dataset)
        # Use first 500 samples for training, next 50 for testing
        X_train = X_normalized[:500]
        y_train = y[:500]
        
        X_test = X_normalized[500:550]  # 50 test samples
        y_test = y[500:550]
        
        logging.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Process training data
        for i, (x, y_label) in enumerate(zip(X_train, y_train)):
            # Convert features to raw binary data
            # 5 features * 4 bytes (float32) = 20 bytes
            feature_bytes = x.tobytes()
            
            # Convert label to raw binary data
            # 1 label * 4 bytes (float32) = 4 bytes
            label_bytes = np.array([y_label], dtype=np.float32).tobytes()
            
            # Send feature data as value and label data as key
            # This matches what the decoder expects: decoder.decode(message.value, message.key)
            future = producer.send(TOPIC, value=feature_bytes, key=label_bytes)
            producer.flush()
            
            # Log progress
            if i % 50 == 0:
                logging.info("Sent training sample %d/%d (occupancy: %d)", i + 1, len(X_train), int(y_label))
        
        # Process test data
        for i, (x, y_label) in enumerate(zip(X_test, y_test)):
            # Convert features to raw binary data
            feature_bytes = x.tobytes()
            
            # Convert label to raw binary data
            label_bytes = np.array([y_label], dtype=np.float32).tobytes()
            
            # Send feature data as value and label data as key
            future = producer.send(TOPIC, value=feature_bytes, key=label_bytes)
            producer.flush()
            
            # Log progress
            if i % 10 == 0:
                logging.info("Sent test sample %d/%d (occupancy: %d)", i + 1, len(X_test), int(y_label))
        
        # Check if messages were sent successfully
        producer.flush()
        producer.close()
        total_samples = len(X_train) + len(X_test)
        logging.info("✅ All %d occupancy samples sent successfully!", total_samples)
        
        # Log data format information
        logging.info("📊 Data format summary:")
        logging.info("   - Features: 5 values (Temperature, Humidity, Light, CO2, HumidityRatio)")
        logging.info("   - Feature size: 5 * 4 bytes = 20 bytes per sample")
        logging.info("   - Labels: 1 value (Occupancy: 0 or 1)")
        logging.info("   - Label size: 1 * 4 bytes = 4 bytes per sample")
        logging.info("   - Total message size: 24 bytes per sample")
        logging.info("   - Topic: %s", TOPIC)
        
    except Exception as e:
        logging.error("❌ Error injecting occupancy training data: %s", str(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logging.info("=== OCCUPANCY TRAINING DATA INJECTION STARTED ===")
    inject_occupancy_training_data()
    logging.info("=== OCCUPANCY TRAINING DATA INJECTION COMPLETED ===")
