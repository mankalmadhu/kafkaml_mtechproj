#!/usr/bin/env python3

import json
import numpy as np
from kafka import KafkaProducer
import logging
import tensorflow as tf

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = ['localhost:31162']  # Port-forwarded Kafka
TOPIC = 'FED-DEBUG-data_topic'

def inject_mnist_training_data():
    try:
        logging.info("Creating Kafka producer...")
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS
            # No serializer needed for raw binary data
        )
        logging.info("✅ Kafka producer created successfully")
        
        # Load real MNIST dataset
        logging.info("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Create subset for testing (smaller dataset)
        x_train = x_train[:500]  # 200 training samples
        y_train = y_train[:500]
        
        x_test = x_test[:50]  # 50 test samples
        y_test = y_test[:50]
        
        logging.info(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")
        
        
        # Process training data
        for i, (x, y) in enumerate(zip(x_train, y_train)):
            # Flatten the image (28x28 -> 784)
            x_flat = x.flatten().astype(np.float32) / 255.0  # Normalize to [0,1]
            
            # Convert to one-hot encoded label
            y_one_hot = tf.keras.utils.to_categorical(y, num_classes=10).astype(np.float32)
            
            # Convert to raw binary data for RawDecoder
            # The decoder expects: x = raw image bytes, y = raw label bytes
            image_bytes = x_flat.tobytes()  # Raw binary data (784 * 4 = 3136 bytes)
            label_bytes = y_one_hot.tobytes()  # 1 * 4 = 4 bytes # Raw binary label (10 * 4 = 40 bytes)
            
            # Send image data as value and label data as key
            # This matches what the decoder expects: decoder.decode(message.value, message.key)
            future = producer.send(TOPIC, value=image_bytes, key=label_bytes)
            producer.flush()
            
            # Log progress
            if i % 40 == 0:
                logging.info("Sent training sample %d/%d (class %d)", i + 1, len(x_train), y)
        
        # Process test data
        for i, (x, y) in enumerate(zip(x_test, y_test)):
            # Flatten the image (28x28 -> 784)
            x_flat = x.flatten().astype(np.float32) / 255.0  # Normalize to [0,1]
            
            # Convert to one-hot encoded label
            y_one_hot = tf.keras.utils.to_categorical(y, num_classes=10).astype(np.float32)
            
            # Convert to raw binary data for RawDecoder
            image_bytes = x_flat.tobytes()  # Raw binary data (784 * 4 = 3136 bytes)
            label_bytes = y_one_hot.tobytes()
            
            # Send image data as value and label data as key
            future = producer.send(TOPIC, value=image_bytes, key=label_bytes)
            producer.flush()
            
            # Log progress
            if i % 10 == 0:
                logging.info("Sent test sample %d/%d (class %d)", i + 1, len(x_test), y)
        
        # Check if messages were sent successfully
        producer.flush()
        producer.close()
        total_samples = len(x_train) + len(x_test)
        logging.info("✅ All %d MNIST samples sent successfully!", total_samples)
        
    except Exception as e:
        logging.error("❌ Error injecting MNIST training data: %s", str(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logging.info("=== STARTING MNIST TRAINING DATA INJECTION ===")
    inject_mnist_training_data()
    logging.info("=== MNIST TRAINING DATA INJECTION COMPLETED ===")
