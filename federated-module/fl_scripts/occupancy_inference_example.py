import tensorflow as tf
import logging
from kafka import KafkaProducer, KafkaConsumer
import numpy as np
import json

logging.basicConfig(level=logging.INFO)

INPUT_TOPIC = 'occupancy_inf_in'
OUTPUT_TOPIC = 'occupancy_inf_out'
BOOTSTRAP_SERVERS = '127.0.0.1:9094'
ITEMS_TO_PREDICT = 10
data_file = '/Users/madhuahobalan/workspace/kafkaml/kafka-ml/federated-module/fl_scripts/datatest.txt'



# Generate random occupancy data for prediction
# Features: Temperature, Humidity, Light, CO2, HumidityRatio
# Based on typical occupancy sensor ranges
def load_occupancy_data_from_file(filename, num_samples):
    """Load occupancy sensor data for inference from a CSV file"""
    import csv
    import random

    # Read all rows from the file, skipping the header
    with open(filename, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # Randomly select num_samples rows
    random.seed(42)
    selected_rows = random.sample(rows, num_samples)

    # Extract features: Temperature, Humidity, Light, CO2, HumidityRatio
    features = []
    for row in selected_rows:
        # The columns are: "date","Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy"
        # Indices:         0      1            2         3      4     5               6
        temperature = float(row[2])
        humidity = float(row[3])
        light = float(row[4])
        co2 = float(row[5])
        humidity_ratio = float(row[6])
        features.append([temperature, humidity, light, co2, humidity_ratio])

    return np.array(features, dtype=np.float32)

# Load data for inference from datatest.txt
occupancy_data = load_occupancy_data_from_file(data_file, ITEMS_TO_PREDICT)
print(f"Generated {ITEMS_TO_PREDICT} occupancy samples for inference")
print("Sample data shape:", occupancy_data.shape)
print("Sample features (Temperature, Humidity, Light, CO2, HumidityRatio):")
for i, sample in enumerate(occupancy_data[:3]):
    print(f"  Sample {i+1}: {sample}")

# Create producer to send data for prediction
producer = KafkaProducer(bootstrap_servers=BOOTSTRAP_SERVERS)
"""Creates a producer to send the occupancy data to predict"""

# Send each sample for prediction
for i, sample in enumerate(occupancy_data):
    # Convert to raw bytes (5 features * 4 bytes = 20 bytes)
    sample_bytes = sample.tobytes()
    producer.send(INPUT_TOPIC, sample_bytes)
    print(f"Sent sample {i+1} to {INPUT_TOPIC}")
    
producer.flush()
producer.close()
print(f"All {ITEMS_TO_PREDICT} samples sent to {INPUT_TOPIC}")

# Create output consumer to receive predictions
output_consumer = KafkaConsumer(
    OUTPUT_TOPIC, 
    bootstrap_servers=BOOTSTRAP_SERVERS, 
    group_id="occupancy_output_group",
    auto_offset_reset='latest'
)
"""Creates an output consumer to receive the occupancy predictions"""

print('\nWaiting for predictions...')
print('Output consumer: ')

# Collect predictions
predictions = []
for i, msg in enumerate(output_consumer):
    if i >= ITEMS_TO_PREDICT:
        break
        
    # Decode prediction (assuming it's a JSON-encoded object with a 'values' field)
    import json
    prediction_bytes = msg.value
    try:
        prediction_json = json.loads(prediction_bytes.decode('utf-8'))
        # Handle if 'values' is a list or a single value
        prediction_value = prediction_json['values']
        if isinstance(prediction_value, list):
            prediction = prediction_value[0]
        else:
            prediction = prediction_value
    except Exception as e:
        print(f"Error decoding prediction: {e}")
        prediction = None
    predictions.append(prediction)
    
    # Convert to occupancy status
    occupancy_status = "Occupied" if prediction > 0.5 else "Not Occupied"
    confidence = prediction if prediction > 0.5 else 1.0 - prediction
    
    print(f"Sample {i+1}: Prediction={prediction:.4f}, Status={occupancy_status}, Confidence={confidence:.4f}")

print(f"\nReceived {len(predictions)} predictions")
print("Summary:")
occupied_count = sum(1 for p in predictions if p > 0.5)
print(f"  Occupied: {occupied_count}/{len(predictions)}")
print(f"  Not Occupied: {len(predictions) - occupied_count}/{len(predictions)}")
print(f"  Average prediction: {np.mean(predictions):.4f}")

# Close consumer
output_consumer.close()
