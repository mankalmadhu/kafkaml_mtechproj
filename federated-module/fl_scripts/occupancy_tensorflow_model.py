#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import json
import logging
import csv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def create_occupancy_model():
    """
    Create a TensorFlow model for occupancy prediction.
    
    Input: 5 features (Temperature, Humidity, Light, CO2, HumidityRatio)
    Output: Binary classification (Occupancy: 0 or 1)
    
    Returns:
        tf.keras.Model: Compiled model ready for training
    """
    
    logging.info("Creating occupancy prediction model...")
    
    # Define model architecture
    model = tf.keras.Sequential([
        # Input layer: 5 features
        tf.keras.layers.Dense(64, activation='relu', input_shape=(5,), name='dense_1'),
        tf.keras.layers.Dropout(0.2, name='dropout_1'),
        
        # Hidden layers
        tf.keras.layers.Dense(32, activation='relu', name='dense_2'),
        tf.keras.layers.Dropout(0.2, name='dropout_2'),
        
        tf.keras.layers.Dense(16, activation='relu', name='dense_3'),
        
        # Output layer: Binary classification
        tf.keras.layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()
        ]
    )
    
    logging.info("✅ Model created and compiled successfully")
    
    return model

def get_model_architecture():
    """
    Get the model architecture as a JSON string for Kafka ML frontend.
    
    Returns:
        str: JSON string containing model architecture
    """
    
    model = create_occupancy_model()
    
    # Convert model to JSON
    model_json = model.to_json()
    
    logging.info("Model architecture JSON generated")
    
    return model_json

def get_model_compile_args():
    """
    Get the model compilation arguments as a JSON string for Kafka ML frontend.
    
    Returns:
        str: JSON string containing compilation arguments
    """
    
    compile_args = {
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "metrics": ["accuracy", "precision", "recall", "auc"]
    }
    
    compile_args_json = json.dumps(compile_args)
    
    logging.info("Model compile args JSON generated")
    
    return compile_args_json

def get_training_settings():
    """
    Get the training settings as a JSON string for Kafka ML frontend.
    
    Returns:
        str: JSON string containing training settings
    """
    
    training_settings = {
        "epochs": 10,
        "batch": 32,
        "validation_split": 0.2,
        "verbose": 1
    }
    
    training_settings_json = json.dumps(training_settings)
    
    logging.info("Training settings JSON generated")
    
    return training_settings_json

def test_model():
    """
    Test the model with dummy data to ensure it works correctly.
    """
    
    logging.info("Testing model with dummy data...")
    
    # Create model
    model = create_occupancy_model()
    
    # Print model summary
    model.summary()
    
    # Create dummy input data (5 features)
    dummy_input = np.random.random((10, 5)).astype(np.float32)
    dummy_labels = np.random.randint(0, 2, (10, 1)).astype(np.float32)
    
    logging.info(f"Dummy input shape: {dummy_input.shape}")
    logging.info(f"Dummy labels shape: {dummy_labels.shape}")
    
    # Test forward pass
    predictions = model.predict(dummy_input, verbose=0)
    logging.info(f"Predictions shape: {predictions.shape}")
    logging.info(f"Sample predictions: {predictions[:5].flatten()}")
    
    # Test training step
    loss = model.evaluate(dummy_input, dummy_labels, verbose=0)
    logging.info(f"Test loss: {loss}")
    
    logging.info("✅ Model test completed successfully")

def test_model_with_real_data():
    """
    Test the model with real data from datatraining.txt file.
    """
    
    logging.info("Testing model with real occupancy data...")
    
    # Create model
    model = create_occupancy_model()
    
    # Load real data
    data_file = '/Users/madhuahobalan/workspace/kafkaml/kafka-ml/federated-module/fl_scripts/datatraining.txt'
    
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
    
    logging.info(f"Loaded real dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logging.info(f"Feature ranges: Temperature={X[:, 0].min():.2f}-{X[:, 0].max():.2f}, "
                f"Humidity={X[:, 1].min():.2f}-{X[:, 1].max():.2f}, "
                f"Light={X[:, 2].min():.2f}-{X[:, 2].max():.2f}, "
                f"CO2={X[:, 3].min():.2f}-{X[:, 3].max():.2f}, "
                f"HumidityRatio={X[:, 4].min():.6f}-{X[:, 4].max():.6f}")
    logging.info(f"Label distribution: Unoccupied={np.sum(y == 0)}, Occupied={np.sum(y == 1)}")
    
    # Normalize features using StandardScaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X).astype(np.float32)
    
    logging.info("Features normalized using StandardScaler")
    
    # Split data for testing
    # Use first 1000 samples for training, next 200 for testing
    X_train = X_normalized[:1000]
    y_train = y[:1000]
    
    X_test = X_normalized[1000:1200]  # 200 test samples
    y_test = y[1000:1200]
    
    logging.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Train the model
    logging.info("Training model with real data...")
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate on test data
    logging.info("Evaluating model on test data...")
    test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(
        X_test, y_test, verbose=0
    )
    
    logging.info(f"Test Results:")
    logging.info(f"  Loss: {test_loss:.4f}")
    logging.info(f"  Accuracy: {test_accuracy:.4f}")
    logging.info(f"  Precision: {test_precision:.4f}")
    logging.info(f"  Recall: {test_recall:.4f}")
    logging.info(f"  AUC: {test_auc:.4f}")
    
    # Make predictions on test data
    predictions = model.predict(X_test, verbose=0)
    
    # Show some sample predictions
    logging.info("Sample predictions:")
    for i in range(min(10, len(X_test))):
        actual = int(y_test[i])
        predicted_prob = predictions[i][0]
        predicted_class = 1 if predicted_prob > 0.5 else 0
        status = "Occupied" if actual == 1 else "Unoccupied"
        pred_status = "Occupied" if predicted_class == 1 else "Unoccupied"
        
        logging.info(f"Sample {i+1}: Actual={status}, Predicted={pred_status} (prob={predicted_prob:.4f})")
    
    # Calculate confusion matrix
    y_pred_binary = (predictions > 0.5).astype(int).flatten()
    
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(y_test, y_pred_binary)
    logging.info(f"Confusion Matrix:")
    logging.info(f"  True Negatives: {cm[0,0]}")
    logging.info(f"  False Positives: {cm[0,1]}")
    logging.info(f"  False Negatives: {cm[1,0]}")
    logging.info(f"  True Positives: {cm[1,1]}")
    
    # Classification report
    report = classification_report(y_test, y_pred_binary, target_names=['Unoccupied', 'Occupied'])
    logging.info(f"Classification Report:\n{report}")
    
    logging.info("✅ Real data test completed successfully")
    
    return model, scaler, history

def save_model_example():
    """
    Example of how to save the model for later use.
    """
    
    logging.info("Saving model example...")
    
    model = create_occupancy_model()
    
    # Save model in different formats
    model.save('occupancy_model.h5')
    # model.save('occupancy_model_savedmodel')
    
    # Save only weights
    model.save_weights('occupancy_model.weights.h5')
    
    logging.info("✅ Model saved in multiple formats")

def load_and_predict_example():
    """
    Example of how to load a saved model and make predictions.
    """
    
    logging.info("Loading model and making predictions example...")
    
    try:
        # Load model
        model = tf.keras.models.load_model('occupancy_model.h5')
        
        # Create sample data (normalized features)
        # Example: [Temperature, Humidity, Light, CO2, HumidityRatio]
        sample_data = np.array([
            [23.0, 27.0, 426.0, 721.0, 0.0048],  # Occupied
            [22.0, 27.0, 0.0, 685.0, 0.0047],    # Unoccupied
        ], dtype=np.float32)
        
        # Make predictions
        predictions = model.predict(sample_data, verbose=0)
        
        logging.info("Sample predictions:")
        for i, pred in enumerate(predictions):
            occupancy_prob = pred[0]
            occupancy_status = "Occupied" if occupancy_prob > 0.5 else "Unoccupied"
            logging.info(f"Sample {i+1}: Probability={occupancy_prob:.4f}, Status={occupancy_status}")
        
        logging.info("✅ Prediction example completed")
        
    except FileNotFoundError:
        logging.warning("Model file not found. Run save_model_example() first.")

if __name__ == "__main__":
    logging.info("=== OCCUPANCY TENSORFLOW MODEL ===")
    
    # Test the model with dummy data
    test_model()
    
    # Test the model with real data
    logging.info("\n" + "="*50)
    model, scaler, history = test_model_with_real_data()
    
    # Generate JSON strings for Kafka ML frontend
    logging.info("\n=== KAFKA ML FRONTEND CONFIGURATION ===")
    
    model_arch = get_model_architecture()
    compile_args = get_model_compile_args()
    training_settings = get_training_settings()
    
    logging.info("Model Architecture JSON:")
    print(model_arch)
    
    logging.info("\nModel Compile Args JSON:")
    print(compile_args)
    
    logging.info("\nTraining Settings JSON:")
    print(training_settings)
    
    # Save model example
    logging.info("\n=== SAVE MODEL EXAMPLE ===")
    save_model_example()
    
    # Load and predict example
    logging.info("\n=== LOAD AND PREDICT EXAMPLE ===")
    load_and_predict_example()
    
    logging.info("=== OCCUPANCY TENSORFLOW MODEL COMPLETED ===")
