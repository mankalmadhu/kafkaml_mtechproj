#!/usr/bin/env python3
"""
KafkaML MNIST Federated Learning End-to-End Automation Script

This script automates the complete workflow for federated learning with blockchain
using the MNIST dataset on KafkaML platform.

Usage:
    python automate_mnist_federated_e2e.py [--config config.yaml]

Author: KafkaML Team
Date: October 10, 2025
"""

import os
import sys
import time
import json
import logging
import argparse
import requests
import yaml
from typing import Dict, Any, Optional, List
from datetime import datetime
from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
import numpy as np

# Add parent directory to path for datasource imports
# This script is in e2e_scripts/, need to access kafka-ml root
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'automation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class KafkaMLAutomation:
    """Main automation class for KafkaML federated learning workflow"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize automation with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.backend_url = config['backend']['url']
        self.kafka_bootstrap = config['kafka']['bootstrap_servers']
        self.results = {
            'model_id': None,
            'config_id': None,
            'deployment_id': None,
            'result_id': None,
            'inference_id': None,
            'federated_id': None,
            'start_time': datetime.now().isoformat(),
            'end_time': None
        }
        
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> requests.Response:
        """
        Make HTTP request to backend API
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint
            data: Request data for POST/PUT
            
        Returns:
            Response object
        """
        url = f"{self.backend_url}{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        try:
            # Log the request for debugging
            logger.info(f"Request: {method} {url}")
            
            if method == 'GET':
                response = requests.get(url, timeout=30)
            elif method == 'POST':
                logger.info(f"Sending data: {list(data.keys()) if data else 'None'}")
                response = requests.post(url, json=data, timeout=30)
            elif method == 'PUT':
                response = requests.put(url, json=data, timeout=30)
            elif method == 'DELETE':
                response = requests.delete(url, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Log response details for debugging
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")
            logger.debug(f"Response content (first 200 chars): {response.text[:200]}")
            
            # Check for error status codes and log response body
            if response.status_code >= 400:
                logger.error(f"HTTP {response.status_code} error for {method} {url}")
                logger.error(f"Response body: {response.text}")
                logger.error(f"Request payload: {data}")
            
            response.raise_for_status()
            return response
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: Cannot connect to {url}")
            logger.error(f"Please ensure the backend is running and accessible")
            logger.error(f"Check: ./check_status.sh or curl {self.backend_url}/models/")
            raise
        except requests.exceptions.Timeout as e:
            logger.error(f"Request timeout: {method} {url}")
            logger.error(f"Backend might be overloaded or not responding")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {method} {url} - {str(e)}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response content: {e.response.text[:500]}")
            raise
    
    def create_model(self) -> int:
        """
        Step 1: Create TensorFlow model
        
        Returns:
            Model ID
        """
        logger.info("=" * 80)
        logger.info("STEP 1: Creating TensorFlow Model")
        logger.info("=" * 80)
        
        model_config = self.config['model']
        
        # Add timestamp to make name unique (max 30 chars)
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        base_name = model_config['name'][:19]  # Max 19 + 1 underscore + 10 timestamp = 30
        unique_name = f"{base_name}_{timestamp}"
        
        # Ensure name is within 30 char limit
        if len(unique_name) > 30:
            unique_name = unique_name[:30]
        
        payload = {
            "name": unique_name,
            "description": model_config['description'],
            "framework": model_config['framework'],
            "distributed": model_config['distributed'],
            "imports": model_config['imports'],
            "code": model_config['code']
        }
        
        logger.info(f"Creating model: {unique_name}")
        response = self._make_request('POST', '/models/', payload)
        
        # Backend returns 201 with no body, need to fetch the created model
        logger.info(f"✓ Model created successfully (Status: {response.status_code})")
        
        # Get the created model by name to retrieve its ID
        logger.info("Fetching created model details...")
        list_response = self._make_request('GET', '/models/')
        models_list = list_response.json()
        
        # Find the model we just created by name
        model_id = None
        for model in models_list:
            if model['name'] == unique_name:
                model_id = model['id']
                self.results['model_id'] = model_id
                logger.info(f"  Model ID: {model_id}")
                logger.info(f"  Name: {model['name']}")
                logger.info(f"  Framework: {model['framework']}")
                break
        
        if not model_id:
            raise ValueError(f"Could not find created model with name: {unique_name}")
        
        return model_id
    
    def create_configuration(self, model_id: int) -> int:
        """
        Step 2: Create configuration
        
        Args:
            model_id: ID of the model to associate
            
        Returns:
            Configuration ID
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Creating Configuration")
        logger.info("=" * 80)
        
        config_cfg = self.config['configuration']
        
        # Add timestamp to make name unique (max 30 chars)
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        base_name = config_cfg['name'][:19]  # Max 19 + 1 underscore + 10 timestamp = 30
        unique_name = f"{base_name}_{timestamp}"
        
        # Ensure name is within 30 char limit
        if len(unique_name) > 30:
            unique_name = unique_name[:30]
        
        payload = {
            "name": unique_name,
            "description": config_cfg['description'],
            "ml_models": [model_id]
        }
        
        logger.info(f"Creating configuration: {unique_name}")
        response = self._make_request('POST', '/configurations/', payload)
        
        # Backend returns 201 with no body, need to fetch the created configuration
        logger.info(f"✓ Configuration created successfully (Status: {response.status_code})")
        
        # Get the created configuration by name to retrieve its ID
        logger.info("Fetching created configuration details...")
        list_response = self._make_request('GET', '/configurations/')
        configs_list = list_response.json()
        
        # Find the configuration we just created by name
        config_id = None
        for config in configs_list:
            if config['name'] == unique_name:
                config_id = config['id']
                self.results['config_id'] = config_id
                logger.info(f"  Configuration ID: {config_id}")
                logger.info(f"  Name: {config['name']}")
                logger.info(f"  Associated models: {[m['name'] for m in config['ml_models']]}")
                break
        
        if not config_id:
            raise ValueError(f"Could not find created configuration with name: {unique_name}")
        
        return config_id
    
    def create_deployment(self, config_id: int) -> int:
        """
        Step 3: Create federated deployment with blockchain
        
        Args:
            config_id: ID of the configuration to deploy
            
        Returns:
            Deployment ID
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Creating Federated Deployment with Blockchain")
        logger.info("=" * 80)
        
        deploy_cfg = self.config['deployment']
        
        # Build tf_kwargs_fit with epochs from config
        epochs = deploy_cfg.get('epochs', 1)
        tf_kwargs_fit = f"epochs={epochs}, shuffle=True"
        
        payload = {
            "configuration": config_id,
            "batch": deploy_cfg['batch'],
            "optimizer": deploy_cfg['optimizer'],
            "learning_rate": deploy_cfg['learning_rate'],
            "loss": deploy_cfg['loss'],
            "metrics": deploy_cfg['metrics'],
            "federated": deploy_cfg['federated'],
            "blockchain": deploy_cfg['blockchain'],
            "agg_rounds": deploy_cfg['agg_rounds'],
            "min_data": deploy_cfg['min_data'],
            "agg_strategy": deploy_cfg['agg_strategy'],
            "data_restriction": deploy_cfg['data_restriction'],
            "gpumem": 0,
            "tf_kwargs_fit": tf_kwargs_fit,
            "tf_kwargs_val": "verbose=1",
            "pth_kwargs_fit": "",
            "pth_kwargs_val": ""
        }
        
        logger.info(f"Creating deployment with:")
        logger.info(f"  Federated: {deploy_cfg['federated']}")
        logger.info(f"  Blockchain: {deploy_cfg['blockchain']}")
        logger.info(f"  Aggregation rounds: {deploy_cfg['agg_rounds']}")
        logger.info(f"  Aggregation strategy: {deploy_cfg['agg_strategy']}")
        logger.info(f"  Minimum data: {deploy_cfg['min_data']}")
        
        response = self._make_request('POST', '/deployments/', payload)
        
        # Backend returns 201 with no body, need to fetch the created deployment
        logger.info(f"✓ Deployment created successfully (Status: {response.status_code})")
        
        # Get the created deployment - fetch latest deployment (most recent)
        logger.info("Fetching created deployment details...")
        list_response = self._make_request('GET', '/deployments/')
        deployments_list = list_response.json()
        
        # Get the most recent deployment (they are ordered by time descending)
        if deployments_list and len(deployments_list) > 0:
            deployment_data = deployments_list[0]
            deployment_id = deployment_data['id']
            self.results['deployment_id'] = deployment_id
            
            logger.info(f"  Deployment ID: {deployment_id}")
            logger.info(f"  Configuration: {deployment_data['configuration']['name']}")
            
            # Debug: Log available keys in deployment data
            logger.info(f"  Deployment data keys: {list(deployment_data.keys())}")
            
            # Extract federated_string_id if present
            federated_string_id = deployment_data.get('federated_string_id')
            logger.info(f"  Federated String ID from API: {federated_string_id!r}")
            
            if federated_string_id:
                self.results['federated_string_id'] = federated_string_id
                logger.info(f"  ✓ Federated String ID saved to results: {federated_string_id}")
            else:
                logger.warning(f"  ⚠ No Federated String ID in deployment (value: {federated_string_id})")
            
            # Get result ID from deployment
            if deployment_data.get('results') and len(deployment_data['results']) > 0:
                result_id = deployment_data['results'][0]['id']
                self.results['result_id'] = result_id
                logger.info(f"  Training result ID: {result_id}")
                logger.info(f"  Status: {deployment_data['results'][0]['status']}")
            else:
                logger.warning("No training results found in deployment")
        else:
            raise ValueError("Could not find created deployment")
        
        logger.info("\n  Backend is now:")
        logger.info("    • Creating Kafka topics")
        logger.info("    • Sending model to federated backend")
        logger.info("    • Creating blockchain controller job (if enabled)")
        
        # Wait for backend to process
        time.sleep(5)
        
        return deployment_id
    
    def ensure_topic_exists(self, topic_name: str, num_partitions: int = 1, replication_factor: int = 1) -> bool:
        """
        Ensure a Kafka topic exists, create it if it doesn't
        
        Args:
            topic_name: Name of the topic
            num_partitions: Number of partitions (default: 1)
            replication_factor: Replication factor (default: 1)
            
        Returns:
            True if topic exists or was created successfully
        """
        try:
            admin_client = KafkaAdminClient(
                bootstrap_servers=self.kafka_bootstrap,
                client_id='kafkaml-e2e-admin'
            )
            
            # Check if topic exists
            existing_topics = admin_client.list_topics()
            
            if topic_name in existing_topics:
                logger.info(f"Topic '{topic_name}' already exists")
                admin_client.close()
                return True
            
            # Create topic
            logger.info(f"Creating topic '{topic_name}' with {num_partitions} partition(s)...")
            topic = NewTopic(
                name=topic_name,
                num_partitions=num_partitions,
                replication_factor=replication_factor
            )
            
            admin_client.create_topics(new_topics=[topic], validate_only=False)
            logger.info(f"✓ Topic '{topic_name}' created successfully")
            
            admin_client.close()
            return True
            
        except TopicAlreadyExistsError:
            logger.info(f"Topic '{topic_name}' already exists (race condition)")
            return True
        except Exception as e:
            logger.error(f"Failed to create topic '{topic_name}': {e}")
            return False
    
    def inject_training_data(self, deployment_id: int, federated_string_id: Optional[str] = None) -> Dict[str, int]:
        """
        Step 4: Inject MNIST training data
        
        Args:
            deployment_id: ID of the deployment
            federated_string_id: Optional federated string ID for collision detection
            
        Returns:
            Dictionary with data statistics
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Injecting Training Data")
        logger.info("=" * 80)
        
        data_cfg = self.config['data_injection']
        
        # Import TensorFlow and datasource
        try:
            import tensorflow as tf
            from datasources.federated_raw_sink import FederatedRawSink
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            logger.error("Make sure TensorFlow and datasources are installed")
            raise
        
        logger.info("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        logger.info(f"  Training samples: {len(x_train)}")
        logger.info(f"  Test samples: {len(x_test)}")

        x_train = x_train[:1000]  # 200 training samples
        y_train = y_train[:1000]
        
        x_test = x_test[:500]  # 50 test samples
        y_test = y_test[:500]
        
        # Ensure the data topic exists before injecting data
        logger.info(f"\nEnsuring topic exists: {data_cfg['topic']}")
        if not self.ensure_topic_exists(data_cfg['topic']):
            raise RuntimeError(f"Failed to create or verify topic: {data_cfg['topic']}")
        
        # Create datasource
        logger.info(f"\nCreating FederatedRawSink for topic: {data_cfg['topic']}")
        logger.info(f"  Deployment ID: {deployment_id}")
        logger.info(f"  Federated String ID parameter: {federated_string_id!r}")
        
        if federated_string_id:
            logger.info(f"  ✓ Will include federated_string_id in datasource")
        else:
            logger.warning(f"  ⚠ No federated_string_id to include (value: {federated_string_id})")
        
        # Use json.dumps() to create proper JSON string (as in working example)
        
        mnist_sink = FederatedRawSink(
            boostrap_servers=data_cfg['bootstrap_servers'],
            topic=data_cfg['topic'],
            deployment_id=deployment_id,
            description=data_cfg['description'],
            dataset_restrictions=json.dumps(data_cfg['dataset_restrictions']),
            validation_rate=data_cfg['validation_rate'],
            test_rate=data_cfg['test_rate'],
            control_topic=data_cfg['control_topic'],
            data_type=data_cfg['data_type'],
            data_reshape=data_cfg['data_reshape'],
            label_type=data_cfg['label_type'],
            label_reshape=data_cfg['label_reshape'],
            federated_string_id=federated_string_id
        )
        
        # Send training data
        logger.info(f"\nSending {len(x_train)} training samples...")
        for i, (x, y) in enumerate(zip(x_train, y_train)):
            mnist_sink.send(data=x, label=y)
            if (i + 1) % 10000 == 0:
                logger.info(f"  Sent {i + 1}/{len(x_train)} training samples")
        
        # Send test data
        logger.info(f"\nSending {len(x_test)} test samples...")
        for i, (x, y) in enumerate(zip(x_test, y_test)):
            mnist_sink.send(data=x, label=y)
            if (i + 1) % 1000 == 0:
                logger.info(f"  Sent {i + 1}/{len(x_test)} test samples")
        
        # Close to trigger collision detection
        logger.info("\nClosing datasource and triggering collision detection...")
        mnist_sink.close()
        
        total_samples = len(x_train) + len(x_test)
        logger.info(f"✓ Data injection completed")
        logger.info(f"  Total samples sent: {total_samples}")
        logger.info(f"  Validation rate: {data_cfg['validation_rate'] * 100}%")
        logger.info(f"  Test rate: {data_cfg['test_rate'] * 100}%")
        
        logger.info("\n  Federated backend will now:")
        logger.info("    • Receive datasource registration")
        logger.info("    • Perform collision detection")
        logger.info("    • Spawn federated worker job(s)")
        
        return {
            'total_samples': total_samples,
            'train_samples': len(x_train),
            'test_samples': len(x_test)
        }
    
    def wait_for_training_completion(self, deployment_id: int, result_id: int, max_wait_seconds: int = 3600) -> Dict[str, Any]:
        """
        Step 5: Wait for training to complete
        
        Args:
            deployment_id: Deployment ID
            result_id: Training result ID
            max_wait_seconds: Maximum time to wait
            
        Returns:
            Training result data
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Monitoring Training Progress")
        logger.info("=" * 80)
        
        logger.info(f"Monitoring deployment {deployment_id}, result {result_id}")
        logger.info(f"Waiting for training to complete (max {max_wait_seconds}s)...")
        logger.info("This may take a while depending on aggregation rounds...\n")
        
        start_time = time.time()
        last_status = None
        check_interval = 30  # Check every 30 seconds
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed > max_wait_seconds:
                logger.warning(f"⚠ Training timeout after {max_wait_seconds}s")
                logger.warning("Training may still be in progress. Check manually.")
                break
            
            try:
                # Get all results for this deployment
                response = self._make_request('GET', f'/deployments/results/{deployment_id}')
                results_list = response.json()
                
                # Find the specific result by result_id
                result_data = None
                for result in results_list:
                    if result['id'] == result_id:
                        result_data = result
                        break
                
                if not result_data:
                    logger.error(f"Result {result_id} not found in deployment {deployment_id}")
                    time.sleep(check_interval)
                    continue
                
                status = result_data['status']
                
                # Log status change
                if status != last_status:
                    logger.info(f"Status: {status}")
                    last_status = status
                
                # Check for completion
                if status == 'finished':
                    logger.info(f"\n✓ Training completed successfully!")
                    logger.info(f"  Total time: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
                    
                    if result_data.get('train_metrics'):
                        train_metrics = result_data['train_metrics']
                        if isinstance(train_metrics, dict) and 'accuracy' in train_metrics:
                            accuracies = train_metrics['accuracy']
                            if isinstance(accuracies, list) and len(accuracies) > 0:
                                logger.info(f"  Final training accuracy: {accuracies[-1]:.4f}")
                    
                    if result_data.get('val_metrics'):
                        val_metrics = result_data['val_metrics']
                        if isinstance(val_metrics, dict) and 'accuracy' in val_metrics:
                            accuracies = val_metrics['accuracy']
                            if isinstance(accuracies, list) and len(accuracies) > 0:
                                logger.info(f"  Final validation accuracy: {accuracies[-1]:.4f}")
                    
                    if result_data.get('training_time'):
                        logger.info(f"  Training time: {result_data['training_time']}s")
                    
                    return result_data
                
                elif status == 'stopped':
                    logger.error("✗ Training was stopped")
                    break
                
                elif status in ['created', 'deployed']:
                    # Still waiting
                    if elapsed % 60 < check_interval:  # Log every minute
                        logger.info(f"  Waiting... ({elapsed:.0f}s elapsed)")
                
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error checking training status: {e}")
                time.sleep(check_interval)
        
        # Return partial data if we exit early
        try:
            response = self._make_request('GET', f'/deployments/results/{deployment_id}')
            results_list = response.json()
            for result in results_list:
                if result['id'] == result_id:
                    return result
            return {}
        except:
            return {}
    
    def create_inference(self, result_id: int) -> int:
        """
        Step 6: Create inference configuration
        
        Args:
            result_id: Training result ID
            
        Returns:
            Inference ID
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Creating Inference Configuration")
        logger.info("=" * 80)
        
        inference_cfg = self.config['inference']
        
        payload = {
            "model_result": result_id,
            "replicas": inference_cfg['replicas'],
            "input_format": inference_cfg['input_format'],
            "input_config": json.dumps(inference_cfg['input_config']),
            "input_topic": inference_cfg['input_topic'],
            "output_topic": inference_cfg['output_topic'],
            "gpumem": 0  # GPU memory allocation (0 for CPU-only)
        }
        
        logger.info(f"Creating inference deployment:")
        logger.info(f"  Result ID: {result_id}")
        logger.info(f"  Input topic: {inference_cfg['input_topic']}")
        logger.info(f"  Output topic: {inference_cfg['output_topic']}")
        logger.info(f"  Replicas: {inference_cfg['replicas']}")
        
        response = self._make_request('POST', f'/results/inference/{result_id}', payload)
        
        # Backend returns 200 OK with no body, need to fetch the inference details
        logger.info(f"✓ Inference deployment request accepted (Status: {response.status_code})")
        
        # Get the created inference - fetch latest inference for this result
        logger.info("Fetching created inference details...")
        list_response = self._make_request('GET', '/inferences/')
        inferences_list = list_response.json()
        
        # Find the inference for our result_id (most recent)
        inference_id = None
        for inference in inferences_list:
            logger.info(f"Inference: {inference}")
            if inference['model_result']  == result_id:
                inference_id = inference['id']
                self.results['inference_id'] = inference_id
                logger.info(f"  Inference ID: {inference_id}")
                logger.info(f"  Status: {inference['status']}")
                break
        
        if not inference_id:
            logger.warning("Could not find created inference, but deployment was accepted")
        
        logger.info("\n  Backend is now:")
        logger.info("    • Creating Kubernetes inference deployment")
        logger.info("    • Loading trained model")
        logger.info("    • Starting to consume from input topic")
        
        # Wait for inference deployment
        logger.info("\nWaiting for inference deployment to be ready...")
        time.sleep(10)
        
        return inference_id
    
    def run_inference(self, num_predictions: int = 10) -> List[Dict[str, Any]]:
        """
        Step 7: Run inference on test data
        
        Args:
            num_predictions: Number of predictions to make
            
        Returns:
            List of predictions
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: Running Inference")
        logger.info("=" * 80)
        
        inference_cfg = self.config['inference']
        
        # Ensure inference topics exist
        logger.info("\nEnsuring inference topics exist...")
        if not self.ensure_topic_exists(inference_cfg['input_topic']):
            raise RuntimeError(f"Failed to create or verify input topic: {inference_cfg['input_topic']}")
        if not self.ensure_topic_exists(inference_cfg['output_topic']):
            raise RuntimeError(f"Failed to create or verify output topic: {inference_cfg['output_topic']}")
        
        # Import TensorFlow
        try:
            import tensorflow as tf
        except ImportError as e:
            logger.error(f"Failed to import TensorFlow: {e}")
            raise
        
        logger.info("Loading MNIST test dataset...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Create producer
        logger.info(f"\nCreating Kafka producer for topic: {inference_cfg['input_topic']}")
        producer = KafkaProducer(bootstrap_servers=self.kafka_bootstrap)
        
        # Select random samples
        random_indices = np.random.choice(x_test.shape[0], num_predictions, replace=False)
        actual_labels = []

        logger.info(f"\nCreating Kafka consumer for topic: {inference_cfg['output_topic']}")
        consumer = KafkaConsumer(
            inference_cfg['output_topic'],
            bootstrap_servers=self.kafka_bootstrap,
            group_id=f"automation_{int(time.time())}",
            auto_offset_reset='latest',
            consumer_timeout_ms=90000  # 90 second timeout
        )
        
        # Wait for consumer to be assigned partitions and positioned at latest offset
        logger.info("Waiting for consumer to be ready...")
        max_wait = 30  # Maximum 30 seconds to wait
        wait_start = time.time()
        
        while not consumer.assignment():
            if time.time() - wait_start > max_wait:
                logger.warning("Consumer partition assignment timeout - proceeding anyway")
                break
            consumer.poll(timeout_ms=100)  # Trigger partition assignment
            time.sleep(0.1)
        
        if consumer.assignment():
            logger.info(f"✓ Consumer ready and assigned to partitions: {consumer.assignment()}")
        
        logger.info(f"\nSending {num_predictions} images for prediction...")
        for idx, i in enumerate(random_indices):
            producer.send(inference_cfg['input_topic'], x_test[i].tobytes())
            actual_labels.append(int(y_test[i]))
            logger.info(f"  Sent image {idx + 1}/{num_predictions} (actual label: {y_test[i]})")
        
        producer.flush()
        producer.close()
        logger.info("✓ All images sent!")
        
        # Create consumer
        
        # Receive predictions
        logger.info("\nReceiving predictions:")
        predictions = []
        correct = 0
        
        for idx, msg in enumerate(consumer):
            try:
                # Parse prediction
                prediction_str = msg.value.decode()
                prediction_obj = json.loads(prediction_str)
                
                # Extract predicted digit as argmax of probability array
                probs = prediction_obj['values']
                predicted = int(np.argmax(probs))
                confidence = float(max(probs))
                
                actual = actual_labels[idx]
                match = predicted == actual
                if match:
                    correct += 1
                
                match_symbol = "✓" if match else "✗"
                
                prediction_result = {
                    'index': idx + 1,
                    'predicted': predicted,
                    'actual': actual,
                    'confidence': confidence,
                    'correct': match
                }
                predictions.append(prediction_result)
                
                logger.info(f"  Prediction {idx + 1}: {predicted} (confidence: {confidence:.4f}) | "
                          f"Actual: {actual} {match_symbol}")
                
                if idx + 1 >= num_predictions:
                    break
                    
            except Exception as e:
                logger.error(f"  Error parsing prediction: {e}")
        
        consumer.close()
        
        # Summary
        accuracy = (correct / len(predictions) * 100) if predictions else 0
        logger.info(f"\n✓ Inference completed!")
        logger.info(f"  Accuracy: {accuracy:.2f}% ({correct}/{len(predictions)})")
        
        return predictions
    
    def save_results(self, output_file: str = "automation_results.json"):
        """
        Save automation results to file
        
        Args:
            output_file: Output file path
        """
        self.results['end_time'] = datetime.now().isoformat()
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\n✓ Results saved to: {output_file}")
    
    def print_summary(self):
        """Print execution summary"""
        logger.info("\n" + "=" * 80)
        logger.info("AUTOMATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Model ID:        {self.results.get('model_id', 'N/A')}")
        logger.info(f"Configuration ID: {self.results.get('config_id', 'N/A')}")
        logger.info(f"Deployment ID:   {self.results.get('deployment_id', 'N/A')}")
        logger.info(f"Result ID:       {self.results.get('result_id', 'N/A')}")
        logger.info(f"Inference ID:    {self.results.get('inference_id', 'N/A')}")
        logger.info(f"Start Time:      {self.results.get('start_time', 'N/A')}")
        logger.info(f"End Time:        {self.results.get('end_time', 'N/A')}")
        logger.info("=" * 80)
    
    def run_full_pipeline(self, skip_steps: Optional[List[str]] = None, pause_between_steps: bool = False):
        """
        Run the complete automation pipeline
        
        Args:
            skip_steps: List of step names to skip
            pause_between_steps: If True, wait for user input after each step
        """
        skip_steps = skip_steps or []
        
        def wait_for_user():
            """Pause and wait for user to press Enter"""
            if pause_between_steps:
                input("\n>>> Press Enter to continue to next step... ")
        
        logger.info("\n" + "=" * 80)
        logger.info("KafkaML MNIST Federated Learning - Full Automation")
        logger.info("=" * 80)
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Backend URL: {self.backend_url}")
        logger.info(f"Kafka Bootstrap: {self.kafka_bootstrap}")
        logger.info("=" * 80)
        
        # Pre-flight check: Verify backend is accessible
        logger.info("\nPerforming pre-flight check...")
        try:
            test_response = requests.get(f"{self.backend_url}/models/", timeout=10)
            logger.info(f"✓ Backend is accessible (status: {test_response.status_code})")
        except requests.exceptions.ConnectionError:
            logger.error("✗ Cannot connect to backend!")
            logger.error(f"Please ensure backend is running at: {self.backend_url}")
            logger.error("Run: ./check_status.sh")
            raise RuntimeError("Backend is not accessible")
        except Exception as e:
            logger.warning(f"Pre-flight check warning: {e}")
        
        logger.info("Pre-flight check complete\n")
        
        try:
            # Step 1: Create model
            if 'model' not in skip_steps:
                model_id = self.create_model()
                wait_for_user()
            else:
                model_id = self.config.get('existing_ids', {}).get('model_id')
                logger.info(f"Skipping model creation, using existing ID: {model_id}")
            
            # Step 2: Create configuration
            if 'configuration' not in skip_steps:
                config_id = self.create_configuration(model_id)
                wait_for_user()
            else:
                config_id = self.config.get('existing_ids', {}).get('config_id')
                logger.info(f"Skipping configuration creation, using existing ID: {config_id}")
            
            # Step 3: Create deployment
            if 'deployment' not in skip_steps:
                deployment_id = self.create_deployment(config_id)
                wait_for_user()
            else:
                deployment_id = self.config.get('existing_ids', {}).get('deployment_id')
                logger.info(f"Skipping deployment creation, using existing ID: {deployment_id}")
            
            # Step 4: Inject data
            if 'data' not in skip_steps:
                federated_string_id = self.results.get('federated_string_id')
                logger.info(f"Passing federated_string_id to data injection: {federated_string_id!r}")
                data_stats = self.inject_training_data(deployment_id, federated_string_id)
                wait_for_user()
            else:
                logger.info("Skipping data injection")
            
            # Step 5: Wait for training
            if 'training' not in skip_steps:
                deployment_id = self.results['deployment_id']
                result_id = self.results['result_id']
                training_result = self.wait_for_training_completion(deployment_id, result_id)
                wait_for_user()
            else:
                logger.info("Skipping training wait")
                result_id = self.config.get('existing_ids', {}).get('result_id')
            
            # Step 6: Create inference
            if 'inference_config' not in skip_steps:
                inference_id = self.create_inference(result_id)
                wait_for_user()
            else:
                inference_id = self.config.get('existing_ids', {}).get('inference_id')
                logger.info(f"Skipping inference creation, using existing ID: {inference_id}")
            
            # Step 7: Run inference
            if 'inference' not in skip_steps:
                predictions = self.run_inference(
                    num_predictions=self.config['inference'].get('num_predictions', 10)
                )
                wait_for_user()
            else:
                logger.info("Skipping inference execution")
            
            # Save and summarize
            self.save_results()
            self.print_summary()
            
            logger.info("\n" + "=" * 80)
            logger.info("✓ AUTOMATION COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"\n✗ AUTOMATION FAILED: {str(e)}")
            logger.exception("Full traceback:")
            self.save_results()
            raise


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_file: Path to config file
        
    Returns:
        Configuration dictionary
    """
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    else:
        logger.warning(f"Config file not found: {config_file}")
        logger.info("Using default configuration")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration
    
    Returns:
        Default configuration dictionary
    """
    return {
        'backend': {
            'url': 'http://localhost:9090'
        },
        'kafka': {
            'bootstrap_servers': 'localhost:9094'
        },
        'model': {
            'name': 'mnist_federated_model',
            'description': 'MNIST federated learning model with blockchain',
            'framework': 'tf',
            'distributed': False,
            'imports': '''import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers''',
            'code': '''model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)), 
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)'''
        },
        'configuration': {
            'name': 'mnist_fed_config',
            'description': 'Configuration for MNIST federated training'
        },
        'deployment': {
            'batch': 32,
            'epochs': 3,
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'loss': 'sparse_categorical_crossentropy',
            'metrics': 'sparse_categorical_accuracy',
            'federated': True,
            'blockchain': True,
            'agg_rounds': 5,
            'min_data': 100,
            'agg_strategy': 'FedAvg',
            'data_restriction': {
                'features': {
                    'label': {
                        'num_classes': 10,
                        'names': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
                    }
                },
                'supervised_keys': {
                    'input': 'image',
                    'output': 'label'
                }
            }
        },
        'data_injection': {
            'bootstrap_servers': 'localhost:9094',
            'topic': 'mnist_fed',
            'description': 'MNIST dataset for federated training',
            'validation_rate': 0.1,
            'test_rate': 0.0,
            'control_topic': 'FEDERATED_DATA_CONTROL_TOPIC',
            'data_type': 'uint8',
            'data_reshape': '784',
            'label_type': 'uint8',
            'label_reshape': '',
            'dataset_restrictions': {
                'features': {
                    'label': {
                        'num_classes': 10,
                        'names': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
                    }
                },
                'supervised_keys': {
                    'input': 'image',
                    'output': 'label'
                }
            }
        },
        'inference': {
            'replicas': 1,
            'input_format': 'RAW',
            'input_config': {
                'data_type': 'uint8',
                'data_reshape': '784'
            },
            'input_topic': 'mnist_in',
            'output_topic': 'mnist_out',
            'num_predictions': 10
        }
    }


def save_default_config(output_file: str = "config_mnist_federated.yaml"):
    """
    Save default configuration to file
    
    Args:
        output_file: Output file path
    """
    config = get_default_config()
    
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Default configuration saved to: {output_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='KafkaML MNIST Federated Learning Automation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run with default configuration
  python automate_mnist_federated_e2e.py
  
  # Run with custom configuration
  python automate_mnist_federated_e2e.py --config my_config.yaml
  
  # Generate default configuration file
  python automate_mnist_federated_e2e.py --save-config
  
  # Skip certain steps (use existing resources)
  python automate_mnist_federated_e2e.py --skip model,configuration
        '''
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--save-config',
        action='store_true',
        help='Save default configuration to file and exit'
    )
    
    parser.add_argument(
        '--skip',
        type=str,
        help='Comma-separated list of steps to skip (model,configuration,deployment,data,training,inference_config,inference)'
    )
    
    parser.add_argument(
        '--pause',
        action='store_true',
        help='Pause after each step and wait for user to press Enter'
    )
    
    args = parser.parse_args()
    
    # Handle save-config
    if args.save_config:
        save_default_config()
        return
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # Parse skip steps
    skip_steps = []
    if args.skip:
        skip_steps = [s.strip() for s in args.skip.split(',')]
    
    # Run automation
    automation = KafkaMLAutomation(config)
    automation.run_full_pipeline(skip_steps=skip_steps, pause_between_steps=args.pause)


if __name__ == '__main__':
    main()

