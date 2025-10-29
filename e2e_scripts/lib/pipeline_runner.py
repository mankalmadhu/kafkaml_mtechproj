"""
Pipeline orchestration for KafkaML E2E automation
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from kafka import KafkaProducer, KafkaConsumer
import numpy as np

# Add parent directory to path for datasource imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(parent_dir)

from .kafkaml_client import KafkaMLClient
from .kafka_admin import KafkaAdmin

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrates the complete E2E pipeline"""
    
    def __init__(self, config: Dict[str, Any], dataset):
        """
        Initialize pipeline runner
        
        Args:
            config: Full configuration dictionary
            dataset: Dataset instance (BaseDataset)
        """
        self.config = config
        self.dataset = dataset
        self.client = KafkaMLClient(config['backend']['url'])
        self.kafka_admin = KafkaAdmin(config['kafka']['bootstrap_servers'])
        self.kafka_bootstrap = config['kafka']['bootstrap_servers']
        
        self.results = {
            'model_id': None,
            'config_id': None,
            'deployment_id': None,
            'result_id': None,
            'inference_id': None,
            'federated_string_id': None,
            'start_time': datetime.now().isoformat(),
            'end_time': None
        }
    
    def _is_multi_device_config(self) -> bool:
        """Check if configuration is for multi-device setup"""
        data_cfg = self.config.get('data_injection', {})
        return 'devices' in data_cfg and isinstance(data_cfg['devices'], list)
    
    def inject_training_data(self, deployment_id: int, federated_string_id: Optional[str] = None) -> Dict[str, int]:
        """
        Inject training data using dataset - dispatches to single or multi-device
        
        Args:
            deployment_id: Deployment ID
            federated_string_id: Optional federated string ID
            
        Returns:
            Dictionary with data statistics
        """
        if self._is_multi_device_config():
            logger.info("Detected multi-device configuration")
            return self._inject_training_data_multi_device(deployment_id, federated_string_id)
        else:
            logger.info("Detected single-device configuration")
            return self._inject_training_data_single_device(deployment_id, federated_string_id)
    
    def _inject_training_data_single_device(self, deployment_id: int, federated_string_id: Optional[str] = None) -> Dict[str, int]:
        """
        Inject training data for single device (existing logic)
        
        Args:
            deployment_id: Deployment ID
            federated_string_id: Optional federated string ID
            
        Returns:
            Dictionary with data statistics
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Injecting Training Data (Single Device)")
        logger.info("=" * 80)
        
        data_cfg = self.config['data_injection']
        dataset_cfg = self.config['dataset']
        
        # Import datasource
        try:
            from datasources.federated_raw_sink import FederatedRawSink
        except ImportError as e:
            logger.error(f"Failed to import FederatedRawSink: {e}")
            raise
        
        # Load data from dataset
        logger.info(f"Loading {dataset_cfg['type']} dataset...")
        x_train, y_train = self.dataset.load_training_data(dataset_cfg.get('train_samples', 1000))
        x_test, y_test = self.dataset.load_test_data(dataset_cfg.get('test_samples', 500))
        
        logger.info(f"  Training samples: {len(x_train)}")
        logger.info(f"  Test samples: {len(x_test)}")
        
        # Ensure topic exists
        logger.info(f"\nEnsuring topic exists: {data_cfg['topic']}")
        if not self.kafka_admin.ensure_topic_exists(data_cfg['topic']):
            raise RuntimeError(f"Failed to create or verify topic: {data_cfg['topic']}")
        
        # Create datasource
        logger.info(f"\nCreating FederatedRawSink for topic: {data_cfg['topic']}")
        logger.info(f"  Deployment ID: {deployment_id}")
        logger.info(f"  Federated String ID: {federated_string_id!r}")
        
        if federated_string_id:
            logger.info(f"  ✓ Will include federated_string_id in datasource")
        else:
            logger.warning(f"  ⚠ No federated_string_id to include")
               
        # Only use streaming if explicitly present in config, then override with agg_rounds
        if 'streaming_data_chunks' in data_cfg:
            deployment_cfg = self.config.get('deployment', {})
            streaming_chunks = deployment_cfg.get('agg_rounds', None)
            logger.info(f"Streaming data chunks enabled: {streaming_chunks} (from agg_rounds)")
        else:
            streaming_chunks = None
            logger.info("Streaming data chunks disabled (not in config)")
        
        sink = FederatedRawSink(
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
            federated_string_id=federated_string_id,
            label_weights=data_cfg.get('label_weights', None),
            streaming_data_chunks=streaming_chunks
        )
        
        # Send training data
        logger.info(f"\nSending {len(x_train)} training samples...")
        for i, (x, y) in enumerate(zip(x_train, y_train)):
            sink.send(data=x, label=y)
            if (i + 1) % 10000 == 0:
                logger.info(f"  Sent {i + 1}/{len(x_train)} training samples")
        
        # Send test data
        logger.info(f"\nSending {len(x_test)} test samples...")
        for i, (x, y) in enumerate(zip(x_test, y_test)):
            sink.send(data=x, label=y)
            if (i + 1) % 1000 == 0:
                logger.info(f"  Sent {i + 1}/{len(x_test)} test samples")
        
        # Close to trigger collision detection
        logger.info("\nClosing datasource and triggering collision detection...")
        sink.close()
        
        total_samples = len(x_train) + len(x_test)
        logger.info(f"✓ Data injection completed")
        logger.info(f"  Total samples sent: {total_samples}")
        
        return {
            'total_samples': total_samples,
            'train_samples': len(x_train),
            'test_samples': len(x_test)
        }
    
    def _inject_training_data_multi_device(self, deployment_id: int, federated_string_id: Optional[str] = None) -> Dict[str, int]:
        """
        Inject training data for multiple devices with random splits
        
        Args:
            deployment_id: Deployment ID
            federated_string_id: Optional federated string ID
            
        Returns:
            Dictionary with data statistics
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Injecting Training Data (Multi-Device)")
        logger.info("=" * 80)
        
        data_cfg = self.config['data_injection']
        dataset_cfg = self.config['dataset']
        devices = data_cfg['devices']
        
        
        # Only use streaming if explicitly present in config, then override with agg_rounds
        if 'streaming_data_chunks' in data_cfg:
            deployment_cfg = self.config.get('deployment', {})
            streaming_chunks = deployment_cfg.get('agg_rounds', None)
            logger.info(f"Streaming data chunks enabled: {streaming_chunks} (from agg_rounds)")
        else:
            streaming_chunks = None
            logger.info("Streaming data chunks disabled (not in config)")
        
        logger.info(f"Number of devices: {len(devices)}")
        for i, dev in enumerate(devices):
            data_file = dev.get('data_file', 'default')
            logger.info(f"  Device {i+1}: {dev['device_id']} - Topic: {dev['topic']} - Data: {data_file}")
        
        # Import datasource
        try:
            from datasources.federated_raw_sink import FederatedRawSink
        except ImportError as e:
            logger.error(f"Failed to import FederatedRawSink: {e}")
            raise
        
        total_samples_sent = 0
        device_stats = []

        dynamic_sampling = data_cfg.get('dynamic_sampling', False)
        
        # Process each device
        for device_idx, device_cfg in enumerate(devices):
            device_id = device_cfg['device_id']
            topic = device_cfg['topic']
            data_file = device_cfg.get('data_file', None)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing Device: {device_id} ({device_idx+1}/{len(devices)})")
            logger.info(f"{'='*80}")
            
            # Load data from the specified file - backend will handle train/validation split
            if data_file:
                logger.info(f"  Loading data from file: {data_file}")
                # fetch data for faulty device if device is coinfigured as faulty
                if device_cfg.get('faulty_device', False):
                    x_data, y_data = self.dataset.load_faulty_train_dataset(3000, filename=data_file)
                else:
                    x_data, y_data = self.dataset.load_training_data(3000, filename=data_file)
                logger.info(f"  Loaded {len(x_data)} samples from {data_file}")
            else:
                # Fallback to old method if no data_file specified
                logger.warning(f"  No data_file specified for {device_id}, using default method")
                x_data, y_data = self.dataset.load_training_data(1000)
            
            logger.info(f"  Total samples for {device_id}: {len(x_data)} (backend will split by validation_rate)")
            
            
            # Compute dynamic label weights for this device's data
            if dynamic_sampling:
                logger.info(f"\nComputing dynamic label weights for {device_id}...")
                try:
                    # Calculate weights based on this device's actual data
                    device_label_weights = self.dataset.compute_label_weights_from_data(y_data)
                    device_class_dist = self.dataset.get_class_distribution_from_data(y_data)
                    
                    logger.info(f"  Class distribution for {device_id}: {device_class_dist}")
                    logger.info(f"  Computed label weights for {device_id}: {device_label_weights}")
                    
                    # Update device configuration with computed weights
                    device_cfg['label_weights'] = device_label_weights
                    
                except Exception as e:
                    logger.warning(f"Failed to compute label weights for {device_id}: {e}")
                    logger.warning(f"Using default weights for {device_id}")
                    # Set default equal weights if computation fails
                    device_cfg['label_weights'] = {0: 1.0, 1: 1.0}  # Default for binary classification
            
            # Ensure topic exists
            logger.info(f"\nEnsuring topic exists: {topic}")
            if not self.kafka_admin.ensure_topic_exists(topic):
                raise RuntimeError(f"Failed to create or verify topic: {topic}")
            
            # Create datasource for this device
            logger.info(f"\nCreating FederatedRawSink for device {device_id}")
            logger.info(f"  Topic: {topic}")
            logger.info(f"  Deployment ID: {deployment_id}")
            logger.info(f"  Federated String ID: {federated_string_id!r}")
            
            sink = FederatedRawSink(
                boostrap_servers=data_cfg['bootstrap_servers'],
                topic=topic,
                deployment_id=deployment_id,
                description=device_cfg['description'],
                dataset_restrictions=json.dumps(data_cfg['dataset_restrictions']),
                validation_rate=data_cfg['validation_rate'],
                test_rate=data_cfg['test_rate'],
                control_topic=device_cfg['control_topic'],
                data_type=data_cfg['data_type'],
                data_reshape=data_cfg['data_reshape'],
                label_type=data_cfg['label_type'],
                label_reshape=data_cfg['label_reshape'],
                federated_string_id=federated_string_id,
                label_weights=device_cfg.get('label_weights', None),
                streaming_data_chunks=streaming_chunks
            )
            
            # Send all data for this device - backend will split by validation_rate
            logger.info(f"\nSending {len(x_data)} samples to {device_id}...")
            for i, (x, y) in enumerate(zip(x_data, y_data)):
                sink.send(data=x, label=y)
                if (i + 1) % 1000 == 0:
                    logger.info(f"  Sent {i + 1}/{len(x_data)} samples")
            
            # Close datasource
            logger.info(f"\nClosing datasource for {device_id}...")
            sink.close()
            
            device_samples = len(x_data)
            total_samples_sent += device_samples
            
            device_stats.append({
                'device_id': device_id,
                'topic': topic,
                'total_samples': device_samples
            })
            
            logger.info(f"✓ Device {device_id} data injection completed")
            logger.info(f"  Total samples sent to {device_id}: {device_samples}")
        
        # Summary
        logger.info(f"\n{'='*80}")
        logger.info("Multi-Device Data Injection Summary")
        logger.info(f"{'='*80}")
        for stats in device_stats:
            logger.info(f"  {stats['device_id']}: {stats['total_samples']} samples")
        logger.info(f"  Total samples sent across all devices: {total_samples_sent}")
        logger.info(f"✓ Multi-device data injection completed")
        
        return {
            'total_samples': total_samples_sent,
            'devices': device_stats
        }
    
    def wait_for_training_completion(self, deployment_id: int, result_id: int, max_wait_seconds: int = 3600) -> Dict[str, Any]:
        """
        Wait for training to complete
        
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
        
        start_time = time.time()
        last_status = None
        check_interval = 30
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed > max_wait_seconds:
                logger.warning(f"⚠ Training timeout after {max_wait_seconds}s")
                break
            
            try:
                results_list = self.client.get_deployment_results(deployment_id)
                
                # Find the specific result
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
                
                if status != last_status:
                    logger.info(f"Status: {status}")
                    last_status = status
                
                if status == 'finished':
                    logger.info(f"\n✓ Training completed successfully!")
                    logger.info(f"  Total time: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
                    
                    logger.info(f"  Result data: {result_data}")
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
                    if elapsed % 60 < check_interval:
                        logger.info(f"  Waiting... ({elapsed:.0f}s elapsed)")
                
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error checking training status: {e}")
                time.sleep(check_interval)
        
        # Return partial data if we exit early
        try:
            results_list = self.client.get_deployment_results(deployment_id)
            for result in results_list:
                if result['id'] == result_id:
                    return result
            return {}
        except:
            return {}
    
    def run_inference(self, num_predictions: int = 10) -> List[Dict[str, Any]]:
        """
        Run inference on test data using dataset
        
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
        if not self.kafka_admin.ensure_topic_exists(inference_cfg['input_topic']):
            raise RuntimeError(f"Failed to create input topic: {inference_cfg['input_topic']}")
        if not self.kafka_admin.ensure_topic_exists(inference_cfg['output_topic']):
            raise RuntimeError(f"Failed to create output topic: {inference_cfg['output_topic']}")
        
        # Load test data from dataset for inference
        logger.info(f"Loading test data from dataset for inference...")
        x_test, y_test = self.dataset.load_inference_data(num_predictions, filename='inference_data.txt')
        
        # Create producer and consumer
        logger.info(f"\nCreating Kafka producer for topic: {inference_cfg['input_topic']}")
        producer = KafkaProducer(bootstrap_servers=self.kafka_bootstrap)
        
        # Select random samples
        random_indices = np.random.choice(x_test.shape[0], min(num_predictions, len(x_test)), replace=False)
        actual_labels = []
        
        logger.info(f"\nCreating Kafka consumer for topic: {inference_cfg['output_topic']}")
        consumer = KafkaConsumer(
            inference_cfg['output_topic'],
            bootstrap_servers=self.kafka_bootstrap,
            group_id=f"automation_{int(time.time())}",
            auto_offset_reset='latest',
            consumer_timeout_ms=90000
        )
        
        # Wait for consumer to be ready
        logger.info("Waiting for consumer to be ready...")
        max_wait = 30
        wait_start = time.time()
        
        while not consumer.assignment():
            if time.time() - wait_start > max_wait:
                logger.warning("Consumer partition assignment timeout - proceeding anyway")
                break
            consumer.poll(timeout_ms=100)
            time.sleep(0.1)
        
        if consumer.assignment():
            logger.info(f"✓ Consumer ready and assigned to partitions: {consumer.assignment()}")
        
        # Send images for prediction
        logger.info(f"\nSending {len(random_indices)} images for prediction...")
        for idx, i in enumerate(random_indices):
            producer.send(inference_cfg['input_topic'], x_test[i].tobytes())
            actual_labels.append(int(y_test[i]))
            logger.info(f"  Sent image {idx + 1}/{len(random_indices)} (actual label: {y_test[i]})")
        
        producer.flush()
        producer.close()
        logger.info("✓ All images sent!")
        
        # Receive predictions
        logger.info("\nReceiving predictions:")
        predictions = []
        correct = 0
        
        for idx, msg in enumerate(consumer):
            try:
                # Parse prediction using dataset
                prediction_str = msg.value.decode()
                prediction_obj = json.loads(prediction_str)
                
                predicted, confidence = self.dataset.parse_prediction(prediction_obj)
                
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
                
                if idx + 1 >= len(random_indices):
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
        """Save automation results to file"""
        self.results['end_time'] = datetime.now().isoformat()
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\n✓ Results saved to: {output_file}")
    
    def print_summary(self):
        """Print execution summary"""
        logger.info("\n" + "=" * 80)
        logger.info("AUTOMATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Model ID:           {self.results.get('model_id', 'N/A')}")
        logger.info(f"Configuration ID:   {self.results.get('config_id', 'N/A')}")
        logger.info(f"Deployment ID:      {self.results.get('deployment_id', 'N/A')}")
        logger.info(f"Result ID:          {self.results.get('result_id', 'N/A')}")
        logger.info(f"Inference ID:       {self.results.get('inference_id', 'N/A')}")
        logger.info(f"Federated String:   {self.results.get('federated_string_id', 'N/A')}")
        logger.info(f"Start Time:         {self.results.get('start_time', 'N/A')}")
        logger.info(f"End Time:           {self.results.get('end_time', 'N/A')}")
        logger.info("=" * 80)
    
    def run(self, skip_steps: Optional[List[str]] = None, pause_between_steps: bool = False):
        """
        Run the complete automation pipeline
        
        Args:
            skip_steps: List of step names to skip
            pause_between_steps: If True, wait for user input after each step
        """
        skip_steps = skip_steps or []
        
        def wait_for_user():
            if pause_between_steps:
                input("\n>>> Press Enter to continue to next step... ")
        
        logger.info("\n" + "=" * 80)
        logger.info("KafkaML E2E Automation Pipeline")
        logger.info("=" * 80)
        logger.info(f"Dataset: {self.config['dataset']['type']}")
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Backend URL: {self.config['backend']['url']}")
        logger.info(f"Kafka Bootstrap: {self.kafka_bootstrap}")
        logger.info("=" * 80)
        
        # Pre-flight check
        logger.info("\nPerforming pre-flight check...")
        if not self.client.check_backend_health():
            raise RuntimeError("Backend is not accessible")
        logger.info("Pre-flight check complete\n")
        
        try:
            # Step 1: Create model
            if 'model' not in skip_steps:
                model_id = self.client.create_model(self.config['model'])
                self.results['model_id'] = model_id
                wait_for_user()
            else:
                model_id = self.config.get('existing_ids', {}).get('model_id')
                logger.info(f"Skipping model creation, using existing ID: {model_id}")
            
            # Step 2: Create configuration
            if 'configuration' not in skip_steps:
                config_id = self.client.create_configuration(self.config['configuration'], model_id)
                self.results['config_id'] = config_id
                wait_for_user()
            else:
                config_id = self.config.get('existing_ids', {}).get('config_id')
                logger.info(f"Skipping configuration creation, using existing ID: {config_id}")
            
            # Step 3: Create deployment
            if 'deployment' not in skip_steps:
                deployment_id, federated_string_id, result_id = self.client.create_deployment(
                    self.config['deployment'], config_id
                )
                self.results['deployment_id'] = deployment_id
                self.results['federated_string_id'] = federated_string_id
                self.results['result_id'] = result_id
                
                logger.info("\n  Backend is now:")
                logger.info("    • Creating Kafka topics")
                logger.info("    • Sending model to federated backend")
                logger.info("    • Creating blockchain controller job (if enabled)")
                time.sleep(5)
                
                wait_for_user()
            else:
                deployment_id = self.config.get('existing_ids', {}).get('deployment_id')
                federated_string_id = self.config.get('existing_ids', {}).get('federated_string_id')
                result_id = self.config.get('existing_ids', {}).get('result_id')
                logger.info(f"Skipping deployment creation, using existing ID: {deployment_id}")
            
            # Step 4: Inject data
            if 'data' not in skip_steps:
                logger.info(f"Passing federated_string_id to data injection: {federated_string_id!r}")
                data_stats = self.inject_training_data(deployment_id, federated_string_id)
                wait_for_user()
            else:
                logger.info("Skipping data injection")
            
            # Step 5: Wait for training
            if 'training' not in skip_steps:
                training_result = self.wait_for_training_completion(deployment_id, result_id)
                wait_for_user()
            else:
                logger.info("Skipping training wait")
            
            # Step 6: Create inference
            if 'inference_config' not in skip_steps:
                inference_id = self.client.create_inference(result_id, self.config['inference'])
                self.results['inference_id'] = inference_id
                
                logger.info("\n  Backend is now:")
                logger.info("    • Creating Kubernetes inference deployment")
                logger.info("    • Loading trained model")
                logger.info("    • Starting to consume from input topic")
                
                logger.info("\nWaiting for inference deployment to be ready...")
                time.sleep(10)
                
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


