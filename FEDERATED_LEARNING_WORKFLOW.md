# Kafka ML Federated Learning End-to-End Workflow

This document provides a comprehensive guide for running federated learning training using the Kafka ML platform.

## Overview

The Kafka ML federated learning system implements a distributed training approach where:
- A central server coordinates the training process
- Multiple federated clients train on local data
- Model weights are aggregated using FedAvg algorithm
- Training progresses through multiple aggregation rounds

## Architecture Components

### Core Components
- **Kafka ML Backend**: Django-based backend managing models and configurations
- **Kafka ML Frontend**: Angular web interface for model management
- **Federated Model Training**: Kubernetes job for federated training execution
- **Kafka Topics**: Message passing system for model control and data
- **Docker Registry**: Container storage for all components

### Key Kafka Topics
- `model_control_topic`: Model architecture and control messages
- `model_data_topic`: Model weights (each layer to same partition, different offset)
- `FED-DEBUG-data_topic`: Training data injection topic
- `federated_aggregator_topic`: Aggregated weights from federated clients

## Step-by-Step Workflow

### Phase 1: Infrastructure Setup

#### 1. Build and Push Docker Images
```bash
# Use the build script to build and push all components
./build.sh
```
This builds and pushes:
- Backend services
- Frontend application
- Federated training components
- Model training executors

#### 2. Deploy Kubernetes Components
```bash
# Apply kustomize configuration for local deployment
kubectl apply -k kustomize/local/
```
This launches:
- Backend pods and services
- Kafka cluster
- Model training components
- All necessary deployments

#### 3. Setup Port Forwarding
```bash
# Run the port forwarding script
./port_fwd.sh
```
Establishes connectivity to:
- Frontend: localhost:8080
- Backend: localhost:8000
- Kafka: localhost:31162

### Phase 2: Model Configuration

#### 4. Access Frontend Interface
- Open browser to `http://localhost:8080`
- Login to Kafka ML web interface

#### 5. Create TensorFlow Model
- Navigate to Models section
- Create new TensorFlow model
- Upload model architecture/definition
- Model is stored in `kafkaml/backend`

#### 6. Create Model Configuration
- Associate configuration with the created model
- Set training parameters
- Configure federated learning settings

#### 7. Create Federated Deployment
- Enable "Federated Learning" checkbox
- Configure federated parameters:
  - Aggregation rounds
  - Aggregation strategy (FedAvg)
  - Client configuration
- Submit deployment

### Phase 3: Training Orchestration

#### 8. Model Training Job Creation
The backend automatically:
- Creates necessary Kafka topics
- Emits model architecture to `model_control_topic`
- Sends initial model weights to `model_data_topic`
- Generates unique federated string ID
- Creates `federated_aggregator_topic`

### Phase 4: Data Injection

#### 9. Inject MNIST Training Data
```bash
cd federated-module/federated_learn_inject_scripts/
python inject_dummy_training_data.py
```

**Data Format:**
- Training samples: 500 MNIST images
- Test samples: 50 MNIST images
- Image format: 784 bytes (28x28 flattened, normalized)
- Label format: 40 bytes (one-hot encoded, 10 classes)
- Topic: `FED-DEBUG-data_topic`

### Phase 5: Federated Training Execution

#### 10. Launch Federated Training Job
```bash
cd federated-module/
kubectl apply -f debug_training_job.yaml
```

**Job Configuration:**
- Image: `host.docker.internal:50000/federated_model_training_tensorflow:latest`
- Federated Model ID: `ddy6heod` (from model training job)
- Client ID: `test123`
- Data Topic: `FED-DEBUG-data_topic`
- Input Format: RAW

### Phase 6: Aggregation Loop

#### 11. Training Process
The federated training job:
- Reads model control messages from `model_control_topic`
- Reads training data from `FED-DEBUG-data_topic`
- Trains model on local data
- Sends trained weights to `federated_aggregator_topic`
- Reports training metrics to Kafka ML backend

#### 12. Weight Aggregation
Kafka ML backend:
- Receives weights from federated clients
- Applies FedAvg aggregation algorithm
- Updates model weights
- Sends aggregated weights back to `model_control_topic`

#### 13. Iteration
Process repeats for specified aggregation rounds:
- Each round improves model accuracy
- Metrics are collected and reported
- Model version is incremented

### Phase 7: Completion

#### 14. Final Model
- Training completes after all aggregation rounds
- Final aggregated model is available for inference
- Training metrics are stored in backend
- Model can be deployed for production use

## Technical Details

### FedAvg Aggregation Algorithm
```python
def aggregate_model(model, trained_model, aggregation_strategy, control_msg, model_metrics):
    if aggregation_strategy == 'FedAvg':
        weights = [model.get_weights(), trained_model.get_weights()]
        new_weights = list()
        for weights_list_tuple in zip(*weights): 
            new_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )
        model.set_weights(new_weights)
```

### Data Format Specifications
- **Image Data**: 784 bytes (28x28 pixels, float32, normalized 0-1)
- **Label Data**: 40 bytes (10 classes, one-hot encoded, float32)
- **Kafka Message**: Image as value, label as key

### Environment Variables
```yaml
KML_CLOUD_BOOTSTRAP_SERVERS: kafka:9092
DATA_BOOTSTRAP_SERVERS: kafka:9092
DATA_TOPIC: FED-DEBUG-data_topic
INPUT_FORMAT: RAW
INPUT_CONFIG: '{"type": "float32", "data_type": "float32", "data_reshape": "784", "label_type": "float32", "label_reshape": "10"}'
FEDERATED_MODEL_ID: ddy6heod
FEDERATED_CLIENT_ID: test123
```

## Configuration Files

### debug_training_job.yaml
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: federated-training-debug-worker-test123
  namespace: kafkaml
spec:
  ttlSecondsAfterFinished: 100
  template:
    spec:
      containers:
      - image: host.docker.internal:50000/federated_model_training_tensorflow:latest
        name: training
        env:
        - name: KML_CLOUD_BOOTSTRAP_SERVERS
          value: kafka:9092
        - name: DATA_BOOTSTRAP_SERVERS
          value: kafka:9092
        - name: DATA_TOPIC
          value: FED-DEBUG-data_topic
        - name: INPUT_FORMAT
          value: RAW
        - name: INPUT_CONFIG
          value: '{"type": "float32", "data_type": "float32", "data_reshape": "784", "label_type": "float32", "label_reshape": "10"}'
        - name: FEDERATED_MODEL_ID
          value: ddy6heod
        - name: FEDERATED_CLIENT_ID
          value: test123
        imagePullPolicy: Always
      restartPolicy: OnFailure
  backoffLimit: 1
```

## Troubleshooting

### Common Issues

#### 1. Port Forwarding Problems
```bash
# Check if ports are properly forwarded
netstat -an | grep 8080
netstat -an | grep 31162
```

#### 2. Kafka Connection Issues
```bash
# Check Kafka pod status
kubectl get pods -n kafkaml | grep kafka

# Check Kafka logs
kubectl logs -n kafkaml <kafka-pod-name>
```

#### 3. Federated Training Job Failures
```bash
# Check job status
kubectl get jobs -n kafkaml

# Check job logs
kubectl logs -n kafkaml job/federated-training-debug-worker-test123
```

#### 4. Data Injection Issues
- Verify Kafka bootstrap servers: `localhost:31162`
- Check topic exists: `FED-DEBUG-data_topic`
- Ensure data format matches expected schema

### Monitoring Commands

#### Check System Status
```bash
# Check all pods
kubectl get pods -n kafkaml

# Check services
kubectl get services -n kafkaml

# Check Kafka topics
kubectl exec -n kafkaml <kafka-pod> -- kafka-topics --list --bootstrap-server localhost:9092
```

#### View Logs
```bash
# Backend logs
kubectl logs -n kafkaml <backend-pod>

# Federated training logs
kubectl logs -n kafkaml job/federated-training-debug-worker-test123

# Kafka logs
kubectl logs -n kafkaml <kafka-pod>
```

## Known Limitations

1. **Hardcoded Values**: Debug job uses hardcoded federated model ID and client ID
2. **Manual Execution**: `kubectl apply` step requires manual intervention
3. **Single Client**: Current setup uses one federated client
4. **Topic Management**: Manual creation of `FED-DEBUG-data_topic`
5. **Error Handling**: Limited error handling in aggregation loop

## Future Improvements

1. **Automation**: Automate job creation and execution
2. **Multi-Client**: Support multiple federated clients
3. **Dynamic Configuration**: Remove hardcoded values
4. **Enhanced Monitoring**: Better visibility into training progress
5. **Error Recovery**: Improved error handling and recovery mechanisms

## References

- [Kafka ML Documentation](../README.md)
- [Federated Learning Module](../federated-module/README.md)
- [Model Training Components](../model_training/)
- [Data Sources](../datasources/)

---

**Last Updated**: December 2024  
**Version**: 1.0  
**Author**: Kafka ML Team
