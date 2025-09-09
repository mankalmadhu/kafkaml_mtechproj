#!/usr/bin/env python3
"""
Simple Federated Training Job Trigger
Generates and deploys federated training job based on edgeBasedTraining data
"""

import os
import json
import subprocess
import tempfile
import logging
from uuid import uuid4

def trigger_federated_training_job(training):
    """
    Trigger federated training job based on training object data
    
    Args:
        training: Training object from edgeBasedTraining with all necessary data
    """
    
    try:
        logging.info("Triggering federated training job...")
        
        # Extract data from training object
        federated_model_id = training.federated_string_id.lower()
        federated_client_id = str(uuid4().hex[:8])
        
        # Determine training case
        case = determine_training_case(training)
        
        # Extract all template variables from training object
        template_vars = extract_template_variables(training, federated_model_id, federated_client_id, case)
        
        # Generate job YAML
        job_yaml_content = generate_job_yaml(template_vars)
        
        # Deploy job
        deploy_job(job_yaml_content)
        
        logging.info(f"Federated training job triggered successfully!")
        logging.info(f"Job name: federated-training-{federated_model_id}-worker-{federated_client_id}")
        
    except Exception as e:
        logging.error(f"Failed to trigger federated training job: {str(e)}")

def determine_training_case(training):
    """Determine training case based on training object"""
    
    # Check if blockchain training
    if hasattr(training, 'blockchain') and training.blockchain:
        return "5"
    
    # Check if distributed training
    distributed = hasattr(training, 'N')  # DistributedFederatedTraining has N attribute
    incremental = hasattr(training, 'stream_timeout')  # Incremental training has stream_timeout
    
    if not distributed:
        if not incremental:
            return "1"  # Standard federated training
        else:
            return "2"  # Incremental federated training
    else:
        if not incremental:
            return "3"  # Distributed federated training
        else:
            return "4"  # Distributed incremental federated training

def extract_template_variables(training, federated_model_id, federated_client_id, case):
    """Extract all template variables from training object and environment variables"""
    
    # Start with core variables that are always available
    template_vars = {
        'FEDERATED_MODEL_ID': federated_model_id,
        'FEDERATED_CLIENT_ID': federated_client_id,
        'CASE': case
    }
    
    # Extract from training object attributes
    training_attributes = {
        'bootstrap_servers': ['DATA_BOOTSTRAP_SERVERS', 'KML_CLOUD_BOOTSTRAP_SERVERS'],
        'data_topic': ['DATA_TOPIC'],
        'unsupervised_topic': ['UNSUPERVISED_TOPIC'],
        'input_format': ['INPUT_FORMAT'],
        'input_config': ['INPUT_CONFIG'],
        'validation_rate': ['VALIDATION_RATE'],
        'test_rate': ['TEST_RATE'],
        'total_msg': ['TOTAL_MSG'],
        'kube_namespace': ['KUBE_NAMESPACE'],
        'training_image': ['TRAINING_IMAGE'],
        'nvidia_visible_devices': ['NVIDIA_VISIBLE_DEVICES']
    }
    
    # Extract from training object
    for attr_name, template_keys in training_attributes.items():
        if hasattr(training, attr_name):
            value = getattr(training, attr_name)
            if value is not None:
                for template_key in template_keys:
                    template_vars[template_key] = str(value) if not isinstance(value, str) else value
    
    # Environment variable mappings (these override training object values)
    env_mappings = {
        'KUBE_NAMESPACE': 'KUBE_NAMESPACE',
        'TRAINING_IMAGE': 'TRAINING_IMAGE',
        'KML_CLOUD_BOOTSTRAP_SERVERS': 'KML_CLOUD_BOOTSTRAP_SERVERS',
        'DATA_BOOTSTRAP_SERVERS': 'DATA_BOOTSTRAP_SERVERS',
        'DATA_TOPIC': 'DATA_TOPIC',
        'UNSUPERVISED_TOPIC': 'UNSUPERVISED_TOPIC',
        'INPUT_FORMAT': 'INPUT_FORMAT',
        'INPUT_CONFIG': 'INPUT_CONFIG',
        'VALIDATION_RATE': 'VALIDATION_RATE',
        'TEST_RATE': 'TEST_RATE',
        'TOTAL_MSG': 'TOTAL_MSG',
        'NVIDIA_VISIBLE_DEVICES': 'NVIDIA_VISIBLE_DEVICES',
        'ETH_RPC_URL': 'ETH_RPC_URL',
        'ETH_CONTRACT_ADDRESS': 'ETH_CONTRACT_ADDRESS',
        'ETH_CONTRACT_ABI': 'ETH_CONTRACT_ABI',
        'ETH_WALLET_ADDRESS': 'ETH_WALLET_ADDRESS',
        'ETH_WALLET_KEY': 'ETH_WALLET_KEY'
    }
    
    # Override with environment variables
    for template_key, env_key in env_mappings.items():
        if env_key in os.environ:
            template_vars[template_key] = os.environ[env_key]
    
    # Handle blockchain variables for Case 5
    if case == "5" and hasattr(training, 'blockchain') and training.blockchain:
        blockchain_config = training.blockchain
        template_vars['ETH_RPC_URL'] = blockchain_config.get('rpc_url', '')
        template_vars['ETH_CONTRACT_ADDRESS'] = blockchain_config.get('contract_address', '')
        template_vars['ETH_CONTRACT_ABI'] = json.dumps(blockchain_config.get('contract_abi', {}))
        template_vars['ETH_WALLET_ADDRESS'] = os.environ.get('FEDML_BLOCKCHAIN_WALLET_ADDRESS', '')
        template_vars['ETH_WALLET_KEY'] = os.environ.get('FEDML_BLOCKCHAIN_WALLET_KEY', '')
    
    # Log what values we're using
    logging.info("Template variables extracted:")
    for key, value in template_vars.items():
        logging.info(f"  {key}: {value}")
    
    return template_vars

def generate_job_yaml(template_vars):
    """Generate job YAML content from template with all variables"""
    
    # Read template file
    template_file = os.path.join(os.path.dirname(__file__), 'federated_training_job_template.yaml')
    
    if not os.path.exists(template_file):
        raise FileNotFoundError(f"Template file not found: {template_file}")
    
    with open(template_file, 'r') as f:
        template_content = f.read()
    
    # Check for missing required variables
    required_vars = ['FEDERATED_MODEL_ID', 'FEDERATED_CLIENT_ID', 'CASE']
    missing_vars = []
    for var in required_vars:
        if var not in template_vars:
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required template variables: {missing_vars}")
    
    # Substitute all variables
    for key, value in template_vars.items():
        placeholder = f'${{{key}}}'
        template_content = template_content.replace(placeholder, str(value))
    
    # Handle special cases with default values (only if not already substituted)
    default_mappings = {
        '${KUBE_NAMESPACE:-kafkaml}': template_vars.get('KUBE_NAMESPACE', 'kafkaml'),
        '${UNSUPERVISED_TOPIC:-""}': template_vars.get('UNSUPERVISED_TOPIC', ''),
        '${NVIDIA_VISIBLE_DEVICES:-all}': template_vars.get('NVIDIA_VISIBLE_DEVICES', 'all'),
        '${ETH_RPC_URL:-""}': template_vars.get('ETH_RPC_URL', ''),
        '${ETH_CONTRACT_ADDRESS:-""}': template_vars.get('ETH_CONTRACT_ADDRESS', ''),
        '${ETH_CONTRACT_ABI:-""}': template_vars.get('ETH_CONTRACT_ABI', ''),
        '${ETH_WALLET_ADDRESS:-""}': template_vars.get('ETH_WALLET_ADDRESS', ''),
        '${ETH_WALLET_KEY:-""}': template_vars.get('ETH_WALLET_KEY', '')
    }
    
    for placeholder, default_value in default_mappings.items():
        template_content = template_content.replace(placeholder, str(default_value))
    
    # Check for any remaining unresolved variables
    import re
    unresolved_vars = re.findall(r'\$\{([^}]+)\}', template_content)
    if unresolved_vars:
        logging.warning(f"Unresolved template variables: {unresolved_vars}")
        logging.warning("These will be left as-is in the generated YAML")
    
    return template_content

def deploy_job(job_yaml_content):
    """Deploy job using kubectl"""
    
    try:
        # Create temporary file for job YAML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(job_yaml_content)
            temp_file = f.name
        
        # Run kubectl apply
        cmd = ['kubectl', 'apply', '-f', temp_file]
        logging.info(f"Deploying job with: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Clean up temporary file
        os.unlink(temp_file)
        
        logging.info(f"Job deployed successfully: {result.stdout}")
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to deploy job: {e.stderr}")
        # Clean up temporary file if it exists
        if 'temp_file' in locals():
            try:
                os.unlink(temp_file)
            except:
                pass
        raise
    except Exception as e:
        logging.error(f"Error deploying job: {str(e)}")
        raise

if __name__ == '__main__':
    # Test the module
    class MockTraining:
        def __init__(self):
            self.federated_string_id = "test123"
            self.blockchain = {}
            self.bootstrap_servers = "kafka:9092"
            self.data_topic = "FED-DEBUG-data_topic"
            self.input_format = "RAW"
            self.input_config = '{"type": "float32", "data_type": "float32", "data_reshape": "784", "label_type": "float32", "label_reshape": "10"}'
            self.validation_rate = 0.1
            self.test_rate = 0.1
            self.total_msg = 500
    
    mock_training = MockTraining()
    trigger_federated_training_job(mock_training)
