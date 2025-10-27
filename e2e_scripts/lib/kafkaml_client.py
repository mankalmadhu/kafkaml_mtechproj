"""
KafkaML Backend API Client
"""

import logging
import requests
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class KafkaMLClient:
    """Client for interacting with KafkaML backend API"""
    
    def __init__(self, backend_url: str):
        """
        Initialize KafkaML API client
        
        Args:
            backend_url: Backend URL (e.g., 'http://localhost:9090')
        """
        self.backend_url = backend_url
    
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
    
    def create_model(self, model_config: Dict[str, Any]) -> int:
        """
        Create a new ML model
        
        Args:
            model_config: Model configuration from YAML
            
        Returns:
            Model ID
        """
        logger.info("=" * 80)
        logger.info("STEP 1: Creating Model")
        logger.info("=" * 80)
        
        # Add timestamp to make name unique (max 30 chars)
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        base_name = model_config['name'][:19]
        unique_name = f"{base_name}_{timestamp}"
        
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
        logger.info(f"✓ Model created successfully (Status: {response.status_code})")
        
        # Get the created model by name
        logger.info("Fetching created model details...")
        list_response = self._make_request('GET', '/models/')
        models_list = list_response.json()
        
        for model in models_list:
            if model['name'] == unique_name:
                model_id = model['id']
                logger.info(f"  Model ID: {model_id}")
                logger.info(f"  Name: {model['name']}")
                logger.info(f"  Framework: {model['framework']}")
                return model_id
        
        raise ValueError(f"Could not find created model with name: {unique_name}")
    
    def create_configuration(self, config_cfg: Dict[str, Any], model_id: int) -> int:
        """
        Create a configuration
        
        Args:
            config_cfg: Configuration settings from YAML
            model_id: ID of the model to associate
            
        Returns:
            Configuration ID
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Creating Configuration")
        logger.info("=" * 80)
        
        # Add timestamp to make name unique
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        base_name = config_cfg['name'][:19]
        unique_name = f"{base_name}_{timestamp}"
        
        if len(unique_name) > 30:
            unique_name = unique_name[:30]
        
        payload = {
            "name": unique_name,
            "description": config_cfg['description'],
            "ml_models": [model_id]
        }
        
        logger.info(f"Creating configuration: {unique_name}")
        response = self._make_request('POST', '/configurations/', payload)
        logger.info(f"✓ Configuration created successfully (Status: {response.status_code})")
        
        # Get the created configuration by name
        logger.info("Fetching created configuration details...")
        list_response = self._make_request('GET', '/configurations/')
        configs_list = list_response.json()
        
        for config in configs_list:
            if config['name'] == unique_name:
                config_id = config['id']
                logger.info(f"  Configuration ID: {config_id}")
                logger.info(f"  Name: {config['name']}")
                logger.info(f"  Associated models: {[m['name'] for m in config['ml_models']]}")
                return config_id
        
        raise ValueError(f"Could not find created configuration with name: {unique_name}")
    
    def create_deployment(self, deploy_cfg: Dict[str, Any], config_id: int) -> Tuple[int, Optional[str], int]:
        """
        Create a deployment
        
        Args:
            deploy_cfg: Deployment configuration from YAML
            config_id: ID of the configuration to deploy
            
        Returns:
            (deployment_id, federated_string_id, result_id)
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Creating Deployment")
        logger.info("=" * 80)
        
        # Build tf_kwargs_fit with epochs
        epochs = deploy_cfg.get('epochs', 1)
        tf_kwargs_fit = f"epochs={epochs}, shuffle=True"
        
        payload = {
            "configuration": config_id,
            "batch": deploy_cfg['batch'],
            "optimizer": deploy_cfg.get('optimizer', 'adam'),
            "learning_rate": deploy_cfg.get('learning_rate', 0.001),
            "loss": deploy_cfg.get('loss', 'sparse_categorical_crossentropy'),
            "metrics": deploy_cfg.get('metrics', 'sparse_categorical_accuracy'),
            "federated": deploy_cfg.get('federated', False),
            "blockchain": deploy_cfg.get('blockchain', False),
            "agg_rounds": deploy_cfg.get('agg_rounds', 5),
            "min_data": deploy_cfg.get('min_data', 100),
            "agg_strategy": deploy_cfg.get('agg_strategy', 'FedAvg'),
            "data_restriction": deploy_cfg.get('data_restriction', {}),
            "gpumem": 0,
            "tf_kwargs_fit": tf_kwargs_fit,
            "tf_kwargs_val": "verbose=1",
            "pth_kwargs_fit": "",
            "pth_kwargs_val": ""
        }
        
        logger.info(f"Creating deployment with:")
        logger.info(f"  Federated: {deploy_cfg.get('federated', False)}")
        logger.info(f"  Blockchain: {deploy_cfg.get('blockchain', False)}")
        logger.info(f"  Aggregation rounds: {deploy_cfg.get('agg_rounds', 5)}")
        
        response = self._make_request('POST', '/deployments/', payload)
        logger.info(f"✓ Deployment created successfully (Status: {response.status_code})")
        
        # Get the created deployment
        logger.info("Fetching created deployment details...")
        list_response = self._make_request('GET', '/deployments/')
        deployments_list = list_response.json()
        
        if deployments_list and len(deployments_list) > 0:
            deployment_data = deployments_list[0]
            deployment_id = deployment_data['id']
            
            logger.info(f"  Deployment ID: {deployment_id}")
            logger.info(f"  Configuration: {deployment_data['configuration']['name']}")
            logger.info(f"  Deployment data keys: {list(deployment_data.keys())}")
            
            # Extract federated_string_id
            federated_string_id = deployment_data.get('federated_string_id')
            logger.info(f"  Federated String ID from API: {federated_string_id!r}")
            
            if federated_string_id:
                logger.info(f"  ✓ Federated String ID: {federated_string_id}")
            else:
                logger.warning(f"  ⚠ No Federated String ID in deployment")
            
            # Get result ID
            result_id = None
            if deployment_data.get('results') and len(deployment_data['results']) > 0:
                result_id = deployment_data['results'][0]['id']
                logger.info(f"  Training result ID: {result_id}")
                logger.info(f"  Status: {deployment_data['results'][0]['status']}")
            else:
                logger.warning("No training results found in deployment")
            
            return deployment_id, federated_string_id, result_id
        else:
            raise ValueError("Could not find created deployment")
    
    def get_deployment_results(self, deployment_id: int):
        """
        Get all results for a deployment
        
        Args:
            deployment_id: Deployment ID
            
        Returns:
            List of training results
        """
        response = self._make_request('GET', f'/deployments/results/{deployment_id}')
        return response.json()
    
    def create_inference(self, result_id: int, inference_cfg: Dict[str, Any]) -> Optional[int]:
        """
        Create inference deployment
        
        Args:
            result_id: Training result ID
            inference_cfg: Inference configuration from YAML
            
        Returns:
            Inference ID or None
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Creating Inference Configuration")
        logger.info("=" * 80)
        
        import json as json_lib
        
        payload = {
            "model_result": result_id,
            "replicas": inference_cfg['replicas'],
            "input_format": inference_cfg['input_format'],
            "input_config": json_lib.dumps(inference_cfg['input_config']) if isinstance(inference_cfg['input_config'], dict) else str(inference_cfg['input_config']),
            "input_topic": inference_cfg['input_topic'],
            "output_topic": inference_cfg['output_topic'],
            "gpumem": 0
        }
        
        logger.info(f"Creating inference deployment:")
        logger.info(f"  Result ID: {result_id}")
        logger.info(f"  Input topic: {inference_cfg['input_topic']}")
        logger.info(f"  Output topic: {inference_cfg['output_topic']}")
        logger.info(f"  Replicas: {inference_cfg['replicas']}")
        
        response = self._make_request('POST', f'/results/inference/{result_id}', payload)
        logger.info(f"✓ Inference deployment request accepted (Status: {response.status_code})")
        
        # Get the created inference
        logger.info("Fetching created inference details...")
        list_response = self._make_request('GET', '/inferences/')
        inferences_list = list_response.json()
        
        # Find the inference for our result_id
        for inference in inferences_list:
            logger.info(f"Inference: {inference}")
            if inference['model_result'] == result_id:
                inference_id = inference['id']
                logger.info(f"  Inference ID: {inference_id}")
                logger.info(f"  Status: {inference['status']}")
                return inference_id
        
        logger.warning("Could not find created inference, but deployment was accepted")
        return None
    
    def check_backend_health(self) -> bool:
        """
        Check if backend is accessible
        
        Returns:
            True if backend is accessible
        """
        try:
            test_response = requests.get(f"{self.backend_url}/models/", timeout=10)
            logger.info(f"✓ Backend is accessible (status: {test_response.status_code})")
            return True
        except requests.exceptions.ConnectionError:
            logger.error("✗ Cannot connect to backend!")
            logger.error(f"Please ensure backend is running at: {self.backend_url}")
            return False
        except Exception as e:
            logger.warning(f"Pre-flight check warning: {e}")
            return False

