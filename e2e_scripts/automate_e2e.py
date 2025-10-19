#!/usr/bin/env python3
"""
KafkaML E2E Automation - Generic Pipeline Runner

This script runs the complete KafkaML federated learning pipeline for any dataset.
Dataset-specific code is abstracted into dataset classes.

Usage:
    python automate_e2e.py --config configs/mnist_federated.yaml
    python automate_e2e.py --config configs/occupancy_federated.yaml

Author: KafkaML Team
Date: October 19, 2025
"""

import os
import sys
import logging
import argparse
import yaml
from typing import Dict, Any
from datetime import datetime

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from lib.pipeline_runner import PipelineRunner
from datasets import MNISTDataset, OccupancyDataset

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


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_file: Path to config file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def create_dataset(dataset_config: Dict[str, Any]):
    """
    Create dataset instance based on configuration
    
    Args:
        dataset_config: Dataset configuration from YAML
        
    Returns:
        Dataset instance
    """
    dataset_type = dataset_config.get('type', 'mnist').lower()
    
    if dataset_type == 'mnist':
        logger.info("Creating MNIST dataset handler")
        return MNISTDataset()
    elif dataset_type == 'occupancy':
        logger.info("Creating Occupancy dataset handler")
        # Pass data_path if specified in config
        data_path = dataset_config.get('data_path')
        return OccupancyDataset(data_path=data_path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='KafkaML E2E Automation - Generic Pipeline Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run with MNIST dataset
  python automate_e2e.py --config configs/mnist_federated.yaml
  
  # Run with Occupancy dataset
  python automate_e2e.py --config configs/occupancy_federated.yaml
  
  # Skip certain steps
  python automate_e2e.py --config configs/mnist_federated.yaml --skip model,configuration
  
  # Pause between steps
  python automate_e2e.py --config configs/mnist_federated.yaml --pause
        '''
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file (required)'
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
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"✓ Loaded configuration from: {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Create dataset
    try:
        dataset = create_dataset(config['dataset'])
        logger.info(f"✓ Dataset handler created: {config['dataset']['type']}")
    except Exception as e:
        logger.error(f"Failed to create dataset handler: {e}")
        sys.exit(1)
    
    # Parse skip steps
    skip_steps = []
    if args.skip:
        skip_steps = [s.strip() for s in args.skip.split(',')]
        logger.info(f"Skipping steps: {skip_steps}")
    
    # Create and run pipeline
    try:
        pipeline = PipelineRunner(config, dataset)
        pipeline.run(skip_steps=skip_steps, pause_between_steps=args.pause)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

