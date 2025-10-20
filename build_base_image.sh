#!/bin/bash

# Build the base image with Python 3.10, Solidity, and Anvil
# This only needs to be built once and can be reused

echo "Building Kafka-ML base image with Python 3.10, Solidity 0.8.20, and Anvil..."
echo "This may take 15-20 minutes on first build, but only needs to be done once."

docker build --platform linux/arm64 -t kafkaml-base:latest -f Dockerfile.base .

if [ $? -eq 0 ]; then
    echo "✅ Base image built successfully!"
    echo "Now you can build other images much faster using:"
    echo "  - docker build -t kafkaml-backend:latest backend/"
    echo "  - docker build -t kafkaml-training:latest model_training/tensorflow/"
else
    echo "❌ Failed to build base image"
    exit 1
fi

