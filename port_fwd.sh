#!/bin/bash

# Kill existing port forwards if they exist
echo "Killing existing port forwards..."
pkill -f "kubectl port-forward deployment/kafka -n kafkaml 9094:9094" || true
pkill -f "kubectl port-forward -n kafkaml svc/frontend 8080:80" || true
pkill -f "kubectl port-forward -n kafkaml svc/backend  9090:8000" || true

# Wait a moment for processes to be killed
sleep 5

echo "Starting new port forwards..."

kubectl port-forward deployment/kafka -n kafkaml 9094:9094&
kubectl port-forward -n kafkaml svc/frontend 8080:80&
kubectl port-forward -n kafkaml svc/backend  9090:8000&

echo "Port forwards started. Use 'pkill -f kubectl' to stop all port forwards."