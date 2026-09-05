#!/bin/bash
# Script to run the kngraph FastAPI server in Docker
# Usage: ./run_server.sh

set -e

# Build the Docker image
echo "🐳 Building Docker container for kngraph scripts..."
docker build -f Dockerfile -t kngraph-dedup-demo .
echo "✅ Docker image built successfully!"

# Run the container with volume mounts
echo "🚀 Running kngraph FastAPI server in Docker container"
docker run --rm \
    -v "$(pwd):/workspace" \
    -v "$(pwd)/output:/workspace/output" \
    -v "$(pwd)/logs:/workspace/logs" \
    -p 8000:8000 \
    --env-file .env \
    kngraph-dedup-demo uvicorn app.server:app --reload --host 0.0.0.0 --port 8000
