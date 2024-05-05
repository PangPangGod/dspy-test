#!/bin/bash

# Set Docker Desktop execution path (Docker Desktop for Linux does not exist; this example assumes Docker is installed)
dockerDesktopPath="/usr/bin/docker" ### TODO : MAC이나 WSL에서 DOCKER 설치해서 이용해보고 수정

if [ -f "$dockerDesktopPath" ]; then
    echo "Starting Docker..."
else
    echo "Cannot find Docker."
    exit 1
fi

echo "Waiting for Docker service to start..."
sleep 5

# Docker image execution
docker run -p 8000:8000 -d --rm --name unstructured-api -e UNSTRUCTURED_PARALLEL_MODE_THREADS=3 downloads.unstructured.io/unstructured-io/unstructured-api:latest --port 8000 --host 0.0.0.0
echo "Docker 'unstructured-api' image has been successfully launched."

## if conda
# Activate conda environment (assuming conda is installed and configured)
source activate dspy
python documenthandle.py

### Cleanup
docker kill unstructured-api
echo "Deleted used Docker image"
dockerProcessName="docker"
dockerProcess=$(pgrep -f $dockerProcessName)

if [ "$dockerProcess" ]; then
    kill -9 $dockerProcess
    echo "Docker process has been terminated."
else
    echo "Cannot find Docker process."
fi
