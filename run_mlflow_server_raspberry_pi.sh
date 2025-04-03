#!/bin/bash

MLFLOW_DIR="/home/andy/MLflow"

echo "Cleaning up MLflow garbage..."

# Delete MLflow runs from .trash
if [ -d "$MLFLOW_DIR/mlruns/.trash" ]; then
    echo "Deleting MLflow runs trash..."
    rm -rf "$MLFLOW_DIR/mlruns/.trash"
fi

# Delete MLflow artifacts from .trash
if [ -d "$MLFLOW_DIR/mlartifacts/.trash" ]; then
    echo "Deleting MLflow artifacts trash..."
    rm -rf "$MLFLOW_DIR/mlartifacts/.trash"
fi

# Load the python environment
source /home/andy/miniforge3/etc/profile.d/conda.sh
conda activate mma_model_pi

# Clear garbage using mlflow
mlflow gc --backend-store-uri "sqlite:///$MLFLOW_DIR/mlflow.db"

echo "Cleanup complete."
echo "Starting MLflow server..."

# Start MLflow server
# mlflow server --host 192.168.68.51 --port 8080 --backend-store-uri "sqlite:///$MLFLOW_DIR/mlflow.db" --default-artifact-root "file://$MLFLOW_DIR/mlartifacts/"
mlflow server --host 192.168.68.51 --port 8080 --backend-store-uri "file://$MLFLOW_DIR/mlruns/" --default-artifact-root "file://$MLFLOW_DIR/mlartifacts/"

exec bash