#!/bin/bash

MLFLOW_DIR="/home/andy/Gdrive_MLflow"

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

echo "Cleanup complete."
echo "Starting MLflow server..."

# Start MLflow server
mlflow server --host 192.168.68.51 --port 8080 --backend-store-uri "file://$MLFLOW_DIR/mlruns/" --default-artifact-root "file://$MLFLOW_DIR/mlartifacts/"

