@echo off
set "MLFLOW_DIR=E:\MLflow - old\"

echo Cleaning up MLflow garbage...

:: Delete MLflow runs from .trash
if exist "%MLFLOW_DIR%\mlruns\.trash" (
    echo Deleting MLflow runs trash...
    rmdir /s /q "%MLFLOW_DIR%\mlruns\.trash"
)

:: Delete MLflow artifacts from .trash
if exist "%MLFLOW_DIR%\mlartifacts\.trash" (
    echo Deleting MLflow artifacts trash...
    rmdir /s /q "%MLFLOW_DIR%\mlartifacts\.trash"
)

echo Cleanup complete.
echo Starting MLflow server...

mlflow server --host 0.0.0.0 --port 8000 --backend-store-uri "file:///%MLFLOW_DIR%/mlruns/" --default-artifact-root "file:///%MLFLOW_DIR%/mlartifacts/"

pause
