# This function takes an h2o autoML results object and saves the outcomes to an MLflow experiment

def h2o_automl_to_mlflow(automl_models):
    
    # Loop through the models in the object
    for model in automl_models:
        # Begin a new model log in mlflow for the model
        with mlflow.start_run():
            # Log the stats and description for the model
            mlflow.log_param('model', x)
            mlflow.log_metric('RMSE', rmse)
            mlflow.log_metric('AUC', auc)

    return "Models saved to MLflow"
