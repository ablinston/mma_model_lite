# This script fits XGBoost models to the final data


import mlflow

# Read in global objects and packages
exec(open('global.py').read())


# Set up experiment training
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment('final_models_oot_testing_v4')


# Get the artifact path using the Train MLflow run in oottesting
train_run_id = 'a7f08a3d5569460b93722b0b7d9dfdff'

# If the model is good, then log it in the registery
mlflow.register_model('runs:/' + train_run_id + '/artifacts/Model_object', name = 'xgb_forest_model')


# test = mlflow.pyfunc.load_model('models:/xgb_forest_model/5')
