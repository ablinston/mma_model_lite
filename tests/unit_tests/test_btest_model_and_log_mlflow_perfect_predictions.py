# Load all the packages and functions needed
import numpy as np
import pandas as pd

from mma_functions.cleaning import clean_target_column
from mma_functions.evaluation import calculate_metrics, calculate_metrics_odds
from mma_functions.pipelines import btest_model_and_log_mlflow

import mlflow
from unittest.mock import patch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
import seaborn as sns
import matplotlib.pyplot as plt

# mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
# mlflow.set_experiment('test')

# Mocking MLflow functions
@patch("mlflow.set_tag")
@patch("mlflow.log_metrics")
@patch("mlflow.log_params")
@patch("mlflow.log_artifact")
def test_btest_model_and_log_mlflow_perfect_predictions(mock_log_artifact, mock_log_params, mock_log_metrics, mock_set_tag):
    # Create synthetic data for perfect predictions
    actuals_data = pd.DataFrame({
        "outcome": ["L/W", "L/W", "W/L", "L/W", "W/L", "W/L", "OTH"],
        "date": pd.to_datetime(["2024-01-01"] * 7),
        "weightclass": ["light", "heavy", "light", "heavy", "light", "heavy", "middle"],
        "fighter1_backtesting_odds": [9e99, 9e99, 1.0000000001, 9e99, 1.0000000001, 1.0000000001, 9e99],
        "fighter2_backtesting_odds": [1.0000000001, 1.0000000001, 9e99, 1.0000000001, 9e99, 9e99, 9e99], 
        "fighter1": ["A"] * 7,
        "fighter2": ["B"] * 7,
    })
    
    prediction_probabilities = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0]
    ])
        
    breakdown_cols = ["weightclass"]
    breakdown_metrics= ["Avg WL only AUC", "X Odds Avg WL only AUC", "Hmean WL only Accuracy", "X Odds Hmean WL only Accuracy", "Logloss", "X Odds Logloss"]
    mlflow_description = "Test perfect predictions"

    # First look at the calculated metrics
    calc_metrics = calculate_metrics(prediction_probabilities, clean_target_column(actuals_data["outcome"]), actuals_data["fighter1_backtesting_odds"], actuals_data["fighter2_backtesting_odds"])
    calc_odds_metrics = calculate_metrics_odds(actuals_data.iloc[0:6,], clean_target_column(actuals_data["outcome"])[0:6])
    
    # Call the function with synthetic data
    # Act
# with mlflow.start_run():
        
    test_results = btest_model_and_log_mlflow(
        actuals_data=actuals_data,
        prediction_probabilities=prediction_probabilities,
        breakdown_cols=breakdown_cols,
        breakdown_metrics=breakdown_metrics,
        mlflow_description=mlflow_description,
        backtesting_params={"min_prob_to_bet": 0, "min_ev_to_bet": -0.01, "max_bet_size": 1},
        log_mlflow=False
    )

    calc_metrics["Logloss"] = np.round(calc_metrics["Logloss"], 8)
    
    # Example validation: Check if "Backtesting_ROI" was logged
    assert calc_metrics == {"Accuracy": 1.0,
     "W/L AUC": 1.0,
     "L/W AUC": 1.0,
     "W/L Accuracy": 1.0,
     "L/W Accuracy": 1.0,
     "Avg WL only AUC": 1.0,
     "Hmean WL only Accuracy": 1.0,
     "Logloss": 0,
     "EV MSE": 0.0,
     "Sample Size": 7}
    
    calc_odds_metrics["X Odds Logloss"] = np.round(calc_odds_metrics["X Odds Logloss"], 8)
    
    assert calc_odds_metrics == {"X Odds Accuracy": 1.0,
     "X Odds W/L AUC": 1.0,
     "X Odds L/W AUC": 1.0,
     "X Odds W/L Accuracy": 1.0,
     "X Odds L/W Accuracy": 1.0,
     "X Odds Avg WL only AUC": 1.0,
     "X Odds Logloss": 0,
     "X Odds EV MSE": 0.0,
     "X Odds Hmean WL only Accuracy": 1.0}

    assert test_results["win_percent"][0] == 100.0
    assert np.round(test_results["profit_loss"].iloc[0], 11) == 6e-10 # odds are all 1.000000001
