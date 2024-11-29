# This function takes the dataset and logs the metrics based purely on the odds in the data

def log_metrics_odds_mlflow(X_test, y_test):

    # Calculate the metrics
    metrics = calculate_metrics_odds(X_test, y_test) 

    # Log the metrics
    mlflow.log_metrics(metrics)
    
    return None