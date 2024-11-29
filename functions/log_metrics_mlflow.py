
def log_metrics_mlflow(y_pred_prob, y_test_df, classes):
    
    """
    Take a set of predictions and actuals and log all required metrics in MLflow
    
    Args:
        y_pred_prob (array): an array of probabilities for the predictions
        y_test_df: a series of actuals to test against
        classes (array): the classes of the predicted variable
        
    Returns:
        Nothing, but the metrics will be logged in MLflow
    
    """
    # pdb.set_trace()

    # Calculate the metrics
    metrics = calculate_metrics(y_pred_prob, y_test_df, classes)
    
    mlflow.log_metrics(metrics)
    

    return None
