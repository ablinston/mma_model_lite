
def predict_xgboost_ensemble(models, X_test_df, y_test_df):
    """
    Generates multi-class predictions from an ensemble of XGBoost models.

    Parameters:
    - models (list): List of trained XGBoost models.
    - X (pd.DataFrame): Feature matrix for making predictions.

    Returns:
    - predictions (np.array): Averaged probabilities for each class from the ensemble.
    - predicted_classes (np.array): Predicted class labels for each instance based on averaged probabilities.
    """
    
    # pdb.set_trace()
    
    # Collect predictions from each model
    model_predictions = []
    for model in models:
        
        # Create the DMatrix for predictions
        dmatrix_test = xgb.DMatrix(X_test_df[model['model'].feature_names], label = y_test_df)
        
        preds = model['model'].predict(dmatrix_test)
        model_predictions.append(preds)
    
    # Stack predictions and average them
    model_predictions = np.array(model_predictions)  # Shape: (num_models, num_samples, num_classes)
    averaged_predictions = np.mean(model_predictions, axis=0)  # Average across models
    
    # Get predicted classes by taking the argmax across the class probabilities
    predicted_classes = np.argmax(averaged_predictions, axis=1)

    return averaged_predictions, predicted_classes

# Example usage
# Assuming you have your ensemble of models trained and a test dataset
# predictions, predicted_classes = predict_ensemble(ensemble_models, X_test)

# `predictions` will contain the averaged probabilities for each class
# `predicted_classes` will contain the predicted class labels based on the highest probability
