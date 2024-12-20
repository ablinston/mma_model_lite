# This function takes a custom xgboost ensemble model and logs the metrics AUC and accuracy in MLflow

def log_metrics_xgboost_ensemble_mlflow(models, X_test_df, y_test_df):

    # First get the predictions from the ensemble
    y_pred_prob, y_pred = predict_xgboost_ensemble(models, X_test_df, y_test_df)

    # Get the outcomes of the test data
    y_test_fold = y_test_df.copy()

    # Convert actual results to binarized format
    y_test_bin = label_binarize(y_test_fold, classes=np.unique(y_test_fold))
    y_pred_bin = label_binarize(y_pred, classes=np.unique(y_test_fold))

    # Calculate the AUC of wins and losses. Check the classes of the target column. Was 1 = L/W and 3 = W/L
    wl_auc = roc_auc_score(y_test_bin[:, 2], y_pred_prob[:, 2])
    lw_auc = roc_auc_score(y_test_bin[:, 0], y_pred_prob[:, 0])

    # Calculate the accuracies
    wl_pred = y_pred_bin[:, 2]
    wl_test = y_test_bin[:, 2]
    lw_pred = y_pred_bin[:, 0]
    lw_test = y_test_bin[:, 0]

    wl_accuracy = (wl_pred[wl_pred == 1] == wl_test[wl_pred == 1]).sum() / wl_pred.sum()
    lw_accuracy = (lw_pred[lw_pred == 1] == lw_test[lw_pred == 1]).sum() / lw_pred.sum()

    # **Calculate LogLoss**
    log_loss_result = log_loss(y_test_df, y_pred_prob)

    # Log the outcome in MLflow
    mlflow.log_metric('Accuracy', accuracy_score(y_test_fold, y_pred))
    mlflow.log_metric('W/L AUC', wl_auc)
    mlflow.log_metric('L/W AUC', lw_auc)
    mlflow.log_metric('W/L Accuracy', wl_accuracy)
    mlflow.log_metric('L/W Accuracy', lw_accuracy)
    mlflow.log_metric('Avg WL only AUC', np.mean([wl_auc, lw_auc]))
    mlflow.log_metric('Hmean WL only Accuracy', hmean([lw_accuracy, wl_accuracy]))
    mlflow.log_metric('Logloss', log_loss_result)
