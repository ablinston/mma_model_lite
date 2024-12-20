
def calculate_metrics(y_pred_prob, y_test_df, classes = np.array([0, 1, 2])):
    
    """
    Take a set of predictions and actuals and calculates a series of metrics
    
    Args:
        y_pred_prob (array): an array of probabilities for the predictions
        y_test_df: a series of actuals to test against
        classes (array): the classes of the predicted variable
        
    Returns:
        A dictionary of metrics
    
    """
    # pdb.set_trace()

    y_pred = np.argmax(y_pred_prob, axis = 1)

    # Convert actual results to binarized format
    y_test_bin = label_binarize(y_test_df, classes = classes)
    y_pred_bin = label_binarize(y_pred, classes = classes)

    # AUC fails when only one class in observations, so try except
    try:
        # Calculate the AUC of wins and losses. Check the classes of the target column. Was 1 = L/W and 3 = W/L
        wl_auc = roc_auc_score(y_test_bin[:, 2], y_pred_prob[:, 2])
        lw_auc = roc_auc_score(y_test_bin[:, 0], y_pred_prob[:, 0])
    except:
        wl_auc = np.nan
        lw_auc = np.nan

    # Calculate the accuracies
    wl_pred = y_pred_bin[:, 2]
    wl_test = y_test_bin[:, 2]
    lw_pred = y_pred_bin[:, 0]
    lw_test = y_test_bin[:, 0]

    wl_accuracy = (wl_pred[wl_pred == 1] == wl_test[wl_pred == 1]).sum() / wl_pred.sum()
    lw_accuracy = (lw_pred[lw_pred == 1] == lw_test[lw_pred == 1]).sum() / lw_pred.sum()

    # **Calculate LogLoss**
    log_loss_result = log_loss(y_test_df, y_pred_prob)

    # Return the metrics    
    return {
        'Accuracy': accuracy_score(y_test_df, y_pred),
        'W/L AUC': wl_auc,
        'L/W AUC': lw_auc,
        'W/L Accuracy': wl_accuracy,
        'L/W Accuracy': lw_accuracy,
        'Avg WL only AUC': np.mean([wl_auc, lw_auc]),
        'Hmean WL only Accuracy': hmean([lw_accuracy, wl_accuracy]),
        'Logloss': log_loss_result,
        'Sample Size': len(y_pred)
    }
