# This function takes the dataset and calculates the set of metrics based on the odds only

def calculate_metrics_odds(X_test, y_test, classes = np.array([0, 1, 2])):

    # First get the predictions from the ensemble
    y_pred_prob = odds_to_probabilities(X_test['fighter1_best_closing_odds'], X_test['fighter2_best_closing_odds'])

    # Fill in nas
    y_pred_prob['L/W'] = y_pred_prob['L/W'].fillna(0.499)
    y_pred_prob['OTH'] = y_pred_prob['OTH'].fillna(0)
    y_pred_prob['W/L'] = y_pred_prob['W/L'].fillna(0.501)

    # Convert the probabilities to predictions
    y_pred = np.argmax(y_pred_prob, axis = 1)

    # Convert the probabilities into a Numpy array
    y_pred_prob = y_pred_prob.to_numpy()

    # Convert actual results to binarized format
    y_test_bin = label_binarize(y_test, classes=classes)
    y_pred_bin = label_binarize(y_pred, classes=classes)

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

    # Return the metrics
    
    return {
        'X Odds Accuracy': accuracy_score(y_test, y_pred),
        'X Odds W/L AUC': wl_auc,
        'X Odds L/W AUC': lw_auc,
        'X Odds W/L Accuracy': wl_accuracy,
        'X Odds L/W Accuracy': lw_accuracy,
        'X Odds Avg WL only AUC': np.mean([wl_auc, lw_auc]),
        'X Odds Hmean WL only Accuracy': hmean([lw_accuracy, wl_accuracy])
    }