def create_dmatrix(X_train, y_train, train_idx, test_idx, win_loss_only = False, split_dec_or_other_filter = False):
    
    # pdb.set_trace()
    # Convert train_idx to a boolean mask
    mask = pd.Series(False, index=X_train.index)
    mask.iloc[train_idx] = True  # Set True for indices in train_idx
    
    # # Reset the index
    # X_train = X_train.reset_index(drop = True)
    # y_train = y_train.reset_index(drop = True)
    
    # Apply the combined filter to X_train and y_train
    if win_loss_only == False:
        if split_dec_or_other_filter:
            filtered_X_train = X_train.copy()[mask & ~(X_train['notformodel_split_dec_or_other'] == 1)]
            filtered_y_train = y_train[mask & ~(X_train['notformodel_split_dec_or_other'] == 1)]
        else:
            filtered_X_train = X_train.copy()[mask]
            filtered_y_train = y_train[mask]
    else:
        if split_dec_or_other_filter:
            filtered_X_train = X_train.copy()[mask & ~(X_train['notformodel_split_dec_or_other'] == 1) & (y_train.isin([0, 2, '0', '2']))] # These are the codes for W/L and L/W
            filtered_y_train = y_train[mask & ~(X_train['notformodel_split_dec_or_other'] == 1) & (y_train.isin([0, 2, '0', '2']))]
        else:
            filtered_X_train = X_train.copy()[mask & (y_train.isin([0, 2, '0', '2']))] # These are the codes for W/L and L/W
            filtered_y_train = y_train[mask & (y_train.isin([0, 2, '0', '2']))]

    # Save the odds for the loss for both test and train data
    train_fighter1_odds = filtered_X_train['fighter1_odds_for_loss']
    train_fighter2_odds = filtered_X_train['fighter2_odds_for_loss']
    
    test_fighter1_odds = X_train.iloc[test_idx]['fighter1_odds_for_loss']
    test_fighter2_odds = X_train.iloc[test_idx]['fighter2_odds_for_loss']

    # Remove odds used for loss from training and test X data
    cols_to_remove = [col for col in X_train.columns if 'for_loss' in col or 'backtesting' in col or 'notformodel' in col]
    filtered_X_train.drop(cols_to_remove, axis = 1, inplace = True)
    X_test = X_train.copy().iloc[test_idx].drop(cols_to_remove, axis = 1)

    # Check if empty
    if len(filtered_X_train) == 0:
        pdb.set_trace()

    # Create the DMatrix
    if split_dec_or_other_filter:
        train_dmatrix = xgb.DMatrix(filtered_X_train, label=filtered_y_train)
        test_dmatrix = xgb.DMatrix(X_test, label=y_train.iloc[test_idx])
    else:
        train_dmatrix = xgb.DMatrix(filtered_X_train, label=filtered_y_train)
        test_dmatrix = xgb.DMatrix(X_test, label=y_train.iloc[test_idx])

    return train_dmatrix, test_dmatrix, train_fighter1_odds, train_fighter2_odds, test_fighter1_odds, test_fighter2_odds