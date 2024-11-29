def create_dmatrix(X_train, y_train, train_idx, test_idx, win_loss_only = False):
    # pdb.set_trace()
    # Convert train_idx to a boolean mask
    mask = pd.Series(False, index=X_train.index)
    mask.iloc[train_idx] = True  # Set True for indices in train_idx

    # Apply the combined filter to X_train and y_train
    if win_loss_only == False:
        filtered_X_train = X_train[mask]
        filtered_y_train = y_train[mask]
    else:
        filtered_X_train = X_train[mask & (y_train.isin([0, 2]))] # These are the codes for W/L and L/W
        filtered_y_train = y_train[mask & (y_train.isin([0, 2]))]

    # Create the DMatrix
    train_dmatrix = xgb.DMatrix(filtered_X_train, label=filtered_y_train)
    test_dmatrix = xgb.DMatrix(X_train.iloc[test_idx], label=y_train.iloc[test_idx])
    return train_dmatrix, test_dmatrix