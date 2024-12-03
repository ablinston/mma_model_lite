# This function creates a dmatrix ready for XGBoost, but it also applies a filter to the training data (not test) to remove any split decisions or 'other' methods of victory.
def create_dmatrix_with_split_other_filter(X_train, y_train, train_idx, test_idx, win_loss_only = False):
    # pdb.set_trace()
    # Convert train_idx to a boolean mask
    mask = pd.Series(False, index=X_train.index)
    mask.iloc[train_idx] = True  # Set True for indices in train_idx

    # Apply the combined filter to X_train and y_train
    if win_loss_only == False:
        filtered_X_train = X_train[mask & ~(X_train['split_dec_or_other'] == 1)]
        filtered_y_train = y_train[mask & ~(X_train['split_dec_or_other'] == 1)]
    else:
        filtered_X_train = X_train[mask & ~(X_train['split_dec_or_other'] == 1) & (y_train.isin([0, 2]))] # These are the codes for W/L and L/W
        filtered_y_train = y_train[mask & ~(X_train['split_dec_or_other'] == 1) & (y_train.isin([0, 2]))]

    # Create the DMatrix
    train_dmatrix = xgb.DMatrix(filtered_X_train.drop(['split_dec_or_other'], axis = 1), label=filtered_y_train)
    test_dmatrix = xgb.DMatrix(X_train.iloc[test_idx].drop(['split_dec_or_other'], axis = 1), label=y_train.iloc[test_idx])
    return train_dmatrix, test_dmatrix