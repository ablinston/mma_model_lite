def get_base_margin(df, base_margin_cols):
    
    """
    This function takes a data frame and a list of columns of odds to be used as probabilities and returns.
    This function assumes there are 3 classes of outcome
    
    Parameters
    ----------
    df: the data frame
    
    base_margin_cols: an ordered list of the odds columns with the order corresponding to how the labels in the target variable to be modelled are ordered. Can also set a flat rate by inputing a dictionary instead of a column name like this `{'flat_prob': 0.01}`
    
    """
    
    # pdb.set_trace()
    
    # Check if there are any cols
    if base_margin_cols == None:        
        return np.zeros(3 * len(df))
    else:
    
        cols = []
        for spec in base_margin_cols:
            if isinstance(spec, dict) and "flat_prob" in spec:
                cols.append(np.full(len(df), spec["flat_prob"]))
            else:
                cols.append(1.0 / df[spec].to_numpy())
        
        # stack into array of shape (n_classes, n_rows) → transpose → (n_rows, n_classes)
        p = np.vstack(cols).T
    
        # Make into an array and clip to avoid 0 and 1
        eps = 1e-9
        p = np.clip(np.array(p), eps, 1 - eps)
    
        p = p / p.sum(axis=1, keepdims=True)  # ensure rows sum to 1
        
        # flatten row-major for XGBoost
        rv = np.log(p).ravel()
        
        # Check length
        stop_if_not(len(rv) == 3 * len(df), "Base margin length check")
        
        return rv  