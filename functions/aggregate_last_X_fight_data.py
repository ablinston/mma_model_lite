def aggregate_last_X_fight_data(df, id_cols, group_by, prefix, X, per_minute_calc = False, bayes_priors_df = None, bayes_alpha_plus_beta_rebase = None):
    """
    Convert fight data into cumulative fight history for the last 3 entries (excluding the current entry).
    THis version of the functions uses polars for the underlying calcs

    arguments:
    df: a data frame with one row for each fighter
    id_cols: the columns not to be aggregated
    group_by: a list of columns to be aggregated by
    prefix: a str to be added to the start of stat columns
    X: the number of historic fights to include
    per_minute_calc (bool): whether to add the per minute stats to the data or not
    bayes_priors_df: a data frame containing the bayes priods from the basian_priors() function. Only required if doing per minute calcs
    
    returns:S
    data frame showing the sum for the last 3 entries for each column excluding the current date.
    """
    
    # pdb.set_trace()
    
    # Get columns for grouping and stats
    id_columns = list(set(id_cols + group_by))
    stat_columns = [col for col in df.columns if col not in id_columns]
    
    # Convert to polars
    # Sort DataFrame by group_by columns and date for ordered cumulative calculation
    pl_sorted = pl.from_pandas(df).sort(by=group_by + ['date'])
    
    # Set the join columns
    join_cols = [col for col in id_cols if 'date' not in col]
    
    # Check that the join cols give unique rows
    stop_if_not(len(pl_sorted) == len(pl_sorted.unique(subset = join_cols)), "Check that join cols identified in aggregate_last_X_fight_data inputs have a unique row for each occurance.")
    
    # pdb.set_trace()
    
    # Group by the group_by columns and calculate rolling sums for the last 3 entries
    rolling_data = pl_sorted.with_columns([
        pl.col(c).shift(1).rolling_sum(X, min_samples = 1).over(group_by)
        for c in stat_columns
    ])
    
    
        
    # Rename the adj columns to make it clear what they are
    rolling_data = rolling_data.rename({col: col[:-4] + "_adjsum" if col.endswith("_adj") else col for col in rolling_data.columns})
    
    # Calculate rolling means for the adj statistics
    rolling_mean_data = pl_sorted.with_columns([
        pl.col(c).shift(1).rolling_mean(X, min_samples = 1).over(group_by)
        for c in [col for col in stat_columns if '_adj' in col]
    ]).select(id_cols + [col for col in stat_columns if '_adj' in col])
    
    # Rename the adj columns to make it clear what they are
    rolling_mean_data = rolling_mean_data.rename({col: col[:-4] + "_adjmean" if col.endswith("_adj") else col for col in rolling_mean_data.columns})
    
    # Combine the dataset
    rolling_data = rolling_data.join(rolling_mean_data.drop('date'), on = join_cols, how = 'left')
            
    # If fights column is in, replace NAs with 0
    if 'fights' in stat_columns:
        rolling_data = rolling_data.with_columns([pl.col('fights').fill_null(0)])
    
    # Rename columns with prefix and concatenate with the id_columns
    rolling_data = rolling_data.rename({x: prefix + x for x in rolling_data.columns if x not in id_columns})
    df_rolling = pl_sorted[join_cols].join(rolling_data, on = join_cols, how = 'left').to_pandas()

    # Calculate per minute stats if needed
    if per_minute_calc:
        df_rolling = calculate_per_minute_stats(df_rolling)
        if bayes_priors_df is not None:
            df_rolling = calculate_per_minute_bayesian_stats(df_rolling, bayes_priors_df, bayes_alpha_plus_beta_rebase = bayes_alpha_plus_beta_rebase)
    
    return df_rolling
