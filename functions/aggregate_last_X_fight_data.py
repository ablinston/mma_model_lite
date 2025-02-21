def aggregate_last_X_fight_data(df, ignore_cols, group_by, prefix, X, per_minute_calc = False, bayes_priors_df = None):
    """
    Convert fight data into cumulative fight history for the last 3 entries (excluding the current entry).

    arguments:
    df: a data frame with one row for each fighter
    ignore_cols: the columns not to be aggregated
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
    id_columns = list(set(ignore_cols + group_by))
    stat_columns = [col for col in df.columns if col not in id_columns]
    
    # Sort DataFrame by group_by columns and date for ordered cumulative calculation
    df_sorted = df.sort_values(by=group_by + ['date']).reset_index(drop=True)

    # Group by the group_by columns and calculate rolling sums for the last 3 entries
    rolling_data = (
        df_sorted.groupby(group_by)
        .apply(lambda x: x[stat_columns].shift(1).rolling(window=X, min_periods=1).sum(), include_groups = False)
        .reset_index(level=group_by, drop=True)
    )
    
    # # Convert your pandas DataFrame to a Dask DataFrame
    # ddf = dd.from_pandas(df_sorted, npartitions=4)  # Adjust the number of partitions as needed

    # # Ensure the data is sorted by the group key to allow for proper rolling
    # ddf = ddf.sort_values(by=group_by + ['date'])

    # # Perform the shift and rolling sum
    # def apply_rolling(group):
    #     group[stat_columns] = group[stat_columns].shift(1)
    #     return group[stat_columns].rolling(window=X, min_periods=1).sum()

    # # Apply the rolling operation with a custom meta to avoid the warning
    # rolling_data = (
    #     ddf.groupby(group_by, group_keys=False)
    #     .apply(apply_rolling, meta={col: 'f8' for col in stat_columns})  # Specify the meta here
    #     .compute()  # Convert back to pandas
    # )
        
    # If fights column is in, replace NAs with 0
    if 'fights' in stat_columns:
        rolling_data['fights'] = rolling_data['fights'].fillna(0)
    
    # Rename columns with prefix and concatenate with the id_columns
    rolling_data = rolling_data.rename(columns=lambda x: prefix + x)
    df_rolling = pd.concat([df_sorted[id_columns].reset_index(drop=True), rolling_data.reset_index(drop=True)], axis=1)

    # Calculate per minute stats if needed
    if per_minute_calc:
        df_rolling = calculate_per_minute_stats(df_rolling)
        if bayes_priors_df is not None:
            df_rolling = calculate_per_minute_bayesian_stats(df_rolling, bayes_priors_df)
    
    # Replace any column names with adj at the end to make it clear its the sum
    df_rolling.rename(columns={col: col[:-4] + "_adjsum" if col.endswith("_adj") else col for col in df.columns}, inplace=True)
    
    return df_rolling
