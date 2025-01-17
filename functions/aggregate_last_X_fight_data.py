def aggregate_last_X_fight_data(df, ignore_cols, group_by, prefix, X):
    """
    Convert fight data into cumulative fight history for the last 3 entries (excluding the current entry).

    arguments:
    df: a data frame with one row for each fighter
    ignore_cols: the columns not to be aggregated
    group_by: a list of columns to be aggregated by
    prefix: a str to be added to the start of stat columns
    X: the number of historic fights to include

    returns:S
    data frame showing the sum for the last 3 entries for each column excluding the current date.
    """
    # Get columns for grouping and stats
    id_columns = list(set(ignore_cols + group_by))
    stat_columns = [col for col in df.columns if col not in id_columns]
    
    # Sort DataFrame by group_by columns and date for ordered cumulative calculation
    df_sorted = df.sort_values(by=group_by + ['date']).reset_index(drop=True)

    # Group by the group_by columns and calculate rolling sums for the last 3 entries
    rolling_data = (
        df_sorted.groupby(group_by)[stat_columns]
        .apply(lambda x: x.shift(1).rolling(window=X, min_periods=1).sum())
        .reset_index(level=group_by, drop=True)
    )

    # Rename columns with prefix and concatenate with the id_columns
    rolling_data = rolling_data.rename(columns=lambda x: prefix + x)
    df_rolling = pd.concat([df_sorted[id_columns].reset_index(drop=True), rolling_data.reset_index(drop=True)], axis=1)

    return df_rolling
