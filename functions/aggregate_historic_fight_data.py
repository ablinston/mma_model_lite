def aggregate_historic_fight_data(df, ignore_cols, group_by, prefix):
    """
    Convert fight data into cumulative fight history at points in time

    arguments:
    df: a data frame with one row for each fighter
    ignore_cols: the columns not to be aggregated
    group_by: a list of columns to be aggregated by
    prefix: a str to be added to the start of stat columns

    returns:
    data frame showing the historic sums for each column excluding the current date
    """
    # pdb.set_trace()

    # Get the columns of stat
    id_columns = list(set(ignore_cols + group_by))
    stat_columns = [col for col in df.columns if col not in id_columns]

    # # Create a new data frame with cumulative sums
    # df_hist = df[id_columns].copy()
    # for column in stat_columns:
    #     # Avoid repeated groupby calls by calculating cumulative sum on a grouped object
    #     cumsum_series = df.groupby(group_by)[column].cumsum()
    #     df_hist[prefix + column] = cumsum_series - df[column]
    
    # Initialize dictionary to store new columns
    cumulative_data = {col: df[id_columns].copy() for col in [f"{prefix}{col}" for col in stat_columns]}
    
    # Calculate cumulative sums for each stat column and store in the dictionary
    for column in stat_columns:
        cumsum_series = df.groupby(group_by)[column].cumsum()
        cumulative_data[prefix + column] = cumsum_series - df[column]
    
    # Concatenate all cumulative columns in a single operation to reduce fragmentation
    df_hist = pd.concat([df[id_columns]] + [pd.Series(data=cumulative_data[prefix + column], name=prefix + column) for column in stat_columns], axis=1)

    return df_hist.copy() # Do the copy to avoid fragmented data frame errors
