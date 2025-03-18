# This function takes a data frame with stats columns, another dataframe with every fighter's historic stats and calculates adjusted
# statistics on the stats data. By default, this function adjusts with the opponents Bayes smoothed stats

def add_adjusted_bayes_stats(
        stats_df, # The cleaned stats df with a bit of processing
        hist_df, # The aggregated historic stats from the aggregate_last_X_fights() function
        columns_with = ['per_minute_bayes']): # A string to select a subset of columns to cal stats for with this string
    
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

    stop_if_not(type(columns_with) == type(['list']), "columns_with type check")

    # We want to merge the historic stats of opponents onto the data
    # FIrst get opponent column
    new_stats_df = add_opponent_column(stats_df.copy())
    
    # Add per minute stats on
    if 'per_minute' in columns_with:
        new_stats_df = calculate_per_minute_stats(new_stats_df, excluded_strings = ['percent', 'fight_length', 'fight_minute', 'opponent', 'bayes'])
    
    # Find the columns to adjust
    columns_to_adjust = []
    adjustment_cols = []
    for item in columns_with:
        columns_to_adjust += [col for col in new_stats_df.columns if item in col and 'against' not in col]
        # Now find corresponding historic columns for the columns to adjust
        adjustment_cols += ['hist_' + col.replace('_per', '_against_per') for col in columns_to_adjust]
        # Now add on the opposing stats for defense
        adjustment_cols += ['hist_' + col for col in columns_to_adjust]
        # Now add on the against stats
        columns_to_adjust += [col for col in new_stats_df.columns if item in col and 'against' in col]
        
    # Now merge on each opponents historic stats
    new_stats_df = new_stats_df.merge(hist_df, left_on = ['event', 'bout', 'opponent'], right_on = ['event', 'bout', 'fighter'], how = 'left')
    
    # Check all the columns are in the data
    missing_cols =  [col for col in adjustment_cols if col not in new_stats_df.columns]
    stop_if_not(len(missing_cols) == 0, "Columns in hist data check")
    
    # Calculate the adjusted stats
    new_stats = pd.DataFrame(
        new_stats_df[columns_to_adjust].to_numpy() - new_stats_df[adjustment_cols].to_numpy()
        , columns = [col + '_adj' for col in columns_to_adjust])
    
    # Final data
    final_stats_df = pd.concat([stats_df, new_stats], axis = 1)
    
    # Check length
    stop_if_not(len(final_stats_df) == len(stats_df), "Dataset length")
    
    return final_stats_df
