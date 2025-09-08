
def bayesian_priors(fight_stats_df, id_cols, group_by, fight_span, ignore_last_n_fights = 15):
    
    """
    Takes a historical dataset and calculates the bayesian priors for each stat that will be needed to calculate
    Bayesian smoothed
    
    Args:
        hist_stats_df (df): a dataframe with historic stats
        id_cols (list): the cols used to identify a fight
        group_by (list): the columns we want to group by for the Bayesian priors
        fight_span (integer): the number of historic fights to use for the averages. 100 is about 18 months in more recent years
        
    Returns:
        A dictionary of metrics
    
    """
    # id_cols = ['event', 'bout', 'fighter']
    # fight_stats_df = fight_stats.copy().merge(fight_results[['event', 'bout', 'weightclass']], on = ['event', 'bout'], how = 'left')
    # group_by = ['weightclass']
    # fight_span = 100
    
    stats_df = fight_stats_df.copy()
    
    # pdb.set_trace()

    # Sort the data by weightclass and date, and add in sort by id_cols to ensure consistency in the ordering between different runs
    stats_df.sort_values(group_by + ['date'] + id_cols, inplace = True)

    # Get a list of all the numeric columns
    numeric_columns = stats_df.select_dtypes(include = ['number', 'float']).columns.tolist()
    
    # Get the aggregate figures for each fight. Excluded the last 15 just to ensure no data from the same date is used. 2 * fightspan because we have a row per fighter
    if len(group_by) > 0:
        total_stats = stats_df.groupby(group_by, group_keys = False).apply(lambda x: x[numeric_columns].rolling(window = 2 * fight_span).sum().shift(ignore_last_n_fights * 2 + 1), include_groups = False) # Shift 1 to not include current observation, then 2 * n fighters
    else:
        total_stats = stats_df[numeric_columns].rolling(window = 2 * fight_span).sum().shift(ignore_last_n_fights * 2 + 1) # Shift 1 to not include current observation, then 2 * n fighters
   
    # Rename the columns
    total_stats.columns = ['bayes_total_' + col for col in total_stats.columns]
    
    # Merge onto the original data
    final_data = stats_df[id_cols + group_by].merge(total_stats, left_index=True, right_index=True, how = 'left')
 
    # Check we have the correct number of rows
    stop_if_not(len(final_data) == len(fight_stats_df))
    
    # Drop the group column and also only keep one record per bout. We want both fighters to use the same priors when calculating the bayes numbers
    final_data = final_data.drop_duplicates(subset = [col for col in id_cols if 'fighter' not in col]).drop(group_by + ['fighter'], axis = 1)
        
    # Return the data frame    
    return final_data
