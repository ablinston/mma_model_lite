def merge_df_both_fighters(left_df, df_list):
    """
    this function will take a list of data frames and merge them onto a starting df
    for both fighter1 and fighter2 with added prefixes

    arguments:
    left_df: the data frame that the other data frames will be merged onto
    df_list: a list of data frames to be merged for fighter1 and fighter2. These should contain the columns ['event', 'bout', 'fighter'] to be merged with

    returns:
    df
    """
    # Loop through each df
    for df in df_list:
        
        # Skip if empty
        if len(df) == 0:
            continue
        
        # Merge for fighter 1
        left_df = left_df.merge(
            df.add_prefix('fighter1_'),
            left_on = ['event', 'bout', 'fighter1'],
            right_on = ['fighter1_event', 'fighter1_bout', 'fighter1_fighter'],
            how = 'left'
            ).drop(columns = ['fighter1_event', 'fighter1_bout', 'fighter1_fighter'])

        # Now merge for fighter 2
        left_df = left_df.merge(
            df.add_prefix('fighter2_'),
            left_on = ['event', 'bout', 'fighter2'],
            right_on = ['fighter2_event', 'fighter2_bout', 'fighter2_fighter'],
            how = 'left'
            ).drop(columns = ['fighter2_event', 'fighter2_bout', 'fighter2_fighter'])
    
    return left_df
