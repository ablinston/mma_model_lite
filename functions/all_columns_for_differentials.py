# This function takes a list of column names and expands it to contain all the columns that will be needed to calculate differentials too

def all_columns_for_differentials(col_list):
    
    # pdb.set_trace()
    
    # Get the underlying for diff cols
    diff_cols = pd.Series([col for col in col_list if 'diff_' in col and 'diff_ratio_' not in col and 'offense' not in col])
    new_fighter1_cols = diff_cols.str.replace('diff', 'fighter1').str.replace('_against', '')
    new_fighter2_cols = diff_cols.str.replace('diff', 'fighter2').str.replace('_against', '')
    
    # Create against versions
    new_fighter1_cols_against = new_fighter1_cols.str.replace('accuracy_percent', 'accuracy_percent_against').str.replace('per_minute', 'against_per_minute')
    new_fighter2_cols_against = new_fighter2_cols.str.replace('accuracy_percent', 'accuracy_percent_against').str.replace('per_minute', 'against_per_minute')
    
    # Get the underlying diff ratio columns needed
    diff_ratio_cols = pd.Series([col for col in col_list if 'diff_ratio_' in col and 'offense' not in col])
    new_fighter1_ratio_cols = diff_ratio_cols.str.replace('diff_ratio', 'fighter1').str.replace('_against', '')
    new_fighter2_ratio_cols = diff_ratio_cols.str.replace('diff_ratio', 'fighter2').str.replace('_against', '')
    
    # Create against versions
    new_fighter1_ratio_cols_against = new_fighter1_ratio_cols.str.replace('accuracy_percent', 'accuracy_percent_against').str.replace('per_minute', 'against_per_minute')
    new_fighter2_ratio_cols_against = new_fighter2_ratio_cols.str.replace('accuracy_percent', 'accuracy_percent_against').str.replace('per_minute', 'against_per_minute')
    
    # Combine these with the original list
    new_col_series = pd.concat([
        pd.Series([col for col in col_list if 'diff' not in col]),
        new_fighter1_cols,
        new_fighter2_cols,
        new_fighter1_ratio_cols,
        new_fighter2_ratio_cols
        ]).drop_duplicates()
    
    # Now look for offence minus defence cols. There are two types. One with offense_minus_defense_ string, and one with fighter1_offense_minus_fighter
    # We'll need the against version and the non against version. In these lists, against is either after accuracy_percent or before _per_minute
        
    ### offense_minus_defense_
    accuracy_percent_cols = pd.Series([col for col in new_col_series.tolist() if 'offense_minus_defense_' in col])
    
    # First get the base name of the columns
    raw_off_cols = accuracy_percent_cols.str.replace('offense_minus_defense_', '')
    
    # Because there's a mix of fighter 1 and 2, get a list for both
    raw_fighter1_cols = raw_off_cols.str.replace('fighter2', 'fighter1')
    raw_fighter2_cols = raw_off_cols.str.replace('fighter1', 'fighter2')
    
    # And add against cols
    raw_fighter1_cols_against = raw_fighter1_cols.str.replace('accuracy_percent', 'accuracy_percent_against').str.replace('per_minute', 'against_per_minute')
    raw_fighter2_cols_against = raw_fighter2_cols.str.replace('accuracy_percent', 'accuracy_percent_against').str.replace('per_minute', 'against_per_minute')
    
    
    ### fighter1_offense_minus_fighter
    split_offense = pd.Series([col for col in new_col_series.tolist() if 'offense_minus_fighter' in col])
    
    # Get the underlying columns
    raw_split_offense = split_offense.str.replace('offense_minus_fighter1_defense_', '').str.replace('offense_minus_fighter2_defense_', '')
    
    # Get just the individual fighter cols
    fighter1_split_cols = raw_split_offense.str.replace('fighter2', 'fighter1')
    fighter2_split_cols = raw_split_offense.str.replace('fighter1', 'fighter2')
    
    # Now get the against cols
    fighter1_split_cols_against = fighter1_split_cols.str.replace('accuracy_percent', 'accuracy_percent_against').str.replace('per_minute', 'against_per_minute')
    fighter2_split_cols_against = fighter2_split_cols.str.replace('accuracy_percent', 'accuracy_percent_against').str.replace('per_minute', 'against_per_minute')

    #### All against cols
    # Need to create against versions of all columns for differentials
    all_non_diff = pd.Series([col for col in col_list if 'diff' not in col and 'offense' not in col])
    
    # Create non against versions
    all_fighter1_non_against = all_non_diff.str.replace('fighter2', 'fighter1').str.replace('_against', '')
    all_fighter2_non_against = all_non_diff.str.replace('fighter1', 'fighter2').str.replace('_against', '')
    
    # Now create against
    all_fighter1_against = all_fighter1_non_against.str.replace('accuracy_percent', 'accuracy_percent_against').str.replace('per_minute', 'against_per_minute')
    all_fighter2_against = all_fighter2_non_against.str.replace('accuracy_percent', 'accuracy_percent_against').str.replace('per_minute', 'against_per_minute')

    # Now combine all the additional columns and new ones
    full_col_list = pd.concat([
        pd.Series(col_list),
        new_fighter1_cols,
        new_fighter2_cols,
        new_fighter1_cols_against,
        new_fighter2_cols_against,
        new_fighter1_ratio_cols,
        new_fighter2_ratio_cols,
        new_fighter1_ratio_cols_against,
        new_fighter2_ratio_cols_against,
        raw_fighter1_cols,
        raw_fighter2_cols,
        raw_fighter1_cols_against,
        raw_fighter2_cols_against,
        fighter1_split_cols,
        fighter2_split_cols,
        fighter1_split_cols_against,
        fighter2_split_cols_against,
        all_fighter1_non_against,
        all_fighter2_non_against,
        all_fighter1_against,
        all_fighter2_against
        ], ignore_index = True).drop_duplicates().tolist()
    
    return full_col_list