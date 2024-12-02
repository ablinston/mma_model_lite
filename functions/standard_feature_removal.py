# This function removes columns from training data that seem arbitrary and could lead to over-fitting

def standard_feature_removal(df):

    data = df.copy()
    
    # Remove cols that shouldn't really affect the fight.
    # Remove dates because they can overfit. Leave date minuses, which are days since
    cols_to_remove = [col for col in data.columns if ('date' in col) & (('diff' not in col) & ('minus' not in col) & (col != 'date'))]

    # Get rid of dob and reach. Seem arbitrary
    cols_to_remove = cols_to_remove + ['fighter1_dob', 'fighter2_dob', 'fighter1_reach', 'fighter2_reach', 'diff_reach']

    # # Remove cols that are just totals for a fighter, like total kicks etc. Leave percentages and ratios and per minute stats
    cols_to_remove = cols_to_remove + [col for col in data.columns if ('_round' in col) and not (('percent' in col) or ('ratio' in col) or ('per_minute' in col) or ('per_fight' in col))]
    cols_to_remove = cols_to_remove + [col for col in data.columns if ('allrounds' in col) and not (('percent' in col) or ('ratio' in col) or ('per_minute' in col) or ('per_fight' in col) or (('against' in col) and (('hist_round' in col) or ('hist_last_3_fights' in col))))]
    # Remove the columns that are single results
    cols_to_remove = cols_to_remove + [col for col in data.columns if ('result_' in col)]

    # Now we want to remove any column that is not a differential or a comparison between fighters
    # This bit is optional. Just testing.
    cols_to_remove = cols_to_remove + [col for col in data.columns if ('diff' not in col) and ('opp_type' not in col) and ('offense_minus' not in col) and ('fight_elevation' not in col) and ('weightclass' not in col) and ('matchup' not in col)]

    # Remove columns not wanted
    data = data[[col for col in data.columns if not col in cols_to_remove]]

    return data