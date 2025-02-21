# This function takes fight stats and calculates per minute versions on the dataset then returns the full data with added per minute columns

# Can set some excluded strings to not include columns containing those phrases

def calculate_per_minute_bayesian_stats(fight_stats_hist, bayes_priors_df, excluded_strings = ['percent', 'fight_length', 'fight_minute', 'ratio', 'per_minute', 'baye']):

    # Merge the bayesian prios onto the data
    stats_df = fight_stats_hist.merge(bayes_priors_df, on = ['event', 'bout'], how = 'left')
    
    stop_if_not(len(stats_df) == len(fight_stats_hist))
    
    # pdb.set_trace()
    
    new_stats_df = fight_stats_hist.copy()
    
    # Go through each round's stats individually
    rounds = ['round1', 'round2', 'round3', 'round4', 'round5', 'allrounds']
    
    for round in rounds:
        # Get all round columns
        round_columns = [col for col in stats_df.columns if (round in col)]
        
        if len(round_columns) == 0:
            continue
        
        # Get the prefix of the columns
        prefix = str(round_columns[0]).split(str(round))[0]
        
        # Now exclude any columns we don't need per minute stats for
        for string in excluded_strings + ['bayes']:
            round_columns = [col for col in round_columns if string not in col]
    
        # Check all the Bayes prior columns are in the data
        if prefix == '':
            stop_if_not(all(item.replace('_against', '') in stats_df.columns for item in ['bayes_total_' + col for col in round_columns]), "Bayes columns check in data")
        else:
            stop_if_not(all(item.replace('_against', '') in stats_df.columns for item in [col.replace(prefix, 'bayes_total_') for col in round_columns]), "Bayes columns check in data")
        
        # Work out the associated bayes columns for numerator and denominator
        if prefix == '':
            bayes_numerators = ['bayes_total_' + item.replace('_against', '') for item in round_columns]
        else:
            bayes_numerators = [item.replace('_against', '').replace(prefix, 'bayes_total_') for item in round_columns]
        
        # Calculate the per minute stats and merged onto data
        new_stats_df = pd.concat([
            new_stats_df,
            pd.DataFrame((stats_df[round_columns] + stats_df[bayes_numerators].to_numpy()).div(stats_df[prefix + round + '_fight_minutes'].to_numpy() + stats_df['bayes_total_' + round + '_fight_minutes'].to_numpy(), axis = 0).to_numpy(),
                         columns = [(col + '_per_minute_bayes') for col in round_columns]),          
        ], axis = 1)
    
    # Do a check
    if len(new_stats_df) == len(fight_stats_hist):
        return new_stats_df
    else:
        print("Error: per minute stats couldn't be calculated")
        return None