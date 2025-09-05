# This function takes fight stats and calculates per minute versions on the dataset then returns the full data with added per minute columns

# Can set some excluded strings to not include columns containing those phrases

def calculate_per_minute_stats(fight_stats_hist, excluded_strings = ['percent', 'fight_length', 'fight_minute', 'ratio', 'baye']):

    stats_df = fight_stats_hist.copy().reset_index(drop = True)
    
    # pdb.set_trace()
    
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
        for string in excluded_strings:
            round_columns = [col for col in round_columns if string not in col]
    
        # Calculate the per minute stats and merged onto data
        stats_df = pd.concat([
            stats_df,
            pd.DataFrame(stats_df[round_columns].div(stats_df[prefix + round +'_fight_minutes'].to_numpy(), axis = 0).to_numpy(),
                         columns = [(col + '_per_minute') for col in round_columns]),          
        ], axis = 1)
    
    # Do a check
    if len(stats_df) == len(fight_stats_hist):
        return stats_df
    else:
        print("Error: per minute stats couldn't be calculated")
        return None