# This function uses the fight stats history to define the type of fighter someone is
# It uses the historic fight stats for each fighter, and the results data at fighter level (not bout)

# It returns a pd Series of fighter types

def define_fighter_type(fight_stats_hist, fighter_results, config):
    
    # pdb.set_trace()
    
    merged_data = fight_stats_hist.merge(fighter_results, on = ['event', 'bout', 'fighter'], how = 'left')
    
    # Use the thresholds in the config. Loop through the fighter types
    for fighter_type in list(config['fighter_type_thresholds'].keys()):
        # Check if its empty
        if config['fighter_type_thresholds'][fighter_type] == None:
            merged_data['fighter_type'] = fighter_type
        # If not empty, look for the conditions
        else:
            conditions = {key:value for key, value in config['fighter_type_thresholds'][fighter_type].items()}

            # Find which rows meet the conditions
            meets_conditions = (merged_data[list(conditions)] > pd.Series(conditions)).all(axis = 1)
            
            # Then set those fighters to the fighter type
            merged_data.loc[meets_conditions, 'fighter_type'] = fighter_type
    
    # Check the results is the correct length
    if len(merged_data['fighter_type']) == len(fight_stats_hist):
        return merged_data['fighter_type']
    else:
        print("Error: fighter type couldn't be derived")
        return None
