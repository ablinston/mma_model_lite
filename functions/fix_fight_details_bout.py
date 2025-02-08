# The 'bout' column on the fight details page of UFCstats sometimes has the fighters in the 
# reverse order. This function takes the fight stats data frame as a reference (where the order of the fighters in the bout is correct)
# and uses that to correct the bout on the fight details data frame

# It returns a fixed fight_details data frame

def fix_fight_details_bout(fight_details, fight_stats_df):

    # pdb.set_trace()
    
    # Create a copy of stats so the original isn't modified and with stripped columns
    fight_stats = fight_stats_df[['EVENT', 'BOUT']].drop_duplicates()
    
    # This function requires spaces to be removed, or they cause issues
    for df in [fight_details, fight_stats]:
        for col in ['EVENT', 'BOUT']:
            try:
                df[[col]] = clean_text_columns(df[[col]])
            except:
                continue

    
    # First, get individual columns for the fighters on the details data
    fight_details[['fighter1', 'fighter2']] = fight_details['BOUT'].str.split('vs\\.', expand = True)
    
    # Now reverse the bout
    fight_details['BOUT_ALT'] = fight_details['fighter2'] + 'vs.' + fight_details['fighter1']

    ## Now find which fights are in the stats data frame under the alternate ordering on the same event
    
    # Create a data frame to merge with a flag
    fight_alt_flag_df = fight_stats.rename(columns = {'BOUT': 'BOUT_STATS'})
    fight_alt_flag_df.loc[:, 'ALT_FLAG'] = 1
    
    # Create a new fight data frame merging whether the alt bout was found
    new_fight_details = fight_details.merge(fight_alt_flag_df, how = 'left', left_on = ['EVENT', 'BOUT_ALT'], right_on = ['EVENT', 'BOUT_STATS'])
    
    # Check the new data has the same length of rows
    if len(fight_details) != len(new_fight_details):
        print("Error when fixing bouts. Inconsistent row number before and after flag merge.")
        return None
    
    # Replace bouts where required
    new_fight_details['BOUT'] = np.where(new_fight_details['ALT_FLAG'] == 1, new_fight_details['BOUT_ALT'], new_fight_details['BOUT'])
    
    # Drop all columns that were added
    new_fight_details.drop(['BOUT_ALT', 'fighter1', 'fighter2', 'ALT_FLAG', 'BOUT_STATS'], axis = 1, inplace = True)

    return new_fight_details
