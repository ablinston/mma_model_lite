# This function prepares a stats dataset for the elo calcs. It takes in pandas data frames,
# the cleaned fight results and fights stats data,
# then outputs two datatables because that will be used for the ELO calc.

# The first output is a data table with each fighter and the ELO for each fight ready to be populated
# The second output is a data table with each fight and the results/stats

# remove_nas is a setting on whether to remove NA stats from the dataset. Some fights have no stats. If they are not remove, NAs will be filled by 0 so a basic version of ELO will apply.

def prep_stats_for_elo_calc(fight_results_df, fight_stats_df, config, start_date = '2000-01-01', remove_nas = False):
    
    # pdb.set_trace()
    
    fight_results = fight_results_df.copy()
    fight_stats = fight_stats_df.copy()
    
    # Recode the target. Don't need to distinguish between D and NC
    fight_results['outcome'] = np.where((fight_results['outcome'] == "D/D") | (fight_results['outcome'] == "NC/NC"), "OTH", fight_results['outcome'])
    
    # Add the fighter columns needed to the data
    fight_results[['fighter1', 'fighter2']] = fight_results['bout'].str.split('vs\\.', expand = True)

    # Merge the two datasets together to get stats for the fight
    merged_data = merge_df_both_fighters(
        left_df =  fight_results[['event', 'bout', 'fighter1', 'fighter2', 'outcome', 'method', 'weightclass', 'date', 'time format']],
        df_list = [
            fight_stats
        ]
    ).reset_index().drop(columns = ['index'])

    # Remove any results that aren't wins or losses by DQ. Remove anything before start_date because of multiple fights per date.
    # Keep np.nan because these are future events we want ELOs for
    merged_data = merged_data[(merged_data['method'].isin(['Decision-Unanimous', 'KO/TKO', 'Submission', 'Decision-Split', np.nan])) & (merged_data['date'] >= start_date)]

    # Cleanup
    del([fight_results, fight_stats])
    
    # Get rid of any rows where we don't have stats
    if remove_nas:
        # Count NaN values in 'round1_fight_minutes' per event
        na_counts_per_event = merged_data['fighter1_round1_fight_minutes'].isna().groupby(merged_data['event']).sum().reset_index()
        # Rename columns for clarity
        na_counts_per_event.columns = ['event', 'na_count_round1_fight_minutes']
        na_counts_per_event = na_counts_per_event[na_counts_per_event['na_count_round1_fight_minutes'] > 0]
        if len(na_counts_per_event) > 0:
            print("Removing bouts from the following events where no fight stats available")
            print(na_counts_per_event)
            merged_data = merged_data[~merged_data['fighter1_round1_fight_minutes'].isna()]

    # Remove any duplicates of fighters on dates
    # Sort so we keep records with results first
    merged_data = merged_data.sort_values(['outcome'])
    merged_data = remove_duplicate_fighter_pairs(merged_data)

    # Now create a data frame that contains only the fighters and the dates they fought. This will be used to track the existing ELO of a fighter on that date
    start_elo = pd.concat([
        merged_data[['fighter1', 'date']].rename(columns = {'fighter1': 'fighter'}),
        merged_data[['fighter2', 'date']].rename(columns = {'fighter2': 'fighter'})
        ]).sort_values(['fighter', 'date'])

    # # Remove any remaining duplicates # Shouldn't be needed
    # start_elo = start_elo.drop_duplicates(['date', 'fighter'])

    # Add the fight number for each fighter
    start_elo['fight_number'] = start_elo.groupby('fighter').cumcount() + 1

    # Compute the starting ELO for each fighter
    start_elo['start_elo'] = np.nan
    start_elo.loc[start_elo['fight_number'] == 1,'start_elo'] = config['start_elo']

    # Now merge these onto the merged fight data to being calculating the ELO at the end of the first fights
    merged_data = merged_data.merge(
        start_elo[['fighter', 'date', 'start_elo']].rename(columns = {'start_elo': 'fighter1_start_elo'}),
        left_on = ['fighter1', 'date'], 
        right_on = ['fighter', 'date'],
        how = 'left'
    ).drop(columns='fighter').merge(
        start_elo[['fighter', 'date', 'start_elo']].rename(columns = {'start_elo': 'fighter2_start_elo'}),
        left_on = ['fighter2', 'date'], 
        right_on = ['fighter', 'date'],
        how = 'left'
    ).drop(columns='fighter')

    ##############################################################################
    # Create the columns that will be used in the ELO calculation

    merged_data['control_proportion_diff'] = (merged_data['fighter1_allrounds_ctrl_minutes'] / merged_data['fighter1_allrounds_fight_minutes']) - (merged_data['fighter2_allrounds_ctrl_minutes'] / merged_data['fighter2_allrounds_fight_minutes'])

    merged_data['sig_strikes_landed_per_minute_diff'] = np.where(
        merged_data['fighter1_allrounds_sig.str._landed'] > merged_data['fighter2_allrounds_sig.str._landed'],
        1,
        np.where(
            merged_data['fighter1_allrounds_sig.str._landed'] < merged_data['fighter2_allrounds_sig.str._landed'],
            -1,
            0)
    )

    # merged_data['strikes_landed_per_minute_head_diff'] = (merged_data['fighter1_allrounds_head_landed'] / merged_data['fighter1_allrounds_fight_minutes']) - (merged_data['fighter2_allrounds_head_landed'] / merged_data['fighter2_allrounds_fight_minutes'])

    # merged_data['strikes_landed_per_minute_body_diff'] = (merged_data['fighter1_allrounds_body_landed'] / merged_data['fighter1_allrounds_fight_minutes']) - (merged_data['fighter2_allrounds_body_landed'] / merged_data['fighter2_allrounds_fight_minutes'])

    # merged_data['strikes_landed_per_minute_leg_diff'] = (merged_data['fighter1_allrounds_leg_landed'] / merged_data['fighter1_allrounds_fight_minutes']) - (merged_data['fighter2_allrounds_leg_landed'] / merged_data['fighter2_allrounds_fight_minutes'])

    # merged_data['strikes_landed_per_minute_ground_diff'] = (merged_data['fighter1_allrounds_ground_landed'] / merged_data['fighter1_allrounds_fight_minutes']) - (merged_data['fighter2_allrounds_ground_landed'] / merged_data['fighter2_allrounds_fight_minutes'])

    # merged_data['strikes_landed_per_minute_clinch_diff'] = (merged_data['fighter1_allrounds_clinch_landed'] / merged_data['fighter1_allrounds_fight_minutes']) - (merged_data['fighter2_allrounds_clinch_landed'] / merged_data['fighter2_allrounds_fight_minutes'])

    merged_data['kds_diff'] = np.where(
        merged_data['fighter1_allrounds_kd'] > merged_data['fighter2_allrounds_kd'],
        1,
        np.where(
            merged_data['fighter1_allrounds_kd'] < merged_data['fighter2_allrounds_kd'],
            -1,
            0)
    )

    merged_data['submissions_diff'] = np.where(merged_data['method'] == 'Submission',
                                               np.where(merged_data['outcome'] == 'W/L', 1, -1),
                                               0)

    merged_data['knockouts_diff'] = np.where(merged_data['method'] == 'KO/TKO',
                                               np.where(merged_data['outcome'] == 'W/L', 1, -1),
                                               0)

    merged_data['unanimous_decs_diff'] = np.where(merged_data['method'] == 'Decision-Unanimous',
                                               np.where(merged_data['outcome'] == 'W/L', 1, -1),
                                               0)

    merged_data['split_decs_diff'] = np.where(merged_data['method'] == 'Decision-Split',
                                               np.where(merged_data['outcome'] == 'W/L', 1, -1),
                                               0)
    
    # Add the fight number for each fighter
    merged_data.sort_values('date', inplace = True)
    merged_data['fighter1_fight_number'] = merged_data.groupby('fighter1').cumcount() + 1
    merged_data['fighter2_fight_number'] = merged_data.groupby('fighter2').cumcount() + 1
    
    # Calculate the inverse of the fight length proportion (doing all as 25 minutes). Positive for a win, negative for loss
    merged_data['fight_end_in_round'] = np.ceil(merged_data['fighter1_allrounds_fight_minutes'] / 5)
    merged_data['inverse_fight_length_proportion'] = np.where(
        merged_data['outcome'] == 'W/L', 
        1 / merged_data['fight_end_in_round'],
        - 1 / merged_data['fight_end_in_round'])
    
    # pdb.set_trace()
    # Fill any NAs with 0 so functions run
    merged_data = merged_data.fillna({
        col: 0 if merged_data[col].dtype in ['float64', 'int64'] else ''
        for col in [colu  for colu in merged_data.columns if colu not in ['fighter1_start_elo', 'fighter2_start_elo']]
    })
    
    # Convert to data table for speed
    merged_data = dt.Frame(merged_data)
    start_elo = dt.Frame(start_elo)
    start_elo.key = ['date', 'fighter']
    
    return start_elo, merged_data
