# This is the master function for processing all the clean data, calculating variables and merging together

def merge_and_process_all_clean_data(
        config,
        elo_config,
        data_processing_config,
        odds_scrape_config,
        model_hyper_params,
        skip_save_data = False # whether to skip the save data and just return merged data
        ):
    
    # pdb.set_trace()

    # Read in the data
    events = fp.ParquetFile('./data/' + config['event_details_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
    upcoming_events = fp.ParquetFile('./data/' + config['upcoming_event_details_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
    fighters = fp.ParquetFile('./data/' + config['fighter_tott_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
    fight_results = fp.ParquetFile('./data/' + config['fight_results_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
    upcoming_fight_details = fp.ParquetFile('./data/' + config['upcoming_fight_details_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
    fight_stats = fp.ParquetFile('./data/' + config['fight_stats_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
    fight_odds = fp.ParquetFile('./data/' + config['fighter_odds_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
    event_odds = fp.ParquetFile('./data/' + config['event_odds_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
    betfair_odds = fp.ParquetFile('./data/' + config['betfair_odds_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
    betfair_upcoming_odds = fp.ParquetFile('./data/' + config['betfair_upcoming_odds_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()

    # Read in the average odds model
    average_odds_model = joblib.load('./results/average_odds_model.joblib')

    ###########################################################################################
    # UPCOMING

    # Add upcoming fights to fight stats
    fight_stats = add_upcoming_to_fight_stats(fight_stats, upcoming_fight_details, events, upcoming_events)

    # Add upcoming events
    events = pd.concat([upcoming_events, events], axis = 0)

    # Add the date to the upcoming fights
    upcoming_fight_details = upcoming_fight_details.merge(upcoming_events[['event', 'date']], on = 'event', how = 'left')

    # Add upcoming fights to the results with empty results
    fight_results = pd.concat([upcoming_fight_details.drop('url', axis = 1), fight_results])



    ###########################################################################################
    # CALCULATE ELOs

    # Get the starting data
    start_elo, elo_fight_data = prep_stats_for_elo_calc(fight_results, fight_stats, config, start_date = '2000-01-01', remove_nas = False)

    # Dderive the ELO for the params in the config
    elo_data, elo_fight_data_welo = derive_start_elos(start_elo, elo_fight_data, elo_config)

    # Create two versions to be added onto the results data
    fighter1_elo = elo_data.copy().to_pandas().drop(['fight_number'], axis = 1).rename(columns = {'fighter': 'fighter1', 'start_elo': 'fighter1_start_elo'}).sort_values(['date']).reset_index(drop=True)
    fighter2_elo = elo_data.copy().to_pandas().drop(['fight_number'], axis = 1).rename(columns = {'fighter': 'fighter2', 'start_elo': 'fighter2_start_elo'}).sort_values(['date']).reset_index(drop=True)

    # Merge this onto the fight results data, because this is a merge to the nearest date before or on the fight date. Can't merged elo_fight_data because draws etc were filtered out.

    fight_results = fight_results.sort_values(['date']).reset_index(drop=True)

    fight_results = pd.merge_asof(
        fight_results,
        fighter1_elo,
        on = 'date',
        by = 'fighter1',
        direction = 'backward'
    )
    fight_results = pd.merge_asof(
        fight_results,
        fighter2_elo,
        on = 'date',
        by = 'fighter2',
        direction = 'backward'
    )

    # Recode missing to the default starting. There will be some missing if their first fight was a DQ, D/D or NC.

    # Check there are no missing ELOs after the start date
    missing_elo_check = fight_results[fight_results['date'] > '2000-01-01'][pd.isna(fight_results['fighter1_start_elo'])]

    # Output to file before treating
    missing_elo_check.to_csv('./data/missing_elo_check.csv')

    fight_results['fighter1_start_elo'] = fight_results['fighter1_start_elo'].fillna(config['start_elo'])
    fight_results['fighter2_start_elo'] = fight_results['fighter2_start_elo'].fillna(config['start_elo'])

    del([start_elo, elo_fight_data, elo_data, elo_fight_data_welo, fighter1_elo, fighter2_elo])

    ###########################################################################################
    # FIGHT STATS PROCESSING

    # Calculate the historical stats of each fighter at the fight date
    fight_stats = fight_stats.sort_values(by=['fighter', 'date'], axis = 0)

    # Copy the fight stats but for against a fighter, then merge onto stats
    fight_stats_against = fight_stats.copy()
    fight_stats_against.rename(columns = lambda x: x + "_against" if "attempt" in x else x, inplace = True)
    fight_stats_against.rename(columns = lambda x: x + "_against" if "landed" in x else x, inplace = True)
    fight_stats_against.rename(columns = lambda x: x + "_against" if "ctrl_minutes" in x else x, inplace = True)
    fight_stats_against.rename(columns = lambda x: x + "_against" if "kd" in x else x, inplace = True)
    fight_stats_against.rename(columns = lambda x: x + "_against" if "sub.att" in x else x, inplace = True)

    # Split the bout into fighters
    # Set the opponent to the fighter for merging
    fight_stats_against[['fighter1', 'fighter2']] = fight_stats_against['bout'].str.split('vs\\.', expand = True)
    fight_stats_against['fighter'] = np.where(fight_stats_against['fighter'] == fight_stats_against['fighter1'], fight_stats_against['fighter2'], fight_stats_against['fighter1'])
    # Get only the columns we want
    against_columns = [col for col in fight_stats_against.columns if "against" in col]
    fight_stats_against = fight_stats_against[['event', 'bout', 'fighter'] + against_columns]

    # Merge onto the fight stats
    rows_before = len(fight_stats)
    fight_stats = fight_stats.merge(fight_stats_against, on = ['event', 'bout', 'fighter'], how = 'left')

    # Check that all the rows are correct
    if rows_before == len(fight_stats):
        print('Row check PASSED')
    else:
        print('Row check FAILED')
        raise SystemExit
    del fight_stats_against

    ###########################################################################################
    # Get the columns of stats

    if data_processing_config['add_bayes_stats_columns']:
        
        # FIrst calculate the Bayesian priors that will be used for Bayesian smoothing
        fight_stats_bayes_priors = bayesian_priors(
            fight_stats.sort_values('date').reset_index(drop = True).merge(fight_results[['event', 'bout', 'weightclass']], on = ['event', 'bout'], how = 'left'), # Join on the weightclass for grouping
            id_cols = ['event', 'bout', 'fighter'], 
            group_by = data_processing_config['bayesian_groupby'], 
            fight_span = data_processing_config['bayes_fightspan'],
            ignore_last_n_fights = 30 if data_processing_config['bayesian_groupby'] == [] else 10
        )
        
    else:
        fight_stats_bayes_priors = None

    # Now aggrgate
    fight_stats_hist = aggregate_last_X_fight_data(
        fight_stats,
        ignore_cols = ['date', 'event', 'bout', 'fighter'],
        group_by = ['fighter'],
        prefix = "hist_",
        X = 100,
        per_minute_calc = True, 
        bayes_priors_df = fight_stats_bayes_priors,
        bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
    ).drop(columns = ['date'])

    # Check that all first fights are 0 stats
    stat_columns = [col for col in fight_stats.columns if col not in ['date', 'event', 'bout', 'fighter']]
    fight_stats_hist_check = fight_stats_hist[fight_stats_hist['hist_fights'] == 0]
    if fight_stats_hist_check[['hist_' + s for s in stat_columns]].max().max() == 0:
        print('Cumulative aggregation check PASSED')
    else:
        print('Cumulative aggregation check FAILED')
        raise SystemExit
    del fight_stats_hist_check



    ###########################################################################################
    # ADJUSTED STATISTICS

    if data_processing_config['add_bayes_stats_columns'] and data_processing_config['add_adj_bayes_columns']:
        fight_stats_bayes = calculate_per_minute_bayesian_stats(fight_stats, fight_stats_bayes_priors, excluded_strings = ['percent', 'fight_length', 'fight_minute', 'ratio', 'per_minute'], bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute'])
        
        # Adding on adjusted stats to the fight stats
        fight_stats_plus_adj = add_adjusted_bayes_stats(fight_stats_bayes, fight_stats_hist, columns_with = ['per_minute_bayes'])
        
        # Replace the fight stats with one with the adj but remove all the extra bayes info (will be added by aggregation later)
        fight_stats = fight_stats_plus_adj[[col for col in fight_stats_plus_adj if 'bayes' not in col or ('bayes' in col and 'adj' in col)]]
        
    # Reaggregate the data
    fight_stats_hist = aggregate_last_X_fight_data(
        fight_stats,
        ignore_cols = ['date', 'event', 'bout', 'fighter'],
        group_by = ['fighter'],
        prefix = "hist_",
        X = 100,
        per_minute_calc = True, 
        bayes_priors_df = fight_stats_bayes_priors,
        bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
    ).drop(columns = ['date'])


    ###########################################################################################
    # Calculate average fight length, win length and loss length
    fight_stats_hist['hist_fight_length_minutes_avg'] = np.where(fight_stats_hist['hist_fights'] == 0, np.nan, fight_stats_hist['hist_allrounds_fight_minutes'] / fight_stats_hist['hist_fights'])

    stop_if_not(len(fight_stats) == len(fight_stats_hist), "Agg stats check")
        
        
    ###########################################################################################
    # Get the stats of the last 3 fights only
    fight_stats_hist_last_3 = aggregate_last_X_fight_data(
        fight_stats,
        ignore_cols = ['date', 'event', 'bout', 'fighter'],
        group_by = ['fighter'],
        prefix = "hist_last_3_fights_",
        X = 3,
        per_minute_calc = True, 
        bayes_priors_df = fight_stats_bayes_priors,
        bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
    ).drop(columns = ['date'])

    # Check that all first fights are NaN
    stat_columns = [col for col in fight_stats_hist_last_3.columns if col not in ['date', 'event', 'bout', 'fighter']]
    fight_stats_hist_last_3_check = fight_stats_hist_last_3[fight_stats_hist_last_3['hist_last_3_fights_fights'] == 0]

    if np.isnan(fight_stats_hist_last_3_check[[s for s in stat_columns if s != 'hist_last_3_fights_fights']].max(skipna = True).max(skipna = True)):
        print('Cumulative aggregation check PASSED')
    else:
        print('Cumulative aggregation check FAILED')
        raise SystemExit
    del fight_stats_hist_last_3_check


    ###########################################################################################
    # Get the fight stats of each fighter against only specific types of fighters
    stances = fighters['stance'].str.replace(' ', '').unique().tolist()
    fight_stats[['fighter1', 'fighter2']] = fight_stats['bout'].str.split('vs\\.', expand = True)
    fight_stats['opponent'] = np.where(fight_stats['fighter1'] == fight_stats['fighter'], fight_stats['fighter2'], fight_stats['fighter1'])
    fight_stats.drop(columns = ['fighter1', 'fighter2'], inplace = True)

    # Check no fighters are equal to opponents
    if all(fight_stats['opponent'] != fight_stats['fighter']):
        print('Opponent check PASSED')
    else:
        print('Opponent check FAILED')
        raise SystemExit

    # Merge on opponent stance
    fight_stats['opponent_stance'] = fight_stats[['opponent']].merge(
        fighters[['fighter', 'stance']], 
        left_on = ['opponent'], 
        right_on = ['fighter'], 
        how = 'left')['stance']

    # Get the historic records of each fighter for each opponent stance
    fight_stats_hist_opponent_stance = aggregate_last_X_fight_data(
        fight_stats,
        ignore_cols = ['date', 'event', 'bout', 'fighter', 'opponent', 'opponent_stance'],
        group_by = ['fighter', 'opponent_stance'],
        prefix = 'hist_opp_stance_',
        X = 100,
        per_minute_calc = True, 
        bayes_priors_df = fight_stats_bayes_priors,
        bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
    ).drop(columns = ['date', 'opponent', 'opponent_stance'])

    # Check the row numbers are correct
    stop_if_not(len(fight_stats) == len(fight_stats_hist_opponent_stance), "Opponent data length check")






    ###########################################################################################
    # Calculate per minute stats for the many hist datasets

    # Check all the row numbers are correct
    stop_if_not(
        len(fight_stats) == len(fight_stats_hist_opponent_stance) == len(fight_stats_hist_last_3),
        "All hist stats data length check")


    ###########################################################################################
    # Restructure the result data into fighter level i.e. duplicate for each fighter then concat

    temp_df = pd.DataFrame()
    temp_df[['fighter1_result', 'fighter2_result']] = fight_results['outcome'].str.split('/', expand = True)
    fighter1 = fight_results.copy().drop(['fighter1_start_elo', 'fighter2_start_elo'], axis = 1)
    fighter1['result'] = temp_df['fighter1_result']
    fighter1['fighter'] = fight_results['fighter1']
    fighter2 = fight_results.copy().drop(['fighter1_start_elo', 'fighter2_start_elo'], axis = 1)
    fighter2['result'] = temp_df['fighter2_result']
    del(temp_df)
    fighter2['fighter'] = fight_results['fighter2']
    fighter1.drop(columns = ['fighter1', 'fighter2', 'outcome'], inplace = True)
    fighter2.drop(columns = ['fighter1', 'fighter2', 'outcome'], inplace = True)
    fighter_results = pd.concat([fighter1, fighter2], ignore_index = True)
    del([fighter1, fighter2])
    fighter_results.sort_values(by = ['fighter', 'date'], inplace = True)

    # Change NC result to be one letter for the sake of strings of results later
    fighter_results['result'] = np.where(fighter_results['result'] == 'NC', 'N', fighter_results['result'])

    # Check we have the correct number of rows
    if len(fighter_results) == 2 * len(fight_results):
        print('Row number check PASSED')
    else:
        print('Merge row check FAILED')
        raise SystemExit

    ###########################################################################################
    # Calculate number of previous wins/losses

    fighter_results = calculate_win_loss_stats(fighter_results)

    # Check that the win and loss finshes make sense
    check = fighter_results[(fighter_results['hist_wins'] < fighter_results['hist_wins_finishes']) | (fighter_results['hist_losses'] < fighter_results['hist_losses_finishes'])]

    stop_if_not(len(check) == 0, "Hist finishes and wins/losses check")

    ###########################################################################################
    # Record the last 10 results
    for n in list(range(1, 11)):
        fighter_results['result_' + str(n) +'_fights_prior'] = fighter_results.groupby(['fighter'])['result'].shift(n).fillna('')
        if n == 2:
            fighter_results['last_' + str(n) + '_results'] = fighter_results['result_' + str(n) +'_fights_prior'] + fighter_results['result_' + str(n - 1) +'_fights_prior']
        elif n > 2:
            fighter_results['last_' + str(n) + '_results'] = fighter_results['result_' + str(n) +'_fights_prior'] + fighter_results['last_' + str(n - 1) + '_results']
        else:
            pass

    ###########################################################################################
    # Get the fight stats of each fighter against only specific types of fighters

    # First define each fighter type
    fight_stats_hist['fighter_type'] = define_fighter_type(fight_stats_hist, fighter_results, config)

    # Add this fighter type to the stats data
    fight_stats['fighter_type'] = fight_stats.merge(fight_stats_hist[['event', 'bout', 'fighter', 'fighter_type']], on = ['event', 'bout', 'fighter'], how = 'left')['fighter_type']

    # Now create a dataset with only opponents of that type
    types = fight_stats_hist['fighter_type'].str.replace(' ', '').unique().tolist()
    types = [type for type in types if type is not None]

    # Merge on opponent type
    fight_stats['opponent_type'] = fight_stats[['event', 'bout', 'opponent']].merge(
        fight_stats_hist[['event', 'bout', 'fighter', 'fighter_type']], 
        left_on = ['event', 'bout', 'opponent'], 
        right_on = ['event', 'bout', 'fighter'],
        how = 'left')['fighter_type']

    # Get the historic records of each fighter for each opponent type
    fight_stats_hist_opponent_type = aggregate_last_X_fight_data(
        fight_stats,
        ignore_cols = ['date', 'event', 'bout', 'fighter', 'opponent', 'opponent_stance', 'fighter_type', 'opponent_type'],
        group_by = ['fighter', 'opponent_type'],
        prefix = 'hist_opp_type_',
        X = 100,
        per_minute_calc = True, 
        bayes_priors_df = fight_stats_bayes_priors,
        bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
    ).drop(columns = ['date', 'opponent', 'opponent_stance', 'fighter_type', 'opponent_type'])


    # Check that we have the correct number of rows
    stop_if_not(len(fight_stats_hist_opponent_type) == len(fight_stats_hist), 'Opponent type row count check')

    # test = fight_stats_hist_opponent_type[fight_stats_hist_opponent_type['fighter']=='ConorMcGregor'][['bout', 'fighter', 'opponent', 'opponent_type', 'hist_opp_type_allrounds_fight_minutes']]

    ###########################################################################################
    # Calculate the date was last knocked out
    dates_knocked_out = fighter_results.copy()
    dates_knocked_out['knocked_out'] = np.where((dates_knocked_out['method'] == 'KO/TKO') & (dates_knocked_out['result'] == 'L'), 1, 0)
    dates_knocked_out['date_knocked_out'] = np.where(dates_knocked_out['knocked_out'] == 1, dates_knocked_out['date'], np.datetime64('NaT'))

    dates_knocked_out.sort_values(by = ['fighter', 'date'], inplace = True)

    # Calculate date last knocked out
    dates_knocked_out['date_last_knocked_out'] = dates_knocked_out.groupby(['fighter'])['date_knocked_out'].shift(1)
    dates_knocked_out['date_last_knocked_out'] = dates_knocked_out.groupby(['fighter'])['date_last_knocked_out'].ffill()

    # Calculate the number times knocked out previously
    dates_knocked_out['hist_times_knocked_out'] = dates_knocked_out.groupby(['fighter'])['knocked_out'].cumsum() - dates_knocked_out['knocked_out']

    row_check = len(dates_knocked_out)

    # Calculate how many times each round they've been knocked out
    for round in [1, 2, 3, 4, 5]:
        round_knocked_out = dates_knocked_out[(dates_knocked_out['round'] == round) & dates_knocked_out['knocked_out'] == 1].copy()
        round_knocked_out['hist_times_knocked_out_round_' + str(round)] = round_knocked_out.groupby(['fighter'])['knocked_out'].cumsum() - 1
        dates_knocked_out = dates_knocked_out.merge(round_knocked_out[['event', 'bout', 'fighter', 'hist_times_knocked_out_round_' + str(round)]], 
                                                    on = ['event', 'bout', 'fighter'], 
                                                    how = 'left')
        del(round_knocked_out)

    dates_knocked_out = dates_knocked_out[['event',
                                           'bout',
                                           'fighter', 
                                           'date_last_knocked_out', 
                                           'hist_times_knocked_out',
                                           'hist_times_knocked_out_round_1',
                                           'hist_times_knocked_out_round_2',
                                           'hist_times_knocked_out_round_3',
                                           'hist_times_knocked_out_round_4',
                                           'hist_times_knocked_out_round_5']]

    # Fill NAs with 0
    dates_knocked_out.loc[:, ['hist_times_knocked_out', 'hist_times_knocked_out_round_1', 'hist_times_knocked_out_round_2', 'hist_times_knocked_out_round_3', 'hist_times_knocked_out_round_4', 'hist_times_knocked_out_round_5']] = dates_knocked_out[['hist_times_knocked_out', 'hist_times_knocked_out_round_1', 'hist_times_knocked_out_round_2', 'hist_times_knocked_out_round_3', 'hist_times_knocked_out_round_4', 'hist_times_knocked_out_round_5']].fillna(0) 

    # Fill NAs in dates with arbitrary early date
    dates_knocked_out['date_last_knocked_out'] = pd.to_datetime(dates_knocked_out['date_last_knocked_out'].fillna('1900-01-01'))

    # Check number of rows hasn't changed
    if len(dates_knocked_out) == row_check:
        print('Merge row check PASSED')
    else:
        print('Merge row check FAILED')
        raise SystemExit
    del(row_check)

    ###########################################################################################
    # Calculate win and loss streaks
    fighter_results['hist_win_streak'] = 0
    fighter_results['hist_loss_streak'] = 0
    for streak in list(range(1, 11)):
        fighter_results['hist_win_streak'] = np.where(fighter_results['last_10_results'].str[-streak:] == ('W' * streak),
                                                      streak,
                                                      fighter_results['hist_win_streak'])
        fighter_results['hist_loss_streak'] = np.where(fighter_results['last_10_results'].str[-streak:] == ('L' * streak),
                                                       streak,
                                                       fighter_results['hist_loss_streak'])

    fighter_results['hist_win_streak'].value_counts()
    # Check the fighters that have been on win streaks of 10
    fighter_results[fighter_results['hist_win_streak'] == 10]['fighter'].unique()
    fighter_results[fighter_results['hist_loss_streak'] == 5]['fighter'].unique()

    # Calculate results by method
    methods = list(set(fighter_results['method'].dropna()))
    row_count = len(fighter_results)
    for result in ['W', 'L']:
        for method in methods:
            # First calculate a simple flag for whether the win or loss was by that method
            fighter_results['hist_' + result + '_' + method] = np.where((fighter_results['method'] == method) & (fighter_results['result'] == result),                                                                   1, 0)
            # Then sum these for previous fights ignoring the current one
            fighter_results['hist_' + result + '_' + method] = fighter_results.groupby(['fighter'])['hist_' + result + '_' + method].cumsum() - fighter_results['hist_' + result + '_' + method]

    # Cleanup columns not needed
    fighter_results = fighter_results[[col for col in fighter_results.columns if not (('last_' in col and '_result' in col) or 'fights_prior' in col)]]


    # Check row count before and after merge
    if len(fighter_results) == row_count:
        print('Row number check PASSED')
    else:
        print('Merge row check FAILED')
        raise SystemExit
    del(row_count)

    ###########################################################################################
    # Remove result as that isn't historical
    fighter_results.drop(columns = ['result', 'method', 'weightclass', 'round', 'rounds', 'time format', 'date'], inplace = True)

        

    # Add on the date of the last fight
    fight_stats['date_last_fight'] = fight_stats.groupby(['fighter'])['date'].shift(1)

    # Date check
    fight_stats_no_na = fight_stats[fight_stats['date_last_fight'].notna() & (fight_stats['date_last_fight'] > fight_stats['date'])]
    if len(fight_stats_no_na) == 0:
        print("Date of last fight check PASSED")
    else:
        print("Date of last fight check FAILED")
        raise SystemExit
    del(fight_stats_no_na)

    ###########################################################################################
    # FIGHT RESULTS PROCESSING

    # Filter for fight formats to be included
    fight_results = fight_results[((fight_results['time format'].isna()) | (fight_results['time format'].isin(config['time_format_filter']))) & fight_results['weightclass'].isin(config['weightclass_filter'])]

    # # Add elevation data to fight results
    # events['fight_elevation'] = get_elevation_series(events['location'], config['google_maps_api_key'])

    # # Merge onto the fight results
    # fight_results = fight_results.merge(
    #     events[['event', 'fight_elevation']],
    #     on = 'event',
    #     how = 'left'
    #     )

    # # Check there are no missing elevations
    # stop_if_not(fight_results['fight_elevation'].isna().sum() == 0, "Event elevation missing data check")

    ###########################################################################################
    # FIGHT ODDS PROCESSING

    # Fix the events and bouts using data including upcoming events. Doing this again after done in cleaning because events and stats been edited
    fight_odds = add_event_and_bout_to_odds_data(fight_odds, events, fight_stats)
    event_odds = add_event_and_bout_to_odds_data(event_odds, events, fight_stats, date_check = '2018-01-01')

    # Add opponent to the upcoming odds. It already has the event and bout matched on.
    add_opponent_column(betfair_upcoming_odds)

    # Combine the upcoming odds with the historic odds
    betfair_odds = pd.concat([betfair_odds, betfair_upcoming_odds], axis = 0)

    fight_odds = process_odds_data(fight_odds.drop_duplicates(subset = ['event', 'bout', 'fighter']), event_odds, betfair_odds, odds_scrape_config, average_odds_model)


    ###########################################################################################
    # MERGE DATA

    merged_data = merge_df_both_fighters(
        left_df =  fight_results[['event', 'bout', 'fighter1', 'fighter2', 'outcome', 'method', 'weightclass', 'date', 'time format', 'fighter1_start_elo', 'fighter2_start_elo']],
        df_list = [
            fight_stats[['event', 'bout', 'fighter', 'date_last_fight']],
            dates_knocked_out,
            fight_stats_hist,
            fight_stats_hist_last_3,
            fight_stats_hist_opponent_stance,
            fight_stats_hist_opponent_type,
            fighter_results,
            fight_odds
        ]
    ).reset_index(drop = True)

    # Save rows for checks later
    merged_original_rows = len(merged_data)
    
    # Merge on the fighter information
    merged_data = merged_data.merge(fighters.add_prefix('fighter1_'), 
                                    left_on = ['fighter1'], 
                                    right_on = ['fighter1_fighter'],
                                    how = 'left')
    merged_data = merged_data.merge(fighters.add_prefix('fighter2_'), 
                                    left_on = ['fighter2'], 
                                    right_on = ['fighter2_fighter'],
                                    how = 'left')

    merged_data.drop(columns = ['fighter2_fighter', 'fighter1_fighter'], inplace = True)

    # Check for no duplicate columns that have been merged
    if len([col for col in merged_data.columns if "_x" in col]) == 0:
        print('Duplicate column check PASSED')
    else:
        duplicates = [col for col in merged_data.columns if "_x" in col]
        print(f'Duplicate column name check FAILED: {duplicates}')
        raise SystemExit

    # Check we have the right number of rows
    if len(merged_data) == len(fight_results):
        print("Merged data rows check PASSED")
    else:
        print("Merged data rows check FAILED")
        raise SystemExit

    # Check for duplicate column names
    duplicate_columns = merged_data.columns[merged_data.columns.duplicated()]

    # Cleanup
    del betfair_odds, fight_results, fight_stats, dates_knocked_out, fight_stats_hist, fight_stats_hist_last_3, fight_stats_hist_opponent_stance, fight_stats_hist_opponent_type,  fighter_results, fight_odds, events, event_odds, fighters
                                        
    gc.collect();
    # pdb.set_trace()

    ###########################################################################################
    # FURTHER FEATURE PROCESSING

    # Calculate all true odds
    merged_data = calculate_all_true_odds_columns(merged_data, impute_missing_betfair = data_processing_config['impute_missing_betfair'], odds_suffix_for_true_odds = data_processing_config['odds_suffix_for_true_odds'], impute_one_sided_missing_odds = data_processing_config['impute_one_sided_missing_odds'])

    # Fix model hyperparams where necessary to account for imputation being used
    if (data_processing_config['impute_missing_betfair'] == True) & (model_hyper_params['odds_to_use'] == 'betfair_exch_odds'):
        model_hyper_params['odds_to_use'] = 'betfair_exch_odds_imputed'

    # Create a QA dataset showing where Betfair odds and scraped odds are showing large differences
    odds_qa_df = merged_data[((
        merged_data['fighter1_betfair_exch_odds'] < 1.9) & (merged_data['fighter1_best_closing_odds'] > 2.1)) | ((
            merged_data['fighter1_betfair_exch_odds'] > 2.1) & (merged_data['fighter1_best_closing_odds'] < 1.9)) | ((
                merged_data['fighter1_betfair_exch_odds'] < 1.9) & (merged_data['fighter2_betfair_exch_odds'] < 1.9))][['fighter1', 'fighter2', 'date', 'fighter1_betfair_exch_odds', 'fighter1_best_closing_odds', 'fighter2_betfair_exch_odds', 'fighter2_best_closing_odds']]

    odds_qa_df.to_csv('./data/qa_merged_betfair_odds_out_of_sync.csv')   
                
    ###########################################################################################
    # PERCENTAGE CALCULATIONS

    merged_data = calculate_percentages(merged_data, fight_stats_bayes_priors, bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_percent'])


    ###########################################################################################
    # RATIO CALCULATIONS

    # Get the columns for attempted strikes but not against
    head_columns = [col for col in merged_data.columns if '_head_attempt' in col and 'against' not in col and 'per_minute' not in col]
    body_columns = [col for col in merged_data.columns if 'body_attempt' in col and 'against' not in col and 'per_minute' not in col]
    leg_columns = [col for col in merged_data.columns if 'leg_attempt' in col and 'against' not in col and 'per_minute' not in col]

    # Do checks on columns
    stop_if_not(check_identical_excl_string(head_columns, body_columns, 'head', 'body'))
    stop_if_not(check_identical_excl_string(head_columns, leg_columns, 'head', 'leg'))

    # Calculate ratios of attempts on body and leg
    head_body_ratios = pd.DataFrame(
        safe_divide_np(merged_data[head_columns].to_numpy(), merged_data[body_columns].to_numpy(), zero_zero_fill = np.nan),
        columns = [col.replace('head_attempt', 'head_body_attempt_ratio') for col in head_columns])

    merged_data = pd.concat([merged_data, head_body_ratios], axis = 1)

    head_leg_ratios = pd.DataFrame(
        safe_divide_np(merged_data[head_columns].to_numpy(), merged_data[leg_columns].to_numpy(), zero_zero_fill = np.nan),
        columns = [col.replace('head_attempt', 'head_leg_attempt_ratio') for col in head_columns])

    merged_data = pd.concat([merged_data, head_leg_ratios], axis = 1)

    del(head_body_ratios, head_leg_ratios, head_columns, body_columns, leg_columns)

    # Do a check to see if row number has stayed the same
    stop_if_not(len(merged_data) == merged_original_rows)

    # Calculate ratio of striking to takedowns by looking at distance striking compared with td attempts, and distance compared with ground and pound
    for fighter in ['fighter1', 'fighter2']:
        
        merged_data[fighter + '_hist_total_distance_strikes_over_td_attempts'] = safe_divide_np(
            merged_data[fighter + '_hist_allrounds_distance_attempt'].to_numpy(), 
            merged_data[fighter + '_hist_allrounds_td_attempt'].to_numpy(), 
            zero_zero_fill = np.nan)
        
        merged_data[fighter + '_hist_total_distance_over_ground_strikes'] = safe_divide_np(
            merged_data[fighter + '_hist_allrounds_distance_attempt'].to_numpy(),
            merged_data[fighter + '_hist_allrounds_ground_attempt'].to_numpy(),
            zero_zero_fill = np.nan)

    gc.collect();

    ###########################################################################################
    # ROUND DIFFERENTIAL CALCULATIONS

    # Look at the changes in per minute striking from round to round to gauge some measure of stamina and accuracy
    for fighter in ['fighter1', 'fighter2']:
        for round in range(2, 6):
            merged_data[fighter + '_round' + str(round) + 'to' + str(round - 1) + '_total_strike_per_minute_ratio'] = \
            merged_data[fighter + '_hist_round' + str(round) + '_total str._attempt_per_minute'] / merged_data[fighter + '_hist_round' + str(round - 1) + '_total str._attempt_per_minute']

            merged_data[fighter + '_round' + str(round) + 'to' + str(round - 1) + '_total_strike_accuracy_ratio'] = \
            merged_data[fighter + '_hist_round' + str(round) + '_total str._accuracy_percent'] / merged_data[fighter + '_hist_round' + str(round - 1) + '_total str._accuracy_percent']
            
            merged_data[fighter + '_last_3_fights_round' + str(round) + 'to' + str(round - 1) + '_total_strike_per_minute_ratio'] = \
            merged_data[fighter + '_hist_last_3_fights_round' + str(round) + '_total str._attempt_per_minute'] / merged_data[fighter + '_hist_last_3_fights_round' + str(round - 1) + '_total str._attempt_per_minute']

            merged_data[fighter + '_last_3_fights_round' + str(round) + 'to' + str(round - 1) + '_total_strike_accuracy_ratio'] = \
            merged_data[fighter + '_hist_last_3_fights_round' + str(round) + '_total str._accuracy_percent'] / merged_data[fighter + '_hist_last_3_fights_round' + str(round - 1) + '_total str._accuracy_percent']
            
    # Do a check to see if row number has stayed the same
    stop_if_not(len(merged_data) == merged_original_rows)

    ###########################################################################################
    # PER FIGHT CALCULATIONS

    # To see what columns are left
    # merged_data[[col for col in merged_data.columns if '_attempt' not in col and '_landed' not in col and '_percent' not in col and '_minutes' not in col]].columns.tolist()

    numerator_strings = ['times_knocked_out']

    # Do each fighters per fight calcs separately
    for fighter in ['fighter1', 'fighter2']:
        # Get the column names
        cols = []
        for i in range(0, len(numerator_strings)):
            cols = cols + [col for col in merged_data.columns if fighter in col and numerator_strings[i] in col]
        new_data = merged_data[cols].div(merged_data[fighter + '_hist_fights'], axis = 0)
        # Deal with any infinities
        new_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        new_data.columns = [col + '_per_fight' for col in cols]
        
        # Merge with the data
        merged_data = pd.concat([merged_data, new_data], axis = 1)

    del([new_data, cols, numerator_strings])

    ###########################################################################################
    # OTHER 'PER' CALCULATIONS

    # This is a set of custom features to identify what kind of fighter someone is

    for fighter in ['fighter1', 'fighter2']:
        # Strikes per TD attempt
        merged_data[fighter + '_sig.str._per_td_attempt'] = merged_data[fighter + '_hist_allrounds_sig.str._attempt'] / merged_data[fighter + '_hist_allrounds_td_attempt']
        # Strikes per sub attempt
        merged_data[fighter + '_sig.str.per_sub.att'] = merged_data[fighter + '_hist_allrounds_sig.str._attempt'] / merged_data[fighter + '_hist_allrounds_sub.att']
        # Knockdowns per strike landed
        merged_data[fighter + '_kd_per_sig.str._landed'] = merged_data[fighter + '_hist_allrounds_kd'] / merged_data[fighter + '_hist_allrounds_sig.str._landed']
        # Sub attempt per td
        merged_data[fighter + '_sub.att.per_td_landed'] = merged_data[fighter + '_hist_allrounds_sub.att'] / merged_data[fighter + '_hist_allrounds_td_landed']
            
        # Subs per attempt
        merged_data[fighter + '_subs_per_attempt'] = merged_data[fighter + '_hist_wins_Submission'] / merged_data[fighter + '_hist_allrounds_sub.att']
        
        # Recode Infs as missing, because could give misleading results
        new_columns = [fighter + '_sig.str._per_td_attempt', fighter + '_sig.str.per_sub.att', fighter + '_kd_per_sig.str._landed', fighter + '_sub.att.per_td_landed', fighter + '_subs_per_attempt']
        
        merged_data[new_columns] = merged_data[new_columns].fillna(-np.inf)
        # merged_data[new_columns] = merged_data[new_columns].replace([-np.inf, np.inf], np.nans)


    ###########################################################################################
    # TIME SINCE CALCULATIONS

    # Calculate days since last KO, Win, Loss, Fight
    # Set the columns to be taken from date
    cols = ['_date_last_fight', '_date_last_knocked_out', '_date_last_loss', '_date_last_win', '_dob']
    # Do each fighters per fight calcs separately
    for fighter in ['fighter1', 'fighter2']:
        for col in cols:
            merged_data[fighter + '_date_minus' + col] = (merged_data['date'] - pd.to_datetime(merged_data[fighter + col]).copy()).dt.days

            # Fill in NAs with arbitrary early date
            # merged_data[fighter + '_date_minus' + col] = merged_data[fighter + '_date_minus' + col].fillna(99999).astype(int)

    # pd.DataFrame(merged_data.columns).to_csv("te.csv")
    ###########################################################################################
    # FIGHTER DIFFERENTIAL CALCULATIONS

    merged_data = add_fighter_differentials(merged_data)
    
    gc.collect();

    ###########################################################################################
    # Formatting tweaks
    # Remove any white spaces in names
    merged_data.columns = merged_data.columns.str.replace(" ", "_")

    # Add a womens flag
    merged_data['womens'] = np.where(merged_data['weightclass'].apply(lambda x: np.where('omen' in x, True, False)), 1, 0)

    # Add the matchup
    merged_data['matchup'] = merged_data['fighter1_fighter_type'] + '-vs-' + merged_data['fighter2_fighter_type']


    # Add what odds are to be used in model training and backtesting
    merged_data['fighter1_backtesting_odds'] = merged_data['fighter1_' + model_hyper_params['odds_to_use']].copy()
    merged_data['fighter2_backtesting_odds'] = merged_data['fighter2_' + model_hyper_params['odds_to_use']].copy()


    ###########################################################################################
    # FLAG TRAIN/TEST/OOT

    np.random.seed(seed = config['seed_all'] + 7156)
    merged_data = flag_train_test_oot(merged_data, config)



    ###########################################################################################
    # OUTPUT DATA

    # Remove columns not needed, or that are all na
    cols_to_remove = [col for col in merged_data.columns if any(substring in col for substring in ['level_0'])]
    # cols_to_remove.append('date')

    merged_data.drop(columns = cols_to_remove, inplace = True)

    # # Drop any columns that are all NA. Commented out as it affects
    # merged_data.dropna(axis = 1, how = 'all', inplace = True)

    # print(merged_data.columns)
    merged_data.reset_index(drop = True, inplace = True)
    merged_data.drop(columns = cols_to_remove, inplace = True, errors = 'ignore')
    del(cols_to_remove)
    
    if not skip_save_data:
        
        # Save the log of the final data
        log_final_data_stats(merged_data, config['final_data_stats_log'], config)
        
        # Save the final data
        # split_by_year_and_save_fastparquet(merged_data, 'date', './data/final_dataset/', compression='ZSTD')
        fp.write('./data/final_dataset/final_dataset.parquet', merged_data, compression = 'ZSTD')
        
        # Output a single fight for checking purposes
        merged_data[(merged_data['fighter1'] == 'khabibnurmagomedov') & (merged_data['fighter2'] == 'conormcgregor')].T.to_csv('./data/final_dataset_checking.csv', index=True)
        
        merged_data[(merged_data['fighter1'] == 'antoniosilva') | (merged_data['fighter2'] == 'antoniosilva')].T.to_csv('./data/final_dataset_fighter_checking.csv')
    
    return merged_data