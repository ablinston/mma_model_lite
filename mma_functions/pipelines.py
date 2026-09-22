from ablinston_ml import stop_if_not
from ablinston_ml.scraping import get_website_soup
from datetime import date, datetime, timedelta
import fastparquet as fp
import gc
import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import time
import xgboost as xgb

from mma_functions import betfair, cleaning, dataset, elo, evaluation, features, modelling, odds, odds_scraping, parsing, scraping, tracking, utils
from mma_functions.config import bet_config, betfair_data_config, config, elo_config, model_hyper_params, odds_scrape_config

# This is the master function for processing all the clean data, calculating variables and merging together

def merge_and_process_all_clean_data(
        config,
        elo_config,
        data_processing_config,
        odds_scrape_config,
        model_hyper_params,
        test_fighter_list = [],
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
    fight_stats = dataset.add_upcoming_to_fight_stats(fight_stats, upcoming_fight_details, events, upcoming_events)

    # Add upcoming events
    events = pd.concat([upcoming_events, events], axis = 0)

    # Add the date to the upcoming fights
    upcoming_fight_details = upcoming_fight_details.merge(upcoming_events[['event', 'date']], on = 'event', how = 'left')

    # Add upcoming fights to the results with empty results
    fight_results = pd.concat([upcoming_fight_details.drop('url', axis = 1), fight_results])



    ###########################################################################################
    # CALCULATE ELOs

    fight_results = elo.add_start_elo_to_fight_results(fight_results, fight_stats, config, elo_config)

    # Recode missing to the default starting. There will be some missing if their first fight was a DQ, D/D or NC.

    # Check there are no missing ELOs after the start date
    missing_elo_check = fight_results[fight_results['date'] > '2000-01-01'][pd.isna(fight_results['fighter1_start_elo'])]

    # Output to file before treating
    missing_elo_check.to_csv('./data/missing_elo_check.csv')

    fight_results['fighter1_start_elo'] = fight_results['fighter1_start_elo'].fillna(config['start_elo'])
    fight_results['fighter2_start_elo'] = fight_results['fighter2_start_elo'].fillna(config['start_elo'])


    ###########################################################################################
    # FIGHT STATS PROCESSING

    # Calculate the historical stats of each fighter at the fight date
    fight_stats = fight_stats.sort_values(by=['fighter', 'date'], axis = 0)
    
    # Create subs attempted and landed versions
    fight_stats.columns = fight_stats.columns.str.replace('_sub.att', '_sub_attempt')
    
    # Get submission success
    fight_stats = features.add_submissions_landed(fight_stats, fight_results)

    # Copy the fight stats but for against a fighter, then merge onto stats
    fight_stats_against = fight_stats.copy()
    fight_stats_against.rename(columns = lambda x: x + "_against" if "attempt" in x else x, inplace = True)
    fight_stats_against.rename(columns = lambda x: x + "_against" if "landed" in x else x, inplace = True)
    fight_stats_against.rename(columns = lambda x: x + "_against" if "ctrl_minutes" in x else x, inplace = True)
    fight_stats_against.rename(columns = lambda x: x + "_against" if "kd" in x else x, inplace = True)

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
    
    if data_processing_config['add_bayes_percentage_columns'] or data_processing_config['add_bayes_per_minute_columns']:
        
        # FIrst calculate the Bayesian priors that will be used for Bayesian smoothing
        fight_stats_bayes_priors = features.bayesian_priors(
            fight_stats.sort_values('date').reset_index(drop = True).merge(fight_results[['event', 'bout', 'weightclass']], on = ['event', 'bout'], how = 'left'), # Join on the weightclass for grouping
            id_cols = ['event', 'bout', 'fighter', 'date'], 
            group_by = data_processing_config['bayesian_groupby'], 
            fight_span = data_processing_config['bayes_fightspan'],
            ignore_last_n_fights = 30 if data_processing_config['bayesian_groupby'] == [] else 10
        )
        
        # Use constant Bayesian priors if required
        if data_processing_config['constant_bayes_priors'] == True:
            
            # Get the single row for constant bayes using the max date
            max_date = fight_stats_bayes_priors[fight_stats_bayes_priors['date'] < pd.to_datetime(data_processing_config['constant_bayes_priors_date'])]['date'].max()
            constant_bayes = fight_stats_bayes_priors[fight_stats_bayes_priors['date'] == max_date].head(1)
            
            # Now set these as the bayes stats for everyone
            fight_stats_bayes_priors = fight_stats_bayes_priors[['event', 'bout']].merge(
                constant_bayes.drop(columns = ['event', 'bout', 'date']), how = 'cross')
        else:
            fight_stats_bayes_priors.drop(columns = ['date'], inplace = True)
        
    else:
        fight_stats_bayes_priors = None

    # Now aggrgate
    fight_stats_hist = features.aggregate_last_X_fight_data(
        fight_stats,
        id_cols = ['date', 'event', 'bout', 'fighter'],
        group_by = ['fighter'],
        prefix = "hist_",
        X = 100,
        per_minute_bayes = data_processing_config['add_bayes_per_minute_columns'],
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
    
    if data_processing_config['add_bayes_per_minute_columns'] and data_processing_config['add_adj_per_minute_bayes_columns']:
        
        fight_stats_bayes = features.calculate_per_minute_bayesian_stats(fight_stats, fight_stats_bayes_priors, excluded_strings = ['percent', 'fight_length', 'fight_minute', 'ratio', 'per_minute'], bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute'])
        
        # Adding on adjusted stats to the fight stats
        fight_stats_plus_adj = features.add_adjusted_stats(fight_stats_bayes, fight_stats_hist, columns_with = ['per_minute_bayes'], exclude_columns_with = [])
        
        # Replace the fight stats with one with the adj but remove all the extra bayes info (will be added by aggregation later)
        fight_stats = fight_stats_plus_adj[[col for col in fight_stats_plus_adj if 'bayes' not in col or ('bayes' in col and 'adj' in col)]]
    
    if data_processing_config['add_adj_per_minute_columns']:
        fight_stats = features.add_adjusted_stats(fight_stats, fight_stats_hist, columns_with = ['per_minute'], exclude_columns_with = ['bayes'])
    
    
    ###########################################################################################
    # Calculate accuracy percentages
    
    fight_stats = features.calculate_adjusted_percentages(
            fight_stats, 
            fight_stats_hist, 
            fight_stats_bayes_priors,
            data_processing_config)
    
    # Only keep the adj percentage columns. We don't want to aggregate the rest, just calc later from agg counts
    fight_stats = fight_stats[[col for col in fight_stats.columns if not ('accuracy_percent' in col and '_adj' not in col)]]
    
    ###########################################################################################
    # Reaggregate the data
    fight_stats_hist = features.aggregate_last_X_fight_data(
        fight_stats,
        id_cols = ['date', 'event', 'bout', 'fighter'],
        group_by = ['fighter'],
        prefix = "hist_",
        X = 100,
        per_minute_bayes = data_processing_config['add_bayes_per_minute_columns'],
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
    fight_stats_hist_last_3 = features.aggregate_last_X_fight_data(
        fight_stats,
        id_cols = ['date', 'event', 'bout', 'fighter'],
        group_by = ['fighter'],
        prefix = "hist_last_3_fights_",
        X = 3,
        per_minute_bayes = data_processing_config['add_bayes_per_minute_columns'],
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

    # fight_stats_hist_last_2 = aggregate_last_X_fight_data(
    #     fight_stats,
    #     id_cols = ['date', 'event', 'bout', 'fighter'],
    #     group_by = ['fighter'],
    #     prefix = "hist_last_2_fights_",
    #     X = 2,
    # per_minute_bayes = data_processing_config['add_bayes_per_minute_columns'],
    #     per_minute_calc = True, 
    #     bayes_priors_df = fight_stats_bayes_priors,
    #     bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
    # ).drop(columns = ['date'])
    
    # fight_stats_hist_last_1 = aggregate_last_X_fight_data(
    #     fight_stats,
    #     id_cols = ['date', 'event', 'bout', 'fighter'],
    #     group_by = ['fighter'],
    #     prefix = "hist_last_1_fights_",
    #     X = 1,
    # per_minute_bayes = data_processing_config['add_bayes_per_minute_columns'],
    #     per_minute_calc = True, 
    #     bayes_priors_df = fight_stats_bayes_priors,
    #     bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
    # ).drop(columns = ['date'])
    

    ###########################################################################################
    # Get the fight stats of each fighter against only specific types of fighters
    
    fight_stats = dataset.add_opponent_column(fight_stats.copy())
    
    # Get the historic records of each fighter for each opponent stance
    fight_stats_hist_opponent_stance = features.calculate_hist_opp_stance_stats(fight_stats, fighters, fight_stats_bayes_priors, data_processing_config)


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
    # Calculate number of previous wins/losses - NONE OF THESE HAD STRONG ENOUGH PREDICTIVE POWER

    fighter_results = features.calculate_win_loss_stats(fighter_results)

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
    fight_stats_hist['fighter_type'] = features.define_fighter_type(fight_stats_hist, fighter_results, config)

    # Add this fighter type to the stats data
    fight_stats = fight_stats.merge(fight_stats_hist[['event', 'bout', 'fighter', 'fighter_type']], on = ['event', 'bout', 'fighter'], how = 'left')

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
    fight_stats_hist_opponent_type = features.aggregate_last_X_fight_data(
        fight_stats,
        id_cols = ['date', 'event', 'bout', 'fighter', 'opponent', 'fighter_type', 'opponent_type'],
        group_by = ['fighter', 'opponent_type'],
        prefix = 'hist_opp_type_',
        X = 100,
        per_minute_bayes = data_processing_config['add_bayes_per_minute_columns'],
        per_minute_calc = True, 
        bayes_priors_df = fight_stats_bayes_priors,
        bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
    ).drop(columns = ['date', 'opponent', 'fighter_type', 'opponent_type'])


    # Check that we have the correct number of rows
    stop_if_not(len(fight_stats_hist_opponent_type) == len(fight_stats_hist), 'Opponent type row count check')

    # test = fight_stats_hist_opponent_type[fight_stats_hist_opponent_type['fighter']=='ConorMcGregor'][['bout', 'fighter', 'opponent', 'opponent_type', 'hist_opp_type_allrounds_fight_minutes']]
    
    ###########################################################################################
    # Calculate the date was last knocked out
    
    dates_knocked_out = features.calculate_knocked_out_stats(fighter_results)

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
    fight_odds = odds.add_event_and_bout_to_odds_data(fight_odds, events, fight_stats)
    event_odds = odds.add_event_and_bout_to_odds_data(event_odds, events, fight_stats, date_check = '2018-01-01')

    # Add opponent to the upcoming odds. It already has the event and bout matched on.
    if len(betfair_upcoming_odds) > 0:
        dataset.add_opponent_column(betfair_upcoming_odds)

        # Combine the upcoming odds with the historic odds
        betfair_odds = pd.concat([betfair_odds, betfair_upcoming_odds], axis = 0)

    fight_odds = odds.process_odds_data(fight_odds.drop_duplicates(subset = ['event', 'bout', 'fighter']), event_odds, betfair_odds, odds_scrape_config, average_odds_model)


    ###########################################################################################
    # MERGE DATA

    merged_data = dataset.merge_df_both_fighters(
        left_df =  fight_results[['event', 'bout', 'fighter1', 'fighter2', 'outcome', 'method', 'weightclass', 'date', 'time format', 'fighter1_start_elo', 'fighter2_start_elo']],
        df_list = [
            fight_stats[['event', 'bout', 'fighter', 'date_last_fight']],
            dates_knocked_out,
            fight_stats_hist,
            # fight_stats_hist_last_1,
            # fight_stats_hist_last_2,
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
    merged_data = odds.calculate_all_true_odds_columns(merged_data, impute_missing_betfair = data_processing_config['impute_missing_betfair'], odds_suffix_for_true_odds = data_processing_config['odds_suffix_for_true_odds'], impute_one_sided_missing_odds = data_processing_config['impute_one_sided_missing_odds'])

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
    # PERCENTAGE CALCULATIONS - MOVED TO BEFORE MERGE
    
    if data_processing_config['add_bayes_percentage_columns']:
        merged_data = features.calculate_percentages(merged_data, fight_stats_bayes_priors, bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_percent'])
        
        # # Calculate bayes adj accuracy if required
        # if data_processing_config['add_adj_percentage_bayes_columns']:
        #     merged_data = add_()
    else:
        merged_data = features.calculate_percentages(merged_data, fight_stats_bayes_priors = None, bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_percent'])

    ###########################################################################################
    # RATIO CALCULATIONS

    # Get the columns for attempted strikes but not against
    head_columns = [col for col in merged_data.columns if '_head_attempt' in col and 'against' not in col and 'per_minute' not in col]
    body_columns = [col for col in merged_data.columns if 'body_attempt' in col and 'against' not in col and 'per_minute' not in col]
    leg_columns = [col for col in merged_data.columns if 'leg_attempt' in col and 'against' not in col and 'per_minute' not in col]

    # Do checks on columns
    stop_if_not(utils.check_identical_excl_string(head_columns, body_columns, 'head', 'body'))
    stop_if_not(utils.check_identical_excl_string(head_columns, leg_columns, 'head', 'leg'))

    # Calculate ratios of attempts on body and leg
    head_body_ratios = pd.DataFrame(
        utils.safe_divide_np(merged_data[head_columns].to_numpy(), merged_data[body_columns].to_numpy(), zero_zero_fill = np.nan),
        columns = [col.replace('head_attempt', 'head_body_attempt_ratio') for col in head_columns])

    merged_data = pd.concat([merged_data, head_body_ratios], axis = 1)

    head_leg_ratios = pd.DataFrame(
        utils.safe_divide_np(merged_data[head_columns].to_numpy(), merged_data[leg_columns].to_numpy(), zero_zero_fill = np.nan),
        columns = [col.replace('head_attempt', 'head_leg_attempt_ratio') for col in head_columns])

    merged_data = pd.concat([merged_data, head_leg_ratios], axis = 1)

    del(head_body_ratios, head_leg_ratios, head_columns, body_columns, leg_columns)

    # Do a check to see if row number has stayed the same
    stop_if_not(len(merged_data) == merged_original_rows)

    # Calculate ratio of striking to takedowns by looking at distance striking compared with td attempts, and distance compared with ground and pound
    for fighter in ['fighter1', 'fighter2']:
        
        merged_data[fighter + '_hist_total_distance_strikes_over_td_attempts'] = utils.safe_divide_np(
            merged_data[fighter + '_hist_allrounds_distance_attempt'].to_numpy(), 
            merged_data[fighter + '_hist_allrounds_td_attempt'].to_numpy(), 
            zero_zero_fill = np.nan)
        
        merged_data[fighter + '_hist_total_distance_over_ground_strikes'] = utils.safe_divide_np(
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
        merged_data[fighter + '_sig.str.per_sub_attempt'] = merged_data[fighter + '_hist_allrounds_sig.str._attempt'] / merged_data[fighter + '_hist_allrounds_sub_attempt']
        # Knockdowns per strike landed
        merged_data[fighter + '_kd_per_sig.str._landed'] = merged_data[fighter + '_hist_allrounds_kd'] / merged_data[fighter + '_hist_allrounds_sig.str._landed']
        # Sub attempt per td
        merged_data[fighter + '_sub_attempt.per_td_landed'] = merged_data[fighter + '_hist_allrounds_sub_attempt'] / merged_data[fighter + '_hist_allrounds_td_landed']
            
        # Subs per attempt
        merged_data[fighter + '_subs_per_attempt'] = merged_data[fighter + '_hist_wins_Submission'] / merged_data[fighter + '_hist_allrounds_sub_attempt']
        
        # Recode Infs as missing, because could give misleading results
        new_columns = [fighter + '_sig.str._per_td_attempt', fighter + '_sig.str.per_sub_attempt', fighter + '_kd_per_sig.str._landed', fighter + '_sub_attempt.per_td_landed', fighter + '_subs_per_attempt']
        
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
    
    merged_data = features.add_fighter_differentials(merged_data)
    
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

    # Add on odds to be used for base
    merged_data['notformodel_wl_base_margin'] = merged_data['fighter1_' + data_processing_config['odds_suffix_for_base_margin']].copy()
    merged_data['notformodel_lw_base_margin'] = merged_data['fighter2_' + data_processing_config['odds_suffix_for_base_margin']].copy()
    

    ###########################################################################################
    # FLAG TRAIN/TEST/OOT
    # pdb.set_trace()
    np.random.seed(seed = config['seed_all'] + 715)
    merged_data = dataset.flag_train_test_oot(merged_data, config, test_fighter_list)



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
        cleaning.log_final_data_stats(merged_data, config['final_data_stats_log'], config)
        
        # Save the final data
        # split_by_year_and_save_fastparquet(merged_data, 'date', './data/final_dataset/', compression='ZSTD')
        fp.write('./data/final_dataset/final_dataset.parquet', merged_data, compression = 'ZSTD')
        
        # Output a single fight for checking purposes
        merged_data[(merged_data['fighter1'] == 'khabibnurmagomedov') & (merged_data['fighter2'] == 'conormcgregor')].T.to_csv('./data/final_dataset_checking.csv', index=True)
        
        merged_data[(merged_data['fighter1'] == 'antoniosilva') | (merged_data['fighter2'] == 'antoniosilva')].T.to_csv('./data/final_dataset_fighter_checking.csv')
    
    return merged_data

# This function predicts future UFC events

def predict_future_events(
        skip_data_update,
        skip_full_data_update,
        skip_save_data,
        skip_betfair_history,
        config,
        data_processing_config,
        model_hyper_params,
        betfair_login_config,
        test_fighters,
        output_path = None,
        skip_data_scrape = False
        ):
    
    """
    This function predicts all the future UFC events. It first scrapes the latest data, then if required, reprocesses
    the data to get the latest forecasting dataset, and then forecasts the upcoming fights based on the standard data filters
    
    Parameters
    ----------
    skip_data_update: bool
        whether to skip updating the data and go straight to predictions
    
    skip_full_data_update: bool
        whether to skip re-running the processing script on cleaned data. If set to False, only the odds will be updated
        
    skip_save_data: bool
        set whether the saving of processed data should be skipped i.e. only used for the predictions
        
    skip_betfair_history: bool
        set whether to skip downloading the latest betfair odds history data
    
    output_path: str
        the path where the outputs from the predictions will be saved. This is in addition to automatic outputs, which will be saved in ./results/
    
    """
    # pdb.set_trace()
    # Set the odds to use as upcoming instead of as used in the regular data
    data_processing_config['odds_suffix_for_true_odds'] = data_processing_config['upcoming_odds_suffix_for_true_odds']
    model_hyper_params['odds_to_use'] = data_processing_config['upcoming_odds_suffix_for_true_odds']

    ##########################################################################################
    ### Update the upcoming event and odds data ###
    ##########################################################################################

    if not skip_data_update:
        
        if not skip_data_scrape:
            # Run these scripts to scrape then clean the raw data.
            # Pass skip_fighter_tott_update so the slow full-fighter re-scrape is
            # skipped whenever we are also skipping the full dataset rebuild.
            scrape_all_unparsed_data(
                skip_betfair_history,
                config,
                betfair_login_config,
                skip_fighter_tott_update = skip_full_data_update
            )
            clean_all_raw_data_and_save(config)
        
            gc.collect();
        
        # Optional, can either rerun all the processing, or just update the odds. By default will rerun all processing if new events were found to have completed
        if skip_full_data_update:
            
            # Clear out memory
            # globals().clear()
            # # Read in global objects and packages
            # exec(open('global.py').read())
            # gc.collect();
            
            # # Make sure for the backtesting odds we use whatever upcoming because they're used to assess the model predictions later
            # model_hyper_params['odds_to_use'] = data_processing_config['upcoming_odds_suffix_for_true_odds']
            
            forecasting_data = odds.update_odds_on_merged_data(
                    cleaned_fight_odds_data_path = './data/' + config['fighter_odds_file_name'].replace('.parquet', '_clean.parquet'),
                    cleaned_event_odds_data_path = './data/' + config['event_odds_file_name'].replace('.parquet', '_clean.parquet'),
                    cleaned_betfair_odds_data_path = './data/' + config['betfair_upcoming_odds_file_name'].replace('.parquet', '_clean.parquet'),
                    # events_path = './data/' + config['event_details_file_name'].replace('.parquet', '_clean.parquet'),
                    final_data_folder = config['final_data_path'],
                    average_odds_model_path = './results/average_odds_model.joblib',
                    config = config,
                    data_processing_config = data_processing_config,
                    upcoming_only = True
                    )
        else:
            # Refresh the dataset
            forecasting_data = merge_and_process_all_clean_data(
                    config,
                    elo_config,
                    data_processing_config,
                    odds_scrape_config,
                    model_hyper_params,
                    test_fighter_list = test_fighters,
                    skip_save_data = skip_save_data # whether to skip the save data and just return merged data
                    )
            
    else:
        forecasting_data = fp.ParquetFile('./data/final_dataset/final_dataset.parquet').to_pandas() # pq.ParquetDataset(config['final_data_path']).read().to_pandas()
        # add in only odds update
     
    gc.collect();
    # pdb.set_trace()
            
    # Apply standard filters
    forecasting_data = cleaning.standard_data_filter(forecasting_data, include_upcoming = True, keep_rows_with_flag = True)

    ##########################################################################################
    ### Choose which events to predict ###
    ##########################################################################################

    upcoming_fights = forecasting_data[forecasting_data['outcome'].isna()]
    # upcoming_fights = forecasting_data[forecasting_data['event'] == 'ufc311:makhachevvs.moicano']

    upcoming_fights_for_model = upcoming_fights[upcoming_fights['notformodel_exclusion_reason'] == ''].copy()

    ##########################################################################################
    ### Get the models predictions ###
    ##########################################################################################

    # Get the data ready for the model
    X_upcoming, temp1, y_upcoming, temp2, temp3, temp4, temp5 = dataset.prep_data_for_model_training(
        upcoming_fights_for_model,
        upcoming_fights = True)

    # Load files
    # rf_xgb_model = mlflow.pyfunc.load_model('models:/xgb_forest_model/2')
    rf_xgb_model = joblib.load(bet_config['model_to_use'])

    # Get the model predictions
    # prediction_probabilities, temp = rf_xgb_model.predict(X_upcoming)
    prediction_probabilities, temp = modelling.predict_xgboost_ensemble(rf_xgb_model, X_upcoming, [0] * len(y_upcoming), model_hyper_params)

    # Add formatted fighters
    upcoming_fights_for_model = dataset.add_formatted_fighter_names(upcoming_fights_for_model, on = ['fighter1', 'fighter2'], formatted_col_names = ['fighter1', 'fighter2'], config = config)
    upcoming_fights = dataset.add_formatted_fighter_names(upcoming_fights, on = ['fighter1', 'fighter2'], formatted_col_names = ['fighter1', 'fighter2'], config = config)

    gc.collect();
    # pdb.set_trace()
    # Summarise the model predictions and betting decisions using the config
    betting_data = evaluation.summarise_model_predictions(
        actuals_data_df = upcoming_fights_for_model, 
        prediction_probabilities = prediction_probabilities, 
        min_prob_to_bet = bet_config['min_prob_to_bet'], 
        min_ev_to_bet = bet_config['min_ev_to_bet'], 
        min_bet_size = bet_config['min_bet_size'], 
        max_bet_size = bet_config['max_bet_size'],
        exchange_commission = bet_config['exchange_commission'],
        target_ev_after_cutoff = bet_config['min_ev_to_bet'])

    # Add the final bet size based on balance
    if bet_config['constant_bets'] == False:
        betting_data['Final_bet'] = np.round((betting_data['bets'] * betting_data['bet_size'] * bet_config['start_balance']) / 5) * 5
    else:
        betting_data['Final_bet'] = np.round((betting_data['bets'] * betting_data['bet_size']) / 5) * 5

    betting_data['Potential_win'] = np.round(betting_data['bets'] * betting_data['Final_bet'] * np.where(betting_data['Bet_trigger'] == betting_data['fighter1'], betting_data['fighter1_backtesting_odds'] - 1, betting_data['fighter2_backtesting_odds'] - 1), 2)

    betting_data['bet_assumed_odds'] = betting_data['bets'] * np.where(betting_data['Bet_trigger'] == betting_data['fighter1'], betting_data['fighter1_backtesting_odds'], betting_data['fighter2_backtesting_odds'])

    # Round odds columns
    for col in [cols for cols in betting_data.columns if 'odds' in cols and 'datetime' not in cols]:
        betting_data.loc[:, col] = np.round(betting_data[col], 2)

    betting_data.rename(columns = {'fighter1_notformodel_true_odds_asat_datetime': 'f1_scrape_date', 
                                   'fighter2_notformodel_true_odds_asat_datetime': 'f2_scrape_date',
                                   'Bet_target_exchange_odds': 'Bet_min_exchange_odds'}, inplace = True)

    # Flag whether odds Betfair imputed
    

    summary_data = betting_data[['date', 'event', 'f1_scrape_date', 'f2_scrape_date', 'fighter1', 'fighter1_backtesting_odds', 'Fighter1_win_prob', 'Fighter1_win_EV', 'fighter2', 'fighter2_backtesting_odds', 'Fighter2_win_prob', 'Fighter2_win_EV', 'Bet_trigger', 'bet_assumed_odds', 'Bet_min_exchange_odds', 'Final_bet', 'Potential_win']]

    # Add the fights that haven't been modelled along with exclusion reason for info
    summary_data = pd.concat([
         summary_data,
         upcoming_fights[upcoming_fights['notformodel_exclusion_reason'] != ''][['date', 'event', 'fighter1', 'fighter2', 'notformodel_exclusion_reason']]
    ])

    print(
          summary_data.to_string()
    )

    # Get only the bets to place
    bets_to_place = betting_data[(betting_data['bets'] == 1) & (betting_data['date'] == betting_data['date'].min())].copy()
    bets_to_place['bet_win_prob'] = np.where(bets_to_place['Bet_trigger'] == bets_to_place['fighter1'], bets_to_place['Fighter1_win_prob'], bets_to_place['Fighter2_win_prob'])

    try:
        del round
    except:
        pass

    for i in range(0, len(bets_to_place)):
        print("")
        print(f"Bet: {bets_to_place.iloc[i]['Bet_trigger']}")
        print(f"Win probability: {bets_to_place.iloc[i]['bet_win_prob']:.1f}%")
        print(f"EV: {bets_to_place.iloc[i]['bet_ev']:.2f} at {bets_to_place.iloc[i]['bet_assumed_odds']:.2f} odds")
        print(f"Min exchange odds (EV >= {bet_config['min_ev_to_bet']:.2f}): {bets_to_place.iloc[i]['Bet_min_exchange_odds']:.2f}")

    # Output the results
    summary_data.to_csv('./results/upcoming_predictions.csv')

    # If on Raspberry Pi, output to Gdrive
    if not output_path == None:
        
        summary_data.to_csv(f'{output_path}/upcoming_predictions_{datetime.today()}.csv')
        
        # Create text file with the bet summaries
        text_file_path = f"{output_path}/bets_to_place.txt"
        
        with open(text_file_path, "w") as file:
            for i in range(0, len(bets_to_place)):
                print("", file = file)
                print(f"Bet: {bets_to_place.iloc[i]['Bet_trigger']}", file = file)
                print(f"Win probability: {bets_to_place.iloc[i]['bet_win_prob']:.1f}%", file = file)
                print(f"EV: {bets_to_place.iloc[i]['bet_ev']:.2f} at {bets_to_place.iloc[i]['bet_assumed_odds']:.2f} odds", file = file)
                print(f"Min exchange odds (EV >= {bet_config['min_ev_to_bet']:.2f}): {bets_to_place.iloc[i]['Bet_min_exchange_odds']:.2f}", file = file)

    return summary_data

# This function scrapes event details from UFCStats.com given a list of urls to events

def scrape_all_unparsed_data(skip_betfair_history, config, betfair_login_config, skip_fighter_tott_update = False):
    """
    Scrapes all raw data from online given the config, saving the results

    Args:
        skip_betfair_history (bool): Whether to skip downloading and updating the betfair historic odds
        skip_fighter_tott_update (bool): Whether to skip re-scraping the full fighter details and tott data.
            When True, the existing saved parquets are used as-is, saving significant runtime.

    Returns:
        Only a success message. Data is saved according to the config
    """

    ### check if there are any unparsed events ###
    # pdb.set_trace()
    # read existing event details
    parsed_event_details_df = fp.ParquetFile('./data/' + config['event_details_file_name']).to_pandas()
    # get list of parsed event names
    list_of_parsed_events = list(parsed_event_details_df['EVENT'])

    # get soup
    soup = get_website_soup(config['completed_events_all_url'], fast = True, chrome_binary = config['chrome_binary_location'], time_delay = True)
        
    # parse event details
    new_event_details_df = parsing.parse_event_details(soup)
    # get list of all event names
    list_of_all_events = list(new_event_details_df['EVENT'])

    # find list event names that have not been parsed
    list_of_unparsed_events = [event for event in list_of_all_events if event not in list_of_parsed_events]



    ###############################################################################
    ##### Parse upcoming fight data #####
    ###############################################################################

    # Get a list of upcoming events
    upcoming_events_df = parsing.parse_event_details(
        get_website_soup(config['upcoming_events_url'], fast = True, chrome_binary = config['chrome_binary_location'], time_delay = True),
        upcoming = True)

    # # If missed the event start time because it is today, move it to the upcoming data
    if pd.to_datetime(new_event_details_df.iloc[0]['DATE']) == pd.to_datetime(datetime.today().strftime('%Y-%m-%d')):
        upcoming_events_df = pd.concat([new_event_details_df.iloc[[0]], upcoming_events_df], axis = 0, ignore_index = True).reset_index(drop = True)
        new_event_details_df = new_event_details_df.iloc[1:, :]
        list_of_unparsed_events = []

    # Only look at the first two to save time
    upcoming_events_df = upcoming_events_df.iloc[0:1]

    # Parse upcoming fight details
    upcoming_fight_details_df = parsing.update_fight_details_data(
        existing_fights_data = pd.DataFrame(),
        new_events_url_list = upcoming_events_df['URL'].to_list(),
        config = config,
        num_threads = config['num_threads'])



    ###############################################################################
    ### parse all missing events ###
    ###############################################################################

    # check if there are any unparsed events
    unparsed_events = False
    # if list_of_unparsed_events is empty then all available events have been parsed
    if not list_of_unparsed_events:
        print('All available events have been parsed.')
    else:
        # set unparsed_events to true
        unparsed_events = True
        # show list of unparsed events
        print(list_of_unparsed_events)

    # if unparsed_events = True
    # the code below continues to run to parse all missing events
    # new data is added to existing data and is written to file

    # Read existing data files
    parsed_fight_details_df = fp.ParquetFile('./data/' + config['fight_details_file_name']).to_pandas()
    parsed_fight_results_df = fp.ParquetFile('./data/' + config['fight_results_file_name']).to_pandas()
    parsed_fight_stats_df = fp.ParquetFile('./data/' + config['fight_stats_file_name']).to_pandas()
    parsed_odds_df = fp.ParquetFile('./data/' + config['fighter_odds_file_name']).to_pandas()
    parsed_event_odds_urls = fp.ParquetFile('./data/event_odds_urls.parquet').to_pandas()
    parsed_event_odds_df = fp.ParquetFile('./data/' + config['event_odds_file_name']).to_pandas()
    parsed_betfair_odds_df = fp.ParquetFile('./data/' + config['betfair_odds_file_name']).to_pandas()

    if unparsed_events == True:
        
        ##### Add new events to events data #####
        updated_event_details_df = pd.concat([parsed_event_details_df, new_event_details_df[new_event_details_df['EVENT'].isin(list_of_unparsed_events)]])
        

        ##### Parse fight details #####

        # define list of urls of missing fights to parse
        list_of_unparsed_events_urls =  list(updated_event_details_df['URL'].loc[(updated_event_details_df['EVENT'].isin(list_of_unparsed_events))])

        # Create new data frame including the new events
        new_parsed_fight_details_df = parsing.update_fight_details_data(
            existing_fights_data = parsed_fight_details_df,
            new_events_url_list = list_of_unparsed_events_urls,
            config = config,
            num_threads = config['num_threads'])

        ##### Parse fight results and fight stats #####
        
        # define list of urls of fights to parse
        list_of_unparsed_fight_details_urls = list(new_parsed_fight_details_df[new_parsed_fight_details_df['EVENT'].isin(list_of_unparsed_events)]['URL'])

        new_parsed_fight_results_df, new_parsed_fight_stats_df = parsing.update_fight_results_and_stats_data(
            parsed_fight_results_df, 
            parsed_fight_stats_df, 
            list_of_unparsed_fight_details_urls, 
            config, 
            num_threads = config['num_threads']
            )

        # Run a fix on the bouts in the details data
        new_parsed_fight_details_df = parsing.fix_fight_details_bout(new_parsed_fight_details_df, new_parsed_fight_stats_df)


    ###############################################################################
    ##### Parse new fighter odds data #####
    ###############################################################################

    # define list of urls of fighters we want new odds data for
    try:
        list_of_unparsed_fighter_odds = list(set(new_parsed_fight_stats_df[new_parsed_fight_stats_df['EVENT'].isin(list_of_unparsed_events)]['FIGHTER'].to_list()))
        list_of_unparsed_event_dates = list(updated_event_details_df[updated_event_details_df['EVENT'].isin(list_of_unparsed_events)]['DATE']) + upcoming_events_df['DATE'].to_list()
    except:
        list_of_unparsed_fighter_odds = []
        list_of_unparsed_event_dates = []

    # Add on the upcoming events too
    list_of_unparsed_fighter_odds = list(set(list_of_unparsed_fighter_odds + dataset.fighter_list_from_bouts(upcoming_fight_details_df['BOUT'])))

    list_of_unparsed_event_dates = list_of_unparsed_event_dates + upcoming_events_df['DATE'].to_list()

    # Convert the dates
    list_of_unparsed_event_dates = [datetime.strptime(x, '%B %d, %Y') for x in list_of_unparsed_event_dates]

    new_parsed_odds_df, new_parsed_event_odds_urls, new_parsed_event_odds_df = odds_scraping.update_odds_data(
        parsed_odds_df, 
        parsed_event_odds_urls, 
        parsed_event_odds_df,
        list_of_unparsed_fighter_odds, 
        list_of_unparsed_event_dates, 
        config, 
        fast_scrape = odds_scrape_config['fast_scrape_update'],
        num_threads = config['num_threads'])
        
    # # Find which fighters didn't get odds properly by looking at the dates of people found
    # # Offsets of -1, 0, and +1 days
    # offsets = [timedelta(days=delta) for delta in [-1, 0, 1]]
    # # List comprehension to apply all offsets to all dates
    # expanded_dates = sorted(set(date + offset for date in list_of_unparsed_event_dates for offset in offsets))
    # found_fighters = new_parsed_odds_df[new_parsed_odds_df['date'].isin(expanded_dates)]['fighter'].tolist()
    # missing_fighters = [fighter for fighter in list_of_unparsed_fighter_odds if fighter not in found_fighters]

    # # Scrape the odds again but a slow scrape
    # if len(missing_fighters) > 0:
    #     new_parsed_odds_df, new_parsed_event_odds_urls, new_parsed_event_odds_df = update_odds_data(
    #         new_parsed_odds_df, 
    #         new_parsed_event_odds_urls, 
    #         new_parsed_event_odds_df,
    #         missing_fighters, 
    #         list_of_unparsed_event_dates, 
    #         config, 
    #         fast_scrape = False,
    #         num_threads = config['num_threads'])

    ###############################################################################
    ##### Update the Betfair odds #####
    ###############################################################################

    if not skip_betfair_history:
        # Update the odds
        new_parsed_betfair_odds_df = betfair.update_betfair_odds(
            parsed_betfair_odds_df, 
            betfair_login_config, 
            download_folder = betfair_data_config['download_folder'], 
            decompressed_output_dir = betfair_data_config['decompressed_output_dir'], 
            decompressed_temp_output_dir = betfair_data_config['decompressed_temp_output_dir']
        )
    else:
        new_parsed_betfair_odds_df = parsed_betfair_odds_df

    time.sleep(5)

    # Get the upcoming Betfair odds data
    upcoming_betfair_odds = betfair.betfair_get_all_upcoming_odds(betfair_login_config, betfair_data_config)

    ###############################################################################
    ##### Parse new fighter data #####
    ###############################################################################

    # Load the saved fighter details and tott datasets from disk — always needed downstream
    parsed_fighter_details_df = fp.ParquetFile('./data/' + config['fighter_details_file_name']).to_pandas()
    parsed_fighter_tott_df = fp.ParquetFile('./data/' + config['fighter_tott_file_name']).to_pandas()

    if skip_fighter_tott_update:

        # Skip the full re-scrape and use the saved datasets as-is
        all_fighter_details_df = parsed_fighter_details_df
        new_parsed_fighter_tott_df = parsed_fighter_tott_df

    else:

        # Scrape the full list of fighters from the UFC site again
        # generate list of urls for fighter details
        list_of_alphabetical_urls = scraping.generate_alphabetical_urls()
        
        # Go through list of alphabetical urls
        all_fighter_details_df = scraping.parallel_scrape(
            url_list = list_of_alphabetical_urls,
            parse_function = parsing.parse_fighter_details,
            num_threads = config['num_threads'],
            fighter_details_column_names = config['fighter_details_column_names'])
        
        # Clean out any 'www' to ensure we find all the required URLs
        all_fighter_details_df['URL'] = all_fighter_details_df['URL'].str.replace('www,', '')
        parsed_fighter_tott_df['URL'] = parsed_fighter_tott_df['URL'].str.replace('www,', '')
        
        # Find the unparsed urls
        unparsed_fighter_urls = all_fighter_details_df['URL'][~all_fighter_details_df['URL'].isin(parsed_fighter_tott_df['URL'].to_list())].to_list()
        # pdb.set_trace()
        # Now parse the unparsed URLs
        if len(unparsed_fighter_urls) > 0:
            
            # Concatenate the old and new scraped data
            new_parsed_fighter_tott_df = pd.concat([
                scraping.parallel_scrape(
                    url_list = unparsed_fighter_urls,
                    # url_list = all_fighter_details_df['URL'].tolist(),
                    parse_function = odds_scraping.get_fighter_data,
                    num_threads = config['num_threads'],
                    fighter_tott_column_names = config['fighter_tott_column_names']),
                parsed_fighter_tott_df
            ])

        else:
            new_parsed_fighter_tott_df = parsed_fighter_tott_df
        

    ###############################################################################
    ##### Data consistency checks #####
    ###############################################################################

    if unparsed_events:
        
        ## Run checks to ensure each of the datasets are consistent and cover the same events
        checks_passed = cleaning.check_data_coverage(
            updated_event_details_df, 
            new_parsed_fight_details_df, 
            new_parsed_fight_results_df, 
            new_parsed_fight_stats_df, 
            new_parsed_odds_df,
            all_fighter_details_df,
            new_parsed_fighter_tott_df,
            new_parsed_event_odds_df,
            new_parsed_betfair_odds_df,
            upcoming_events_df,
            upcoming_fight_details_df,
            upcoming_betfair_odds,
            config)
        
        # Check new rows have been added
        stop_if_not(
            (len(updated_event_details_df) >= len(parsed_event_details_df)) & (
                len(new_parsed_fight_details_df) >= len(parsed_fight_details_df)) & (
                    len(new_parsed_fight_results_df) >= len(parsed_fight_results_df)) & (
                        len(new_parsed_fight_stats_df) >= len(parsed_fight_stats_df)) & (
                            len(new_parsed_odds_df) >= len(parsed_odds_df)) & (
                                len(new_parsed_fighter_tott_df) >= len(parsed_fighter_tott_df)) & (
                                    len(new_parsed_event_odds_df) >= len(parsed_event_odds_df)),
                                'Row change checks')
        
        # AB: If the fighter_tott has missing info, just run the whole thing rather than update old one
        
        if checks_passed:
            
            # Log the aspects of these files
            data_summary = cleaning.update_data_summary(
                results_csv_path = './data/raw_data_summary.csv', 
                data_frames = {
                    'updated_event_details_df': updated_event_details_df, 
                    'parsed_fight_details_df': new_parsed_fight_details_df, 
                    'parsed_fight_results_df': new_parsed_fight_results_df, 
                    'parsed_odds_df': new_parsed_odds_df, 
                    'parsed_event_odds_df' : new_parsed_event_odds_df,
                    'parsed_fighter_tott_df': new_parsed_fighter_tott_df,
                    'parsed_fight_stats_df': new_parsed_fight_stats_df,
                    'parsed_betfair_odds_df': new_parsed_betfair_odds_df},
                summary_date = datetime.today(),
                save_output = True)
            
            # Manually check if you're happy with the above then save
            
            # write new fight details to file
            fp.write('./data/' + config['event_details_file_name'], updated_event_details_df, compression = 'ZSTD')
            fp.write('./data/' + config['upcoming_event_details_file_name'], upcoming_events_df, compression = 'ZSTD')
            fp.write('./data/' + config['fight_details_file_name'], new_parsed_fight_details_df, compression = 'ZSTD')
            fp.write('./data/' + config['upcoming_fight_details_file_name'], upcoming_fight_details_df, compression = 'ZSTD')
            fp.write('./data/' + config['fight_results_file_name'], new_parsed_fight_results_df, compression = 'ZSTD')
            fp.write('./data/' + config['fight_stats_file_name'], new_parsed_fight_stats_df, compression = 'ZSTD')
            fp.write('./data/' + config['fighter_odds_file_name'], new_parsed_odds_df, compression = 'ZSTD')
            fp.write('./data/event_odds_urls.parquet', new_parsed_event_odds_urls, compression = 'ZSTD')
            fp.write('./data/' + config['event_odds_file_name'], new_parsed_event_odds_df, compression = 'ZSTD')
            
            fp.write('./data/' + config['betfair_odds_file_name'], new_parsed_betfair_odds_df, compression = 'ZSTD')
            fp.write('./data/' + config['betfair_upcoming_odds_file_name'], upcoming_betfair_odds, compression = 'ZSTD')
            
            fp.write('./data/' + config['fighter_details_file_name'], all_fighter_details_df, compression = 'ZSTD')
            fp.write('./data/' + config['fighter_tott_file_name'], new_parsed_fighter_tott_df, compression = 'ZSTD')
            
        else:
            print("Data checks not passed. Data not saved. See previous output messages for more information.")
            raise SystemExit

    else:

        ## Run checks to ensure each of the datasets are consistent and cover the same events
        checks_passed = cleaning.check_data_coverage(
            parsed_event_details_df, 
            parsed_fight_details_df, 
            parsed_fight_results_df, 
            parsed_fight_stats_df, 
            new_parsed_odds_df,
            all_fighter_details_df,
            new_parsed_fighter_tott_df,
            new_parsed_event_odds_df,
            new_parsed_betfair_odds_df,
            upcoming_events_df,
            upcoming_fight_details_df,
            upcoming_betfair_odds,
            config)
            
        #######################
        ## Save files if all has run successfully
        #######################
        
        if checks_passed:
            
            # write new fight details to file
            fp.write('./data/' + config['upcoming_event_details_file_name'], upcoming_events_df, compression = 'ZSTD')
            fp.write('./data/' + config['upcoming_fight_details_file_name'], upcoming_fight_details_df, compression = 'ZSTD')
            
            fp.write('./data/' + config['betfair_odds_file_name'], new_parsed_betfair_odds_df, compression = 'ZSTD')
            fp.write('./data/' + config['betfair_upcoming_odds_file_name'], upcoming_betfair_odds, compression = 'ZSTD')

            fp.write('./data/' + config['fighter_odds_file_name'], new_parsed_odds_df, compression = 'ZSTD')
            fp.write('./data/event_odds_urls.parquet', new_parsed_event_odds_urls, compression = 'ZSTD')
            fp.write('./data/' + config['event_odds_file_name'], new_parsed_event_odds_df, compression = 'ZSTD')
            
        else:
            print("Data checks not passed. Data not saved. See previous output messages for more information.")
            raise SystemExit

    return "Raw data scrape complete."

def backtest_model(
        actuals_data, 
        prediction_probabilities, 
        breakdown_columns, 
        min_prob_to_bet, 
        min_ev_to_bet, 
        min_bet_size = 1, 
        max_bet_size = 1, 
        constant_bets = True, 
        start_balance = 1000,
        point_in_time_only_odds = True,
        target_ev_after_cutoff = 0,
        output_data_for_debug = True, ):
    """
    
    A function to backtest a model's predictions against actuals data that contain the odds'.
    
    This function assume constant betting amounts rather than fractions of a pot. This means it's easier to track the consistency over time and not be affected by early wins and losses affecting the overall metrics from a test'

    Parameters
    ----------
    actuals_data : data frame
        DESCRIPTION.
    prediction_probabilities : array
        DESCRIPTION.
    breakdown_cols : TYPE
        DESCRIPTION.
    min_prob_to_bet : float64
        the minimum probability of a fighter winning required to place the bet
    min_ev_to_bet : float64
        the minimum expected return required to place the bet (EV of 0 means no profit expected), and it is displayed in terms of betting units units
    min_bet_size : float64
        the minimum bet size in units. This should be left as 1 and only max varied unless wanting to model £ bets
    max_bet_size : float64
        the maximum bet size. Set this to 1 to give uniform betting regardless of the probability
    point_in_time_only_odds : bool
        A True or False flag to say whether we only use the backtesting odds to make bets. If False, it will also use the best Betfair odds after cutoff and make bets based on the min EV
    target_ev_after_cutoff: float64
        The minimum EV that the test will target for odds after the cutoff time only

    Returns
    -------
    None.

    """
    
    # ###
    # # Debug
    # actuals_data = X_train[['id']].merge(data, on = 'id', how = 'left')
    # prediction_probabilities, temp = predict_xgboost_ensemble(rf_xgb_model, X_train, y_train)
    # breakdown_cols = ['weightclass', 'year']
    # min_prob_to_bet = 0.5
    # min_ev_to_bet = 0.1
    # ###
    
    # pdb.set_trace()
    
    breakdown_cols = ['bet_ev_band', 'Predict_to_win', 'Fighter_to_win'] + breakdown_columns.copy()
    
    # Use the new function to summarise the model predictions and bet triggers
    actuals_data, wl_trigger, lw_trigger = evaluation.summarise_model_predictions(actuals_data, prediction_probabilities, min_prob_to_bet, min_ev_to_bet, min_bet_size, max_bet_size, return_triggers = True, point_in_time_only_odds = point_in_time_only_odds, target_ev_after_cutoff = target_ev_after_cutoff)
    
    # Record the odds that are to be used in backtesting. These may be edited next
    fighter1_backtesting_odds = actuals_data['fighter1_backtesting_odds'].copy()
    fighter2_backtesting_odds = actuals_data['fighter2_backtesting_odds'].copy()
    
    # Find any bets that occur after the cutoff if necessary by using target odds
    if not point_in_time_only_odds:
        
        breakdown_cols = ['bet_triggered_after_cutoff'] + breakdown_cols
        
        # Check where the odds meet the requirement and a bet isn't triggered at the cutoff point
        additional_wl_trigger = np.where((wl_trigger == 0) & (
            actuals_data['fighter1_notformodel_best_betfair_odds_after_cutoff'] >=
            actuals_data['fighter1_target_exchange_odds']) & (
            actuals_data['Fighter1_win_prob'] >= min_prob_to_bet),
                1, 0)
                
        additional_lw_trigger = np.where((lw_trigger == 0) & (
            actuals_data['fighter2_notformodel_best_betfair_odds_after_cutoff'] >=
            actuals_data['fighter2_target_exchange_odds']) & (
            actuals_data['Fighter2_win_prob'] >= min_prob_to_bet),
                1, 0)
        
        # Deal with double triggers
        all_triggers = additional_wl_trigger + additional_lw_trigger
        additional_wl_trigger = np.where(all_triggers == 2, 0, additional_wl_trigger)
        additional_lw_trigger = np.where(all_triggers == 2, 0, additional_lw_trigger)
                
        # Now add these triggers and the odds to the existing vectors
        fighter1_backtesting_odds = np.where(additional_wl_trigger == 1, actuals_data['fighter1_target_exchange_odds'], fighter1_backtesting_odds)
        
        fighter2_backtesting_odds = np.where(additional_lw_trigger == 1, actuals_data['fighter2_target_exchange_odds'], fighter2_backtesting_odds)
        
        wl_trigger = np.where(additional_wl_trigger == 1, 1, wl_trigger)
        lw_trigger = np.where(additional_lw_trigger == 1, 1, lw_trigger)
        
        actuals_data['bet_triggered_after_cutoff'] = np.where(all_triggers == 1, 'Bet triggered after cutoff', 'Usual bet at cutoff')
        
        # Correct the bet count and amount
        actuals_data['bets'] = np.where(additional_wl_trigger == 1, 1, actuals_data['bets'])
        actuals_data['bet_size'] = np.where(additional_wl_trigger == 1, actuals_data['fighter1_target_bet_size'],
                                            np.where(additional_lw_trigger == 1, actuals_data['fighter2_target_bet_size'], actuals_data['bet_size']))
        
    else:
        actuals_data['bet_triggered_after_cutoff'] = 0
        
    # Do a check that the inputs are consistent if doing a non constant bets. If using non constant bets, the bet sizing should be as proportions, not £ amounts
    if not constant_bets:
        stop_if_not((min_bet_size < 1) and (max_bet_size < 1), "Bet size for non-constant bets check")
        
    # Now calculate the returns if a bet was made
    actuals_data['profit_loss'] = ((fighter1_backtesting_odds - 1) * wl_trigger * (actuals_data['outcome'] == 'W/L') - wl_trigger * (actuals_data['outcome'] != 'W/L') + (fighter2_backtesting_odds - 1) * lw_trigger * (actuals_data['outcome'] == 'L/W') - lw_trigger * (actuals_data['outcome'] != 'L/W')) * actuals_data['bet_size']
    
    if constant_bets:
        pass
    else:
        # If bets are not constant, we need to work out the cumulative returns
        
        # Save the profit and loss
        actuals_data['profit_loss_proportion'] = actuals_data['profit_loss'].copy()
        
        # First add 1 to get the proportion of the balance still left
        actuals_data['profit_loss_multiplier'] = 1 + actuals_data['profit_loss_proportion']
    
        # Then convert to a cumulative product after sorting by date
        actuals_data.sort_values('date', inplace = True)
        
        # Create balances
        actuals_data['end_balance'] = start_balance * actuals_data['profit_loss_multiplier'].cumprod()
        actuals_data['start_balance'] = actuals_data['end_balance'].shift(1).fillna(start_balance)
        
        # Then calculate profit and loss
        actuals_data['profit_loss'] = actuals_data['end_balance'] - actuals_data['start_balance']
        
    # Calculate some further columns that will be used for stats
    actuals_data['total'] = 'Total'
    actuals_data['fights'] = 1
    actuals_data['wins'] = 1 * (actuals_data['profit_loss'] > 0)
    actuals_data['max_bet_size'] = actuals_data['bet_size']
    
    # If not doing constant bets, this script isn't able to break down results by anything as it stands, so reset the breakdown cols
    if not constant_bets:
        breakdown_cols = []
    
    ## Summarise the results
    results = actuals_data.groupby('total', observed = False).agg({'fights': lambda x: len(x), 'bets': 'sum', 'wins': 'sum', 'bet_size': 'sum', 'profit_loss': 'sum', 'max_bet_size': 'max'})
    for col in breakdown_cols:
        results = pd.concat([
            results,
            actuals_data.groupby(col, observed = False).agg({'fights': lambda x: len(x), 'bets': 'sum', 'wins': 'sum', 'bet_size': 'sum', 'profit_loss': 'sum', 'max_bet_size': 'max'})
        ])
    
    # Calculate percentage info
    results['percent_bet_on'] = results['bets'] / results['fights'] * 100
    results['win_percent'] = results['wins'] / results['bets'] * 100
    
    results['max_drawdown'] = np.nan
    # For the total, calculate the longest streaks
    results['max_losing_streak'] = np.nan
    if constant_bets:
        results['roi'] = results['profit_loss'] / results['bet_size'] * 100 # the profit and loss assumes every bet equal size
        results.loc['Total', 'max_drawdown'] = utils.minimum_sum_series(actuals_data['profit_loss'])
        results.loc['Total', 'max_losing_streak'] = features.longest_streak(actuals_data['profit_loss'], -1)
    else:
        # Work out the ROI per fight, otherwise going to be very large
        results['roi'] = ((results['profit_loss'] / start_balance + 1) ** (1 / results['bets']) - 1) * 100 # the profit and loss assumes every bet equal size
        results.loc['Total', 'max_drawdown'] = utils.minimum_sum_series(actuals_data['profit_loss_proportion'])

    # # Add the row names as a column
    results = pd.concat([pd.DataFrame({'Group': results.index}, index = results.index), results], axis=1)
    # results.rename(columns={'index': 'RowName'}, inplace=True)
    
    # Get fight level output
    actuals_data.sort_values('date', inplace = True)
    
    if 'y_pred_prob' in actuals_data.columns:
        fight_output = actuals_data[['date', 'fighter1', 'Fighter1_win_prob', 'Fighter1_win_EV', 'fighter1_backtesting_odds', 'fighter2', 'Fighter2_win_prob', 'Fighter2_win_EV', 'fighter2_backtesting_odds', 'outcome', 'target', 'predictions', 'bets', 'bet_ev', 'profit_loss', 'bet_triggered_after_cutoff' ,'kelly_criterion', 'bet_size', 'max_bet_size', 'total', 'fights', 'wins', 'y_pred_prob']]
    else:
        fight_output = None
    
    # Output the underlying data if required
    if output_data_for_debug:
        fight_output.to_csv('results/backtesting_data_for_checking.csv')
    
    # Mpve column positions for better searching
    results = utils.move_column(results, 'profit_loss', 1)
    results = utils.move_column(results, 'roi', 2)
    
    return results, fight_output

# This function takes a model and a test dataset and computes test statistics then logs them in mlflow.

def btest_model_and_log_mlflow(actuals_data, prediction_probabilities, breakdown_cols, breakdown_metrics, mlflow_description, target_column = 'outcome', backtesting_params = {'min_prob_to_bet': 0.5, 'min_ev_to_bet': 0.05, 'max_bet_size': 1}, log_mlflow = True):

    """
    Tests a model by taking in some actuals data, a set of prediction probabilities, and any breakdown variables to use. It logs various metrics in MLflow as well as outputing some statistics

    Args:
        actuals_data (data frame): this is the data that the predictions are for. It should not contain dummy versions of columns to be uesd in breakdown cols
        prediction_probabilities (array): the probabilities for each class of the outcome
        breakdown_cols (list): a list of column names to create metric breakdowns for
        breakdown_metrics (list): the metrics to output for the breakdown. These names must align with metrics calculated in the calculate_metrics() function
        mlflow_description: a description to give for the run in MLflow
        log_mlflow (bool): set to False to skip the MLflow logging and just return the backtesting results

    Returns:
        None, but prints the breakdown outputs and logs everything in MLflow
    
    """
    
    # pdb.set_trace()
    
    ####
    # # Debug testing
    # actuals_data = data[data['id'].isin(X_train['id'].tolist())].copy()
    # X_data = X_train
    # y_data = y_train
    # prediction_probabilities, temp = predict_xgboost_ensemble(rf_xgb_model, X_train, y_train)
    # breakdown_cols = ['weightclass']
    # breakdown_metrics  = ['Avg WL only AUC', 'X Odds Avg WL only AUC', 'Hmean WL only Accuracy', 'X Odds Hmean WL only Accuracy']
    # mlflow_description = 'test'
    # target_column = 'outcome'
    ####

    # Get the class predictions from the probabilities
    actuals_data['predictions'] = np.argmax(prediction_probabilities, axis = 1)
    actuals_data['y_pred_prob'] = list(prediction_probabilities)
    
    # Categorise the outcome variable to align with prediction outputs
    actuals_data['target'] = cleaning.clean_target_column(actuals_data[target_column])
    
    if log_mlflow:
        # Create the MLflow description
        mlflow.set_tag("mlflow.note.content", mlflow_description)

        # Log the overall metrics in MLflow
        tracking.log_metrics_mlflow(prediction_probabilities, actuals_data['target'], actuals_data['fighter1_backtesting_odds'], actuals_data['fighter2_backtesting_odds'], classes = np.array([0, 1, 2]))

        # Log the overall odds metrics in MLflow
        tracking.log_metrics_odds_mlflow(actuals_data, actuals_data['target'])
    
    # # Log the parameters of the model
    # mlflow.log_params(model['params'])
    
    ## Now begin calculating the metrics for the various breakdowns
    
    # Initiate the results dictionary
    metrics_by_group = {}
    
    # Loop through the columns to be analysed
    for col in breakdown_cols:
        
        print('Calculating metrics for ' + col)
        
        # Initial the new dictionary for this column
        metrics_by_group.update({col: {}})
        # pdb.set_trace()
        # Convert the column to a string to ensure consistent outputs
        actuals_data[col] = actuals_data[col].apply(str)
        
        # Now loop through all the unique values within the column
        for group in np.unique(actuals_data[col]).tolist():
            
            # print(group)
            
            # Get the rows we're interested in
            row_filter = actuals_data[col] == group
            # print(row_filter.sum())
            
            # Use try except in case sample sizes too small
            try:
                # Calculate the metrics
                metrics = evaluation.calculate_metrics(prediction_probabilities[row_filter], actuals_data['target'][row_filter], actuals_data['fighter1_backtesting_odds'][row_filter], actuals_data['fighter1_backtesting_odds'][row_filter])
                
                # Add odds metrics
                metrics.update(evaluation.calculate_metrics_odds(actuals_data[row_filter], actuals_data['target'][row_filter]))
                
                # Add these metrics to the results dictionary
                metrics_by_group[col].update({group: metrics})
            except:
                print(f"Metrics for {col}: {group} couldn't be calculated.")
                continue
        # pdb.set_trace()
        # Get the outputs for each column and log them
        summary_df = evaluation.metric_table_from_metrics_dictionary(metrics_by_group, col)[['group'] + breakdown_metrics]
        
        # Sort it
        summary_df.sort_values('group', inplace = True)
        
        # Calculate any differences with odds if available
        for sub_col in summary_df.columns:
            if 'X Odds ' + sub_col in summary_df.columns:
                summary_df['Diff ' + sub_col] = summary_df[sub_col] - summary_df['X Odds ' + sub_col]
        
        print(summary_df)
        
        # Name the index
        summary_df.index.name = 'Group'

        if log_mlflow:
            # Save a temporary csv and log it as an artifact
            summary_df.to_csv('results/temp/summary.csv')
            mlflow.log_artifact('results/temp/summary.csv', col + ' metrics')

            # Create chart of logloss and EV MSE
            if 'Logloss' in breakdown_metrics and 'X Odds Logloss' in breakdown_metrics:
                fig, ax = evaluation.metric_chart(x = summary_df['group'], y_model = summary_df['Logloss'], y_odds = summary_df['X Odds Logloss'], x_label = 'Group', y_label = 'Logloss')
                mlflow.log_figure(fig, col +  " metrics/logloss_chart.png")

            if 'EV MSE' in breakdown_metrics and 'X Odds EV MSE' in breakdown_metrics:
                fig, ax = evaluation.metric_chart(x = summary_df['group'], y_model = summary_df['EV MSE'], y_odds = summary_df['X Odds EV MSE'], x_label = 'Group', y_label = 'EV MSE')
                mlflow.log_figure(fig, col +  " metrics/ev_mse_chart.png")

            if 'Hmean WL only Accuracy' in breakdown_metrics and 'X Odds Hmean WL only Accuracy' in breakdown_metrics:
                fig, ax = evaluation.metric_chart(x = summary_df['group'], y_model = summary_df['Hmean WL only Accuracy'], y_odds = summary_df['X Odds Hmean WL only Accuracy'], x_label = 'Group', y_label = 'Hmean WL only Accuracy')
                mlflow.log_figure(fig, col +  " metrics/accuracy_chart.png")
            
        # pdb.set_trace()
    
    # Finally, get the backtesting results
    backtesting, backtesting_fight_output = backtest_model(actuals_data, prediction_probabilities, breakdown_cols, **backtesting_params)
   
    # Summarise the ROI for EV bands
    roi_by_ev = evaluation.summarise_bet_roi_by_ev(backtesting_fight_output)
    
    if log_mlflow:
        # Create chart and log in MLflow
        fig = evaluation.plot_roi_and_sample_size(roi_by_ev)
        mlflow.log_figure(fig, "Backtesting results/ROI_by_EV.png")

        # Plot and log the calibration curve
        fig = evaluation.plot_calibration_curve(
            np.where(actuals_data['outcome'] == 'W/L', 1, 0),
            prediction_probabilities[:, 2] # Associated probabilities
        )
        mlflow.log_figure(fig, "Backtesting results/F1_Win_Calibration_Curve.png")

        # Plot and log the calibration curve
        fig = evaluation.plot_calibration_curve(
            np.where(actuals_data['outcome'] == 'L/W', 1, 0),
            prediction_probabilities[:, 0] # Associated probabilities
        )
        mlflow.log_figure(fig, "Backtesting results/F2_Win_Calibration_Curve.png")

        # Log the overall backtesting results
        backtesting.to_csv('results/temp/summary.csv', index = True)
        mlflow.log_artifact('results/temp/summary.csv', 'Backtesting results')
        backtesting_fight_output.to_csv('results/temp/fight_backtesting.csv', index = True)
        mlflow.log_artifact('results/temp/fight_backtesting.csv', 'Backtesting results')

        # Chart profit and loss
        cumulative_profit_loss = backtesting_fight_output[backtesting_fight_output['bets'] == 1]['profit_loss'].cumsum()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(pd.to_datetime(backtesting_fight_output[backtesting_fight_output['bets'] == 1]['date']).dt.date, cumulative_profit_loss)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Profit")
        plt.xticks(rotation=90)
        mlflow.log_figure(fig, "Backtesting results/Running_Cumulative_Profit_Loss.png")

        # Log the total metrics in MLflow
        mlflow.log_params(backtesting_params)

        mlflow.log_metric('Backtesting_P_and_L', backtesting.loc['Total', 'profit_loss'])
        mlflow.log_metric('Backtesting_Win_Percent', backtesting.loc['Total', 'win_percent'])
        mlflow.log_metric('Backtesting_Max_Drawdown', backtesting.loc['Total', 'max_drawdown'])
        mlflow.log_metric('Backtesting_ROI', backtesting.loc['Total', 'roi'])
        mlflow.log_metric('Backtesting_Bets', backtesting.loc['Total', 'bets'])
        mlflow.log_metric('Backtesting_Percent_Bet_On', backtesting.loc['Total', 'percent_bet_on'])
        mlflow.log_metric('Backtesting_Normalised_Total_Percent_Return', 100 * ((1 + backtesting.loc['Total', 'roi'] / 100) ** backtesting.loc['Total', 'bets'] - 1))
        mlflow.log_metric('Backtesting_Average_Bet_Size', backtesting.loc['Total', 'bet_size'] / backtesting.loc['Total', 'bets'])
        mlflow.log_metric('Backtesting_Actual_Max_Bet_Size', backtesting.loc['Total', 'max_bet_size'])

    return backtesting

def clean_all_raw_data(events, upcoming_events, upcoming_fight_details, fighters, fight_results, fight_stats, fighter_odds, event_odds, betfair_odds, betfair_upcoming_odds):
    
    """
    This function takes all the raw scraped data and cleans them, producing datasets which will then be saved.
    
    Returns: dict of cleaned datasets
    
    """
    # pdb.set_trace()
    # Convert all column names to lowercase
    for df in [events, upcoming_events, upcoming_fight_details, fight_results, fight_stats, fighters, fighter_odds, event_odds, betfair_odds, betfair_upcoming_odds]:
        df.columns = df.columns.str.lower()
        
    # Convert fighter data to numeric
    # Replace spaces and inches, then split and convert to cm
    fighters[['feet', 'inches']] = fighters['height'].str.replace(' ', '').str.replace('"', '').str.replace('-', '').str.split("'", expand = True)
    fighters.drop(columns = ['height'], inplace = True)
    fighters['height'] = pd.to_numeric(fighters['feet']) * 30.48 + pd.to_numeric(fighters['inches']) * 2.54
    
    fighters = utils.to_numeric(fighters, ['weight', 'reach'])
    fight_stats = utils.to_numeric(fight_stats, ['kd'])
    
    # Get cleaned versions of the columns
    fighters['Fighter'] = fighters['fighter'].copy()
    
    
    ##################
    ## Betfair initial processing
    
    # Remove any duplicates from the betfair data, de-prioritising Yes/No
    betfair_odds['yesno'] = np.where(betfair_odds['fighter'].str.replace('.', '').str.lower().isin(['yes', 'no']), 1, 0)    
    # # Convert Yes/No fighters into names on betfair odds # MANY WERE JUST DUPLICATES OF BOUTS WITH PROPER DATA
    # betfair_odds = betfair_yesno_to_names(betfair_odds)
    betfair_odds = betfair_odds[betfair_odds['yesno'] == 0]
    
    # Now sort and remove duplicates
    # betfair_odds.sort_values(['yesno'], inplace = True)
    betfair_odds = betfair_odds.drop_duplicates(subset = ['fighter', 'bout', 'event_date'])
    betfair_odds.drop(columns = ['yesno'], inplace = True)
    
    # Replace text in betfair odds bout to be consistent with other data
    betfair_odds['bout'] = betfair_odds['bout'].str.replace(" v ", " vs. ")
    
    # Get the surnames on the betfair odds data for use later, before cleaning the spaces from the names
    dataset.add_opponent_column(betfair_odds)
    betfair_odds['fighter_surname'] = betfair_odds['fighter'].str.split().str[-1]
    betfair_odds['opponent_surname'] = betfair_odds['opponent'].str.split().str[-1]
    
    ################################
    
    # Remove spaces in events to avoid merge errors
    for df in [events, upcoming_events, upcoming_fight_details, fight_results, fight_stats, fighters, fighter_odds, event_odds, betfair_odds, betfair_upcoming_odds]:
        try:
            df[['event']] = utils.clean_text_columns(df[['event']])
        except:
            pass
        try:
            df[['bout']] = utils.clean_text_columns(df[['bout']])
        except:
            pass
        try:
            df[['fighter']] = utils.clean_text_columns(df[['fighter']])
        except:
            pass
        try:
            df[['opponent']] = utils.clean_text_columns(df[['opponent']])
        except:
            pass
    
    # Save the fighter formatted names separately
    fighter_names = fighters[['fighter', 'Fighter']].drop_duplicates(subset = ['fighter'])
    fighters.drop(columns = ['Fighter'], inplace = True)
    
    # Compute the control time in minutes
    fight_stats['ctrl_minutes'] = utils.time_to_minutes_decimal(fight_stats['ctrl'])
    
    # Format dates
    fighters['dob'] = pd.to_datetime(fighters['dob'].str.replace("-", ""), format = '%b %d, %Y')
    events['date'] = pd.to_datetime(events['date'], format = '%B %d, %Y')
    upcoming_events['date'] = pd.to_datetime(upcoming_events['date'], format = '%B %d, %Y')
    
    # Add the event date to the fight results
    fight_results = fight_results.merge(events[['event', 'date']], on = 'event', how = 'left')
    
    # Reformat weightclass
    fight_results['weightclass'] = fight_results['weightclass'].str.replace('UFC', '').str.replace('Bout', '').str.replace('Title', '').str.replace(' ', '').str.replace('Interim', '')
    upcoming_fight_details['weightclass'] = upcoming_fight_details['weightclass'].str.replace('UFC', '').str.replace('Bout', '').str.replace('Title', '').str.replace(' ', '').str.replace('Interim', '')
    
    # Clean the methods of victory
    fight_results['method'] = fight_results['method'].str.replace(' ', '')
    fight_results['method'] = np.where(fight_results['method'].isin(config['methods']), fight_results['method'] , 'Other')
    fight_results['method'] = np.where(fight_results['method'] == 'Decision-Majority', 'Decision-Split', fight_results['method'])
    
    # Compute rounds
    fight_results['rounds'] = fight_results['time format'].str[0:1]
    
    # Compute numeric values for strikes and tds
    fight_stats = utils.split_of_columns(fight_stats, ['sig.str.', 'total str.', 'td', 'head', 'body', 'leg', 'distance', 'clinch', 'ground'])
    
    # Split fight into fighters
    fight_results[['fighter1', 'fighter2']] = fight_results['bout'].str.split('vs\\.', expand = True)
    upcoming_fight_details[['fighter1', 'fighter2']] = upcoming_fight_details['bout'].str.split('vs\\.', expand = True)
    
    # Aggregate the fight stats to be at fight-fighter level
    # Merge on the final round time from the fight results
    fight_results['fight_minutes'] = utils.time_to_minutes_decimal(fight_results['time'])
    fight_stats = utils.to_numeric(fight_stats, ['round', 'sub.att'])
    fight_results = utils.to_numeric(fight_results, ['round'])
    # Event is needed in the join, otherwise a rematch picks up the first fight's round time
    fight_stats = fight_stats.merge(fight_results[['event', 'bout', 'round', 'fight_minutes']], on = ['event', 'bout', 'round'], how = 'left')
    # Set fight time (5 minute round assumed)
    fight_stats['fight_minutes'] = fight_stats['fight_minutes'].fillna(config['round_time_default'])
    
    # Remove columns not needed
    fighters.drop(columns = ['feet', 'inches', 'url'], inplace = True)
    fight_results.drop(columns = ['time', 'referee', 'details', 'url', 'fight_minutes'], inplace = True)
    fight_stats.drop(columns = ['ctrl'], inplace = True)
    
    # Find the columns to be aggregated
    agg_cols = ['kd', 'ctrl_minutes', 'fight_minutes', 'sub.att'] + [col for col in fight_stats.columns if '_attempt' in col or '_landed' in col]
    id_cols = ['event', 'bout', 'fighter']
    
    fight_stats_agg = fight_stats[id_cols].drop_duplicates()
    
    # Group and calculate sums for each round and the total
    for round in range(1, 6):
        prefix = "round" + str(round) + "_"
        # Filter for that round then rename the columns for that round
        new_stats = fight_stats[id_cols + agg_cols + ['round']][fight_stats['round'] == round].drop(columns = ['round']).rename(
            columns = dict(zip(agg_cols, [prefix + col for col in agg_cols])))
        # Get rid of duplicates. AB: Don't know why they're in there and a general drop_duplicates() doesn't work, but they have the same stats
        new_stats.drop_duplicates(subset = id_cols, inplace = True)
        # Merge onto the data
        fight_stats_agg = fight_stats_agg.merge(new_stats, on = id_cols, how = 'left')
    
    del([new_stats, prefix])
    
    # Fill the NAs in the fight stats with 0 for landed and attempts
    stats_cols = [col for col in fight_stats_agg.columns.to_list() if col not in id_cols]
    fight_stats_agg.loc[:, stats_cols] = fight_stats_agg.loc[:, stats_cols].fillna(0)
    
    # Calculate the total for all rounds then merge onto data
    fight_stats_all_round = fight_stats.groupby(id_cols)[agg_cols].sum().reset_index().rename(columns = dict(zip(agg_cols, ['allrounds_' + col for col in agg_cols])))
    fight_stats_agg = fight_stats_agg.merge(fight_stats_all_round, on = id_cols, how = 'left')
    del(fight_stats_all_round)
    
    # Check the row numbers
    if (len(fight_stats_agg) == len(fight_stats[id_cols].drop_duplicates())):
        print('Row check PASSED')
    else:
        print('Row check FAILED')
        raise SystemExit
    
    fight_stats_agg['fights'] = 1
    # Add the date to the fight stats
    fight_stats_agg = fight_stats_agg.merge(events[['event', 'date']], on = 'event', how = 'left')
    
    # Fix column formats
    fight_stats_agg['fighter'] = fight_stats_agg['fighter'].astype(str)
    
    # Remove duplicate fighters. Sort first to keep records without NAs
    fighters = fighters.sort_values(by = 'reach')
    fighters = fighters.drop_duplicates(subset = ['fighter'])
    fighters = fighters.sort_values(by = 'fighter')
    
    # Remove stats where fighter = '0'. These seem to be fights where there are no stats available so not useful
    fight_stats_agg = fight_stats_agg[fight_stats_agg['fighter'] != '0']
    
    ###########################################
    ## Fighter odds cleaning
    
    # Add upcoming to events df
    all_events = pd.concat([upcoming_events, events], axis = 0)
    # Add upcoming to fight stats for new events and bouts
    fight_stats_all = dataset.add_upcoming_to_fight_stats(fight_stats, upcoming_fight_details, events, upcoming_events)
    
    # List out the odds columns
    odds_columns = ['open_odds', 'closing_odds_low', 'closing_odds_high']
    
    # Convert the odds columns into numeric
    fighter_odds[odds_columns] = fighter_odds[odds_columns].apply(pd.to_numeric, errors = 'coerce')
    
    # Count how many rows are n/a for each event, focussing on open odds as they will most likely be used in the model
    fighter_odds_nas = fighter_odds[fighter_odds['event'].apply(lambda x: 'UFC' in str(x))].groupby('event').agg(
        date = ('date', 'first'),
        na_count = ('closing_odds_low', lambda x: x.isna().sum()),
        row_count = ('fighter', lambda x: x.notna().sum()))
    
    # Output this to check later to see which UFC events have NAs
    fighter_odds_nas[fighter_odds_nas['na_count'] > 0].to_csv('./data/checks_fighter_odds_nas.csv')
    
    # Now remove NAs from the data frame
    fighter_odds = fighter_odds[(fighter_odds['closing_odds_low'].notna()) & ~(fighter_odds['event'] == 'Future Events(')]
    
    # Clean the name columns. Not bothering with event because that column may not match the UFC events columns
    fighter_odds.loc[:, ['fighter', 'opponent']] = utils.clean_text_columns(fighter_odds[['fighter', 'opponent']])
    
    # Add the correct event and bout to the odds data
    fighter_odds = odds.add_event_and_bout_to_odds_data(fighter_odds, all_events, fight_stats_all)
    
    
    ###########################################
    ## Event odds cleaning
    
    event_odds_cols = odds_scrape_config['events_colnames']
    
    # # Convert odds to numeric
    # event_odds[event_odds_cols] = event_odds[event_odds_cols].apply(pd.to_numeric, errors = 'coerce')
    
    # Add event and bout to the data
    event_odds = odds.add_event_and_bout_to_odds_data(event_odds, all_events, fight_stats_all, date_check = '2018-01-01')
    
    # Remove empty rows
    event_odds = event_odds.dropna(subset = event_odds_cols, how = 'all')
    event_odds = event_odds.dropna(subset = 'event')
    
    
    
    ###########################################
    ## Betfair odds cleaning
    
    # Replace fighter names with those that appear in stats where the dates match
    betfair.betfair_clean_fighter_names(betfair_odds, all_events, fight_stats_all, fuzz_threshold = 70, day_threshold = 2)
    
    # Change column name for matching
    betfair_odds.rename(columns = {'event_date': 'date'}, inplace = True)
        
    # Add on events and bouts
    betfair_odds = odds.add_event_and_bout_to_odds_data(betfair_odds, all_events, fight_stats_all, date_check = '2017-01-01')
    
    ## Now clean the UPCOMING BOUTS
    # pdb.set_trace()
    # First get the bout information and sync to the stats data
    betfair_upcoming_odds = betfair.betfair_add_upcoming_bouts_details(betfair_upcoming_odds, upcoming_fight_details)
    
    
    ###########################################
    # Remove multiple fights on same date - some events had fighters fight multiple times. This messes up the matching and some functions so we'll get rid of them
    
    # Take the fight results data as the primary one
    fight_results = dataset.remove_duplicate_fighter_pairs(fight_results)
    
    # # Also remove fights where we have no stats # KEPT IN FOR ELO RANKS
    # fight_results = fight_results[(fight_results['event'].isin(fight_stats_agg['event'])) & (fight_results['bout'].isin(fight_stats_agg['bout']))]
    
    # Now remove those results from the other datasets
    fight_stats_agg = fight_stats_agg[(fight_stats_agg['event'].isin(fight_results['event'].tolist())) & (fight_stats_agg['bout'].isin(fight_results['bout'].tolist()))]
    
    # Remove odds but not for more recent events
    comparison_date = datetime.combine(date.today() - timedelta(days=5), datetime.min.time())
    
    fighter_odds = fighter_odds[((fighter_odds['event'].isin(fight_results['event'].tolist())) & (fighter_odds['bout'].isin(fight_results['bout'].tolist()))) | (fighter_odds['date'] > comparison_date)]
    
    event_odds = event_odds[((event_odds['event'].isin(fight_results['event'].tolist())) & (event_odds['bout'].isin(fight_results['bout'].tolist()))) | (event_odds['date'] > comparison_date)]
    
    ###########################################
    
    # Overwrite any outputs that need it
    return {
        'events': events,
        'upcoming_events': upcoming_events,
        'fighters': fighters,
        'upcoming_fight_details': upcoming_fight_details,
        'fight_results': fight_results,
        'fight_stats_agg': fight_stats_agg,
        'fighter_odds': fighter_odds,
        'event_odds': event_odds,
        'betfair_odds': betfair_odds,
        'fighter_names': fighter_names,
        'betfair_upcoming_odds': betfair_upcoming_odds
    }

def clean_all_raw_data_and_save(config):
    
    """
    This function takes all the raw scraped data and cleans them, then saves to file
    
    Returns: nothing
    
    """
    # pdb.set_trace()
        
    # Read in the data
    events = fp.ParquetFile('./data/' + config['event_details_file_name']).to_pandas()
    upcoming_events = fp.ParquetFile('./data/' + config['upcoming_event_details_file_name']).to_pandas()
    upcoming_fight_details = fp.ParquetFile('./data/' + config['upcoming_fight_details_file_name']).to_pandas()
    fighters = fp.ParquetFile('./data/' + config['fighter_tott_file_name']).to_pandas()
    fight_results = fp.ParquetFile('./data/' + config['fight_results_file_name']).to_pandas()
    fight_stats = fp.ParquetFile('./data/' + config['fight_stats_file_name']).to_pandas()
    fighter_odds = fp.ParquetFile('./data/' + config['fighter_odds_file_name']).to_pandas()
    event_odds = fp.ParquetFile('./data/' + config['event_odds_file_name']).to_pandas()
    betfair_odds = fp.ParquetFile('./data/' + config['betfair_odds_file_name']).to_pandas()
    betfair_upcoming_odds = fp.ParquetFile('./data/' + config['betfair_upcoming_odds_file_name']).to_pandas()
    
    # Data clearning
    cleaned_dict = clean_all_raw_data(events, upcoming_events, upcoming_fight_details, fighters, fight_results, fight_stats, fighter_odds, event_odds, betfair_odds, betfair_upcoming_odds)
    
    ###########################################
    
    # Save the outputs
    
    fp.write('./data/' + config['event_details_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['events'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['upcoming_event_details_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['upcoming_events'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['fighter_tott_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['fighters'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['upcoming_fight_details_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['upcoming_fight_details'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['fight_results_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['fight_results'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['fight_stats_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['fight_stats_agg'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['fighter_odds_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['fighter_odds'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['event_odds_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['event_odds'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['betfair_odds_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['betfair_odds'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['betfair_upcoming_odds_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['betfair_upcoming_odds'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['fighter_names_file_name'],
             cleaned_dict['fighter_names'],
             compression = 'ZSTD')

    return None

# This function will create a random forest of boosted trees.

def train_ensemble_xgboost(X, y, X_columns_for_training, num_models=10, num_boost_round=500, colsample_bynode=0.8, subsample=0.8, early_stopping_rounds=30, params=None, verbose_eval = False):
    """
    Trains an ensemble of boosted trees with random feature and row sampling for each model,
    ensuring that the validation sample consists of all data rows not included in the training data.

    Parameters:
    - X (pd.DataFrame): Feature matrix for training.
    - y (pd.Series or np.array): Target vector for training.
    - X_columns_for_training (list): a list of columns that should be used for training
    - num_models (int): Number of models (boosted trees) in the ensemble.
    - num_boost_round (int): Max number of boosting rounds for each model.
    - colsample_bynode (float): Fraction of columns to randomly sample for each tree.
    - subsample (float): Fraction of rows to sample for each model.
    - early_stopping_rounds (int): Number of rounds for early stopping.
    - params (dict): Parameters for the XGBoost model.

    Returns:
    - models (list): List of trained XGBoost models.
    """

    # pdb.set_trace()

    # Default parameters if none are provided
    if params is None:
        params = {
            'objective': 'multi:softprob',  # Set for multi-class classification; use 'binary:logistic' for binary
            'eval_metric': 'auc',
            'tree_method': 'hist',  # Efficient training with 'hist' method, use 'gpu_hist' if GPU is available
            'learning_rate': 0.2,
            'max_depth': 3,
            'colsample_bynode': colsample_bynode,
            'subsample': subsample,
            'verbosity': 1
        }

    # Set random seed for Numpy
    np.random.seed(config['seed_all'])
    
    # Get a full list of columns needed for switching fighters then cut down the data
    all_columns_for_diffs = features.all_columns_for_differentials(X_columns_for_training)
    X = X[[col for col in all_columns_for_diffs if col in X.columns] + [col for col in X.columns if 'notformodel' in col or ('odds' in col and col not in all_columns_for_diffs)]].copy()
    
    # Add an id column to the X data
    X['id'] = X.index.copy()
    
    # Calculate new odds columns to be removed later
    X['train_fighter1_odds'] = X['fighter1_backtesting_odds'].to_numpy() # avoids pandas thinking they're the same column
    X['train_fighter2_odds'] = X['fighter2_backtesting_odds'].to_numpy()

    models = []
    for i in range(num_models):
        
        # pdb.set_trace()
                
        model = {}
        
        # Randomly sample rows with replacement for the training set
        if subsample < 1:
            X_train_df, y_train_df = dataset.sample_fighters(
                X = X, 
                y = y, 
                n_rows = int(len(X) * subsample),
                rows_with_bagging = None if not params['xgb_forest_bagging'] else len(X))
            # X_train_df, y_train_df = resample(X, y, replace=True, n_samples=int(len(X) * subsample), random_state=i)
        else:
            X_train_df = X.copy()
            y_train_df = y.copy()
        
        # # Switch fighters if needed
        if params['switch_fighters_randomly'] == True:
            X_train_df, y_train_df = features.switch_fighter1_fighter2(X = X_train_df, y = y_train_df, filter_boolean = np.random.choice([True, False], size = len(X_train_df)))
            
        # Add noise to odds if required
        if params['add_noise_to_odds']:
            X_train_df = odds.add_noise_to_odds(X_train_df, norm_sd = params['odds_noise_standard_dev'])
        
        # Remove split decisions if required
        if params['split_dec_or_other']:
            y_train_df = y_train_df[X_train_df['notformodel_split_dec_or_other'] == 0]
            X_train_df = X_train_df[X_train_df['notformodel_split_dec_or_other'] == 0]
        
        # Calculate the number of columns to randomly select
        if colsample_bynode < 1:
            num_columns_to_select = int(colsample_bynode * len(X_columns_for_training))
            
            # First, duplicate all the diff rows, otherwise 'fighter' rows have 2x the chance of being chosen
            diff_cols = [col for col in X_columns_for_training if 'fighter1' not in col and 'fighter2' not in col]
            
            # Get a random selection of column names
            selected_columns = list(set(np.random.choice(X_columns_for_training + diff_cols, num_columns_to_select, replace = False)))
        else:
            selected_columns = X_columns_for_training
        
        # # Make sure both fighter1 and figher2 versions of columns in there or switch will fail
        # selected_columns = list(set(selected_columns + list_swap_fighter1_fighter2(selected_columns)))
        
        # Filter win loss only if applicable
        if params['win_loss_only'] == True:
            X_train_df = X_train_df.copy()[y_train_df.isin([0, 2])]
            y_train_df = y_train_df.copy()[y_train_df.isin([0, 2])]

        # Create the validation set from all rows not included in the training set
        # Mask to get the validation data
        train_ids = X_train_df['id'].tolist()
        X_val = X.copy()[~X['id'].isin(train_ids).to_numpy()]
        y_val = y.copy()[~X['id'].isin(train_ids).to_numpy()]

        # Get the odds for the custom loss function if needed
        train_fighter1_odds = X_train_df['train_fighter1_odds'].to_numpy()
        train_fighter2_odds = X_train_df['train_fighter2_odds'].to_numpy()
        test_fighter1_odds = X_val['train_fighter1_odds'].to_numpy()
        test_fighter2_odds = X_val['train_fighter2_odds'].to_numpy()

        # Get the base margins
        base_margin_train = modelling.get_base_margin(X_train_df, params['base_margin_cols'])
        base_margin_val = modelling.get_base_margin(X_val, params['base_margin_cols'])

        X_train_df =  X_train_df[selected_columns + ['id', 'train_fighter1_odds', 'train_fighter2_odds']]

        # Drop the ID column
        cols_to_drop = [col for col in X_train_df.columns if 'train_fighter' in col or 'backtesting' in col or col == 'id' or 'notformodel' in col]
        X_train_df.drop(cols_to_drop, axis = 1, inplace = True)
        X_val.drop(cols_to_drop, axis = 1, inplace = True)
        
        # Create the DMatrix for XGBoost with the bootstrap sample for training and the full remaining data for validation
        dmatrix_train = xgb.DMatrix(X_train_df, label=y_train_df, base_margin = base_margin_train)
        dmatrix_val = xgb.DMatrix(X_val[X_train_df.columns], label=y_val, base_margin = base_margin_val)
        
        # Train the model
        if params['calibrate_models']: # Calibration doesn't work with EV loss function at the moment. Not built.
            
            # Add a dummy 1 value in case it's missing (that causes the fit to fail)
            X_dummy = np.zeros((1, X_train_df.shape[1]))  # or any constant
            y_dummy = np.array([1])  # the missing class
            
            # Convert to Data Frames
            X_dummy = pd.DataFrame(X_dummy, columns = X_train_df.columns)
            
            X_train_aug = pd.concat([X_train_df, X_dummy], axis = 0)
            y_train_aug = pd.concat([y_train_df, pd.Series(y_dummy)], axis = 0)
            
            # Train the model in a way compatible with SKLearn
            raw_model = xgb.XGBClassifier(params)
            raw_model.fit(X_train_aug, y_train_aug,
                          eval_set = [(X_val[X_train_df.columns], y_val)],
                          early_stopping_rounds = early_stopping_rounds,
                          verbose = verbose_eval)                
            
            # Create a calibration model
            model_cal = TemperatureScaler(raw_model)
            # model_cal = CalibratedClassifierCV(estimator = raw_model, 
            #                                    method = 'sigmoid', # This is a Platt calibration
            #                                    cv = 'prefit') # the model has already been trained
            
            # Now fit to the validation data
            model_cal.fit(X_val[X_train_df.columns], y_val)
            
            model['xgb_model'] = raw_model
            model['model'] = model_cal
        
        elif model_hyper_params['loss_function'] == 'ev_loss_function': # any eval metric above will be ignored
            # Pass the odds to the loss function
            custom_loss = modelling.create_custom_loss(train_fighter1_odds, train_fighter2_odds)
            custom_loss_test = modelling.ev_loss_function_test(test_fighter1_odds, test_fighter2_odds)
            
            # Train an XGBoost model on the bootstrap sample with column sampling
            model['model'] = xgb.train(
                params,
                dmatrix_train,
                num_boost_round = num_boost_round,
                evals=[(dmatrix_val, 'eval')],
                early_stopping_rounds = early_stopping_rounds,
                obj = custom_loss,
                custom_metric = custom_loss_test,
                verbose_eval = verbose_eval)
            
            # model['model'].model_features = X_train_df.columns
        
        else:
            # Train an XGBoost model on the bootstrap sample with column sampling
            model['model'] = xgb.train(
                params,
                dmatrix_train,
                num_boost_round = num_boost_round,
                evals=[(dmatrix_val, 'eval')],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval = verbose_eval)
            
        # # Manually set the list of model features
        # model['model'].feature_names = list(X_train_df.columns)
        
        # Join the feature names into a single string and save them as a model attribute
        model['features'] = selected_columns
        
        # Save the parameters used
        model['params'] = params
        model['calibrated'] = params['calibrate_models']
        
        # Add model to the list
        models.append(model)
        
        del X_train_df
        gc.collect()
        print(f"Model {i + 1}/{num_models} trained.")

    return models
