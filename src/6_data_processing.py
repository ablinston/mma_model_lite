'''

Overview
This script takes the cleaned data, merges it and aggregates necessary background info

'''

# imports
exec(open('global.py').read())

# Read in the data
events = fp.ParquetFile('./data/' + config['event_details_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
fighters = fp.ParquetFile('./data/' + config['fighter_tott_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
fight_results = fp.ParquetFile('./data/' + config['fight_results_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
fight_stats = fp.ParquetFile('./data/' + config['fight_stats_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
fight_odds = fp.ParquetFile('./data/' + config['fighter_odds_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()



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
fight_stats_against[['fighter1', 'fighter2']] = fight_stats_against['bout'].str.split('vs\.', expand = True)
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
fight_stats_hist = aggregate_historic_fight_data(
    fight_stats,
    ignore_cols = ['date', 'event', 'bout', 'fighter'],
    group_by = ['fighter'],
    prefix = "hist_"
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
# Calculate average fight length, win length and loss length
fight_stats_hist['hist_fight_length_minutes_avg'] = np.where(fight_stats_hist['hist_fights'] == 0, np.nan, fight_stats_hist['hist_allrounds_fight_minutes'] / fight_stats_hist['hist_fights'])

###########################################################################################
# Get the stats of the last 3 fights only
fight_stats_hist_last_3 = aggregate_last_3_fight_data(
    fight_stats,
    ignore_cols = ['date', 'event', 'bout', 'fighter'],
    group_by = ['fighter'],
    prefix = "hist_last_3_fights_"
).drop(columns = ['date'])

# Check that all first fights are NaN
stat_columns = [col for col in fight_stats.columns if col not in ['date', 'event', 'bout', 'fighter']]
fight_stats_hist_last_3_check = fight_stats_hist_last_3[fight_stats_hist_last_3['hist_last_3_fights_fights'] == 0]

if np.isnan(fight_stats_hist_last_3_check[['hist_last_3_fights_' + s for s in stat_columns]].max(skipna = True).max(skipna = True)):
    print('Cumulative aggregation check PASSED')
else:
    print('Cumulative aggregation check FAILED')
    raise SystemExit
del fight_stats_hist_last_3_check


###########################################################################################
# Get the fight stats of each fighter against only specific types of fighters
stances = fighters['stance'].str.replace(' ', '').unique().tolist()
fight_stats[['fighter1', 'fighter2']] = fight_stats['bout'].str.split('vs\.', expand = True)
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
fight_stats_hist_opponent_stance = aggregate_historic_fight_data(
    fight_stats,
    ignore_cols = ['date', 'event', 'bout', 'fighter', 'opponent', 'opponent_stance'],
    group_by = ['fighter', 'opponent_stance'],
    prefix = 'hist_opp_stance_'
).drop(columns = ['date', 'opponent', 'opponent_stance'])

# Check the row numbers are correct
stop_if_not(len(fight_stats) == len(fight_stats_hist_opponent_stance), "Opponent data length check")


###########################################################################################
# Calculate per minute stats for the many hist datasets

fight_stats_hist = calculate_per_minute_stats(fight_stats_hist)
fight_stats_hist_opponent_stance = calculate_per_minute_stats(fight_stats_hist_opponent_stance)
fight_stats_hist_last_3 = calculate_per_minute_stats(fight_stats_hist_last_3)

# Check all the row numbers are correct
stop_if_not(
    len(fight_stats) == len(fight_stats_hist) == len(fight_stats_hist_opponent_stance) == len(fight_stats_hist_last_3),
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
fight_stats_hist_opponent_type = aggregate_historic_fight_data(
    fight_stats,
    ignore_cols = ['date', 'event', 'bout', 'fighter', 'opponent', 'opponent_stance', 'fighter_type', 'opponent_type'],
    group_by = ['fighter', 'opponent_type'],
    prefix = 'hist_opp_type_'
).drop(columns = ['date', 'opponent', 'opponent_stance', 'fighter_type', 'opponent_type'])

# Add on the per minute stats
fight_stats_hist_opponent_type = calculate_per_minute_stats(fight_stats_hist_opponent_type)

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
dates_knocked_out[['hist_times_knocked_out', 'hist_times_knocked_out_round_1', 'hist_times_knocked_out_round_2', 'hist_times_knocked_out_round_3', 'hist_times_knocked_out_round_4', 'hist_times_knocked_out_round_5']] = dates_knocked_out[['hist_times_knocked_out', 'hist_times_knocked_out_round_1', 'hist_times_knocked_out_round_2', 'hist_times_knocked_out_round_3', 'hist_times_knocked_out_round_4', 'hist_times_knocked_out_round_5']].fillna(0) 

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
methods = list(set(fighter_results['method']))
row_count = len(fighter_results)
for result in ['W', 'L']:
    for method in methods:
        # First calculate a simple flag for whether the win or loss was by that method
        fighter_results['hist_' + result + '_' + method] = np.where((fighter_results['method'] == method) & (fighter_results['result'] == result),                                                                   1, 0)
        # Then sum these for previous fights ignoring the current one
        fighter_results['hist_' + result + '_' + method] = fighter_results.groupby(['fighter'])['hist_' + result + '_' + method].cumsum() - fighter_results['hist_' + result + '_' + method]


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
fight_results = fight_results[fight_results['time format'].isin(config['time_format_filter']) & fight_results['weightclass'].isin(config['weightclass_filter'])]

# Add elevation data to fight results
events['fight_elevation'] = get_elevation_series(events['location'], config['google_maps_api_key'])

# Merge onto the fight results
fight_results = fight_results.merge(
    events[['event', 'fight_elevation']],
    on = 'event',
    how = 'left'
    )

# Check there are no missing elevations
stop_if_not(fight_results['fight_elevation'].isna().sum() == 0, "Event elevation missing data check")

###########################################################################################
# FIGHT ODDS PROCESSING

# Convert opening odds to decimal
fight_odds['open_odds'] = fight_odds['open_odds'].apply(american_odds_to_decimal)

# Record the best odds as the highest (as that is what will give the biggest return)
fight_odds['best_closing_odds'] = fight_odds['closing_odds_high'].apply(american_odds_to_decimal)

# Calculate the odds change
fight_odds['odds_change'] = fight_odds['best_closing_odds'] - fight_odds['open_odds']

# Keep only the columns we need, and remove duplicates (these duplicates are where the event and bout are empty)
fight_odds = fight_odds[['event', 'bout', 'fighter', 'open_odds', 'best_closing_odds', 'odds_change']].drop_duplicates(subset = ['event', 'bout', 'fighter'])


###########################################################################################
# MERGE DATA

merged_data = merge_df_both_fighters(
    left_df =  fight_results[['event', 'bout', 'fighter1', 'fighter2', 'outcome', 'method', 'weightclass', 'date', 'time format', 'fighter1_start_elo', 'fighter2_start_elo', 'fight_elevation']],
    df_list = [
        fight_stats[['event', 'bout', 'fighter', 'date_last_fight']],
        dates_knocked_out,
        fight_stats_hist,
        fight_stats_hist_last_3,
        fight_stats_hist_opponent_stance,
        fight_stats_hist_opponent_type,
        fighter_results,
        # win_loss_stats,
        fight_odds
    ]
).reset_index().drop(columns = ['index'])

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

###########################################################################################
# PERCENTAGE CALCULATIONS

# Calculate percentages for those columns with attempts and lands
percent_columns = [col for col in merged_data.columns if '_attempt' in col and 'per_minute' not in col]
landed_columns = [col for col in merged_data.columns if '_landed' in col and 'per_minute' not in col]
percent_columns.sort()
landed_columns.sort()

# Check we have all the corresponding columns
if ([string.replace('_attempt', '') for string in percent_columns] == [string.replace('_landed', '') for string in landed_columns]):
    print("Percent column correspodence check PASSED")
else:
    print("Percent column correspodence check FAILED")
    raise SystemExit

# Now calculate the percentages
new_percent_cols = [col.replace('_attempt', '_accuracy_percent') for col in percent_columns]
merged_data[landed_columns].index = merged_data[percent_columns].index

# Shapes where strange so had to use numpy
new_data = pd.DataFrame(merged_data[landed_columns].to_numpy() / merged_data[percent_columns].to_numpy(), columns = new_percent_cols)

# Remove percentages for sample sizes below minimum thresholds, excluding takedowns
percent_columns_no_td = [col for col in percent_columns if "_td_" not in col]
new_percent_cols_no_td = [col for col in new_percent_cols if "_td_" not in col]
new_data.reset_index(inplace = True)
new_data[new_percent_cols_no_td] = np.where(merged_data[percent_columns_no_td] < config['min_attempts'], np.nan, new_data[new_percent_cols_no_td])

# Combine with data
merged_data = pd.concat([merged_data, new_data], axis = 1)

del([new_data, percent_columns, landed_columns, new_percent_cols])


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
head_body_ratios = pd.DataFrame(merged_data[head_columns].to_numpy() / merged_data[body_columns].to_numpy(),
                                columns = [col.replace('head_attempt', 'head_body_attempt_ratio') for col in head_columns])
merged_data = pd.concat([merged_data, head_body_ratios], axis = 1)
head_leg_ratios = pd.DataFrame(merged_data[head_columns].to_numpy() / merged_data[leg_columns].to_numpy(),
                                columns = [col.replace('head_attempt', 'head_leg_attempt_ratio') for col in head_columns])
merged_data = pd.concat([merged_data, head_leg_ratios], axis = 1)
del(head_body_ratios, head_leg_ratios, head_columns, body_columns, leg_columns)

# Do a check to see if row number has stayed the same
stop_if_not(len(merged_data) == merged_original_rows)

# Calculate ratio of striking to takedowns by looking at distance striking compared with td attempts, and distance compared with ground and pound
for fighter in ['fighter1', 'fighter2']:
    merged_data[fighter + '_hist_total_distance_strikes_over_td_attempts'] = merged_data[fighter + '_hist_allrounds_distance_attempt'] / merged_data[fighter + '_hist_allrounds_td_attempt']
    merged_data[fighter + '_hist_total_distance_over_ground_strikes'] = merged_data[fighter + '_hist_allrounds_distance_attempt'] / merged_data[fighter + '_hist_allrounds_ground_attempt']



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
    
    merged_data[new_columns] = merged_data[new_columns].replace([-np.inf, np.inf], np.nan)


###########################################################################################
# TIME SINCE CALCULATIONS

# Calculate days since last KO, Win, Loss, Fight
# Set the columns to be taken from date
cols = ['_date_last_fight', '_date_last_knocked_out', '_date_last_loss', '_date_last_win', '_dob']
# Do each fighters per fight calcs separately
for fighter in ['fighter1', 'fighter2']:
    for col in cols:
        merged_data[fighter + '_date_minus' + col] = (merged_data['date'] - merged_data[fighter + col]).dt.days

# pd.DataFrame(merged_data.columns).to_csv("te.csv")
###########################################################################################
# FIGHTER DIFFERENTIAL CALCULATIONS

# Check there are no duplicates first
duplicate_columns = merged_data.columns[merged_data.columns.duplicated()]
stop_if_not(len(duplicate_columns) == 0)

# Add in some combination of differentials for an overall fight
# Get the fighter columns
fighter1_cols = [col for col in merged_data.columns if 'fighter1_' in col]
fighter2_cols = [col for col in merged_data.columns if 'fighter2_' in col]

# Remove string columns
# Identify string columns
string_columns = merged_data.select_dtypes(include=['object']).columns
fighter1_cols = [col for col in fighter1_cols if col not in string_columns]
fighter2_cols = [col for col in fighter2_cols if col not in string_columns]

fighter1_cols.sort()
fighter2_cols.sort()

# Separate out the dates
date_columns = merged_data.select_dtypes(include=['datetime64']).columns

fighter1_cols_no_dates = [col for col in fighter1_cols if col not in date_columns]
fighter2_cols_no_dates = [col for col in fighter2_cols if col not in date_columns]
fighter1_cols_dates = [col for col in fighter1_cols if col in date_columns]
fighter2_cols_dates = [col for col in fighter2_cols if col in date_columns]

# Check that the columns correspond to one another
if check_identical_excl_string(fighter1_cols, fighter2_cols, 'fighter1_', 'fighter2_'):
    print("Column correspondence check PASSED")
else:
    print("Column correspondence check FAILED")
    raise SystemExit

stop_if_not(check_identical_excl_string(fighter1_cols_no_dates, fighter2_cols_no_dates, 'fighter1_', 'fighter2_'))

# Create differential columns
diff_columns = pd.DataFrame(merged_data[fighter1_cols_no_dates].to_numpy() - merged_data[fighter2_cols_no_dates].to_numpy(),
                            columns = [col.replace('fighter1_', 'diff_') for col in fighter1_cols_no_dates])

# Create differential columns but as ratios rather than differences
diff_ratio_columns = pd.DataFrame(merged_data[fighter1_cols_no_dates].to_numpy() / merged_data[fighter2_cols_no_dates].to_numpy(),
                            columns = [col.replace('fighter1_', 'diff_ratio_') for col in fighter1_cols_no_dates])

diff_date_columns = pd.DataFrame((merged_data[fighter1_cols_dates].to_numpy(dtype='datetime64') - merged_data[fighter2_cols_dates].to_numpy(dtype='datetime64')) / np.timedelta64(1, 'D'),
                            columns = [col.replace('fighter1_', 'diff_') for col in fighter1_cols_dates])

# Remove differential columns that might not be correct or not needed
diff_columns.drop(columns = ['diff_weight', 'diff_open_odds', 'diff_best_closing_odds'], inplace = True)
diff_ratio_columns.drop(columns = ['diff_ratio_weight'], inplace = True)

# Combine the columns onto the master dataset
merged_data = pd.concat([merged_data, diff_columns, diff_ratio_columns, diff_date_columns], axis = 1)
del([diff_columns, diff_date_columns, diff_ratio_columns])

# Add direct differentials between fighter offence and defence
fighter1_accuracy_off_cols = [col for col in merged_data.columns if ('accuracy_percent' in col or 'ctrl_minutes_per_minute' in col) and 'diff' not in col and 'fighter1' in col and 'against' not in col]
fighter2_accuracy_off_cols = [col for col in merged_data.columns if ('accuracy_percent' in col or 'ctrl_minutes_per_minute' in col) and 'diff' not in col and 'fighter2' in col and 'against' not in col]
fighter1_accuracy_def_cols = [col for col in merged_data.columns if ('accuracy_percent' in col or 'ctrl_minutes_against_per_minute' in col) and 'diff' not in col and 'fighter1' in col and 'against' in col]
fighter2_accuracy_def_cols = [col for col in merged_data.columns if ('accuracy_percent' in col or 'ctrl_minutes_against_per_minute' in col) and 'diff' not in col and 'fighter2' in col and 'against' in col]

# Check the columns line up
stop_if_not(check_identical_excl_string(fighter1_accuracy_off_cols, [col.replace('_against', '') for col in fighter2_accuracy_def_cols], 'fighter1', 'fighter2'))
stop_if_not(check_identical_excl_string(fighter2_accuracy_off_cols, [col.replace('_against', '') for col in fighter1_accuracy_def_cols], 'fighter2', 'fighter1'))

# Calculate the differentials
fighter1_off_fighter2_def = pd.DataFrame(merged_data[fighter1_accuracy_off_cols].to_numpy() - (1 - merged_data[fighter2_accuracy_def_cols].to_numpy()),
                                         columns = [col.replace('fighter1', 'fighter1_offense_minus_fighter2_defense') for col in fighter1_accuracy_off_cols])
fighter1_def_fighter2_off = pd.DataFrame(merged_data[fighter2_accuracy_off_cols].to_numpy() - (1 -  merged_data[fighter1_accuracy_def_cols].to_numpy()),
                                         columns = [col.replace('fighter2', 'fighter2_offense_minus_fighter1_defense') for col in fighter2_accuracy_off_cols])

# Combined with the data
merged_data = pd.concat([merged_data, fighter1_off_fighter2_def, fighter1_def_fighter2_off], axis = 1)

# Check rows is consistent
if (len(merged_data) == merged_original_rows):
    print("Row number check PASSED")
else:
    print("Row number check FAILED")
    raise SystemExit


###########################################################################################
# Formatting tweaks
# Remove any white spaces in names
merged_data.columns = merged_data.columns.str.replace(" ", "_")

# Add a womens flag
merged_data['womens'] = np.where(merged_data['weightclass'].apply(lambda x: np.where('omen' in x, True, False)), 1, 0)

###########################################################################################
# OUTPUT DATA

# Remove columns not needed, or that are all na
cols_to_remove = [col for col in merged_data.columns if any(substring in col for substring in ['bout', 'event', 'level_0'])]
# cols_to_remove.append('date')

merged_data.drop(columns = cols_to_remove, inplace = True)
merged_data.dropna(axis = 1, how = 'all', inplace = True)

# print(merged_data.columns)
merged_data.reset_index(inplace = True)
merged_data.drop(columns = cols_to_remove, inplace = True)
del(cols_to_remove)

split_by_year_and_save_fastparquet(merged_data, 'date', './data/final_dataset/', compression='ZSTD')
# fp.write('./data/final_dataset.parquet', merged_data, compression = 'GZIP')

# Output a single fight for checking purposes
merged_data[(merged_data['fighter1'] == 'IsraelAdesanya') & (merged_data['fighter2'] == 'SeanStrickland')].T.to_csv('./data/final_dataset_checking.csv', index=True)

merged_data[(merged_data['fighter1'] == 'AntonioSilva') | (merged_data['fighter2'] == 'AntonioSilva')].T.to_csv('./data/final_dataset_fighter_checking.csv')
