
exec(open('global.py').read())

# Read in the data
events = fp.ParquetFile('./data/' + config['event_details_file_name']).to_pandas()
upcoming_events = fp.ParquetFile('./data/' + config['upcoming_event_details_file_name']).to_pandas()
upcoming_fight_details = fp.ParquetFile('./data/' + config['upcoming_fight_details_file_name']).to_pandas()
fighters = fp.ParquetFile('./data/' + config['fighter_tott_file_name']).to_pandas()
fight_results = fp.ParquetFile('./data/' + config['fight_results_file_name']).to_pandas()
fight_stats = fp.ParquetFile('./data/' + config['fight_stats_file_name']).to_pandas()
fighter_odds = fp.ParquetFile('./data/' + config['fighter_odds_file_name']).to_pandas()
event_odds = fp.ParquetFile('./data/' + config['event_odds_file_name']).to_pandas()

# Convert all column names to lowercase
events.columns = events.columns.str.lower()
upcoming_events.columns = upcoming_events.columns.str.lower()
upcoming_fight_details.columns = upcoming_fight_details.columns.str.lower()
fighters.columns = fighters.columns.str.lower()
fight_results.columns = fight_results.columns.str.lower()
fight_stats.columns = fight_stats.columns.str.lower()
fighter_odds.columns = fighter_odds.columns.str.lower()
event_odds.columns = event_odds.columns.str.lower()

# Convert fighter data to numeric
# Replace spaces and inches, then split and convert to cm
fighters[['feet', 'inches']] = fighters['height'].str.replace(' ', '').str.replace('"', '').str.replace('-', '').str.split("'", expand = True)
fighters.drop(columns = ['height'], inplace = True)
fighters['height'] = pd.to_numeric(fighters['feet']) * 30.48 + pd.to_numeric(fighters['inches']) * 2.54

fighters = to_numeric(fighters, ['weight', 'reach'])
fight_stats = to_numeric(fight_stats, ['kd'])

# Get cleaned versions of the columns
fighters['Fighter'] = fighters['fighter'].copy()


# Remove spaces in events to avoid merge errors
for df in [events, upcoming_events, upcoming_fight_details, fight_results, fight_stats, fighters, fighter_odds, event_odds]:
    try:
        df[['event']] = clean_text_columns(df[['event']])
    except:
        pass
    try:
        df[['bout']] = clean_text_columns(df[['bout']])
    except:
        pass
    try:
        df[['fighter']] = clean_text_columns(df[['fighter']])
    except:
        pass
    try:
        df[['opponent']] = clean_text_columns(df[['opponent']])
    except:
        pass

# Save the fighter formatted names separately
fighter_names = fighters[['fighter', 'Fighter']].drop_duplicates(subset = ['fighter'])
fighters.drop(columns = ['Fighter'], inplace = True)

# Compute the control time in minutes
fight_stats['ctrl_minutes'] = time_to_minutes_decimal(fight_stats['ctrl'])

# Format dates
fighters['dob'] = pd.to_datetime(fighters['dob'], errors = 'coerce')
events['date'] = pd.to_datetime(events['date'], errors = 'coerce')
upcoming_events['date'] = pd.to_datetime(upcoming_events['date'], errors = 'coerce')

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
fight_stats = split_of_columns(fight_stats, ['sig.str.', 'total str.', 'td', 'head', 'body', 'leg', 'distance', 'clinch', 'ground'])

# Split fight into fighters
fight_results[['fighter1', 'fighter2']] = fight_results['bout'].str.split('vs\\.', expand = True)
upcoming_fight_details[['fighter1', 'fighter2']] = upcoming_fight_details['bout'].str.split('vs\\.', expand = True)

# Aggregate the fight stats to be at fight-fighter level
# Merge on the final round time from the fight results
fight_results['fight_minutes'] = time_to_minutes_decimal(fight_results['time'])
fight_stats = to_numeric(fight_stats, ['round', 'sub.att'])
fight_results = to_numeric(fight_results, ['round'])
fight_stats = fight_stats.merge(fight_results[['bout', 'round', 'fight_minutes']], on = ['bout', 'round'], how = 'left')
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
fight_stats_agg.fillna(0, inplace = True)

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
fighter_odds.loc[:, ['fighter', 'opponent']] = clean_text_columns(fighter_odds[['fighter', 'opponent']])

# Add the correct event and bout to the odds data
fighter_odds = add_event_and_bout_to_odds_data(fighter_odds, events, fight_stats)


###########################################
## Event odds cleaning

event_odds_cols = odds_scrape_config['events_colnames']

# # Convert odds to numeric
# event_odds[event_odds_cols] = event_odds[event_odds_cols].apply(pd.to_numeric, errors = 'coerce')

# Add event and bout to the data
event_odds = add_event_and_bout_to_odds_data(event_odds, events, fight_stats, date_check = '2018-01-01')

# Remove empty rows
event_odds = event_odds.dropna(subset = event_odds_cols, how = 'all')
event_odds = event_odds.dropna(subset = 'event')


###########################################
# Remove multiple fights on same date - some events had fighters fight multiple times. This messes up the matching and some functions so we'll get rid of them

# Take the fight results data as the primary one
fight_results = remove_duplicate_fighter_pairs(fight_results)

# # Also remove fights where we have no stats # KEPT IN FOR ELO RANKS
# fight_results = fight_results[(fight_results['event'].isin(fight_stats_agg['event'])) & (fight_results['bout'].isin(fight_stats_agg['bout']))]

# Now remove those results from the other datasets
fight_stats_agg = fight_stats_agg[(fight_stats_agg['event'].isin(fight_results['event'].tolist())) & (fight_stats_agg['bout'].isin(fight_results['bout'].tolist()))]

# Remove odds but not for more recent events
comparison_date = datetime.combine(date.today() - timedelta(days=5), datetime.min.time())

fighter_odds = fighter_odds[((fighter_odds['event'].isin(fight_results['event'].tolist())) & (fighter_odds['bout'].isin(fight_results['bout'].tolist()))) | (fighter_odds['date'] > comparison_date)]

event_odds = event_odds[((event_odds['event'].isin(fight_results['event'].tolist())) & (event_odds['bout'].isin(fight_results['bout'].tolist()))) | (event_odds['date'] > comparison_date)]

###########################################

# Save the outputs

fp.write('./data/' + config['event_details_file_name'].replace('.parquet', '_clean.parquet'),
         events,
         compression = 'ZSTD')

fp.write('./data/' + config['upcoming_event_details_file_name'].replace('.parquet', '_clean.parquet'),
         upcoming_events,
         compression = 'ZSTD')

fp.write('./data/' + config['fighter_tott_file_name'].replace('.parquet', '_clean.parquet'),
         fighters,
         compression = 'ZSTD')

fp.write('./data/' + config['upcoming_fight_details_file_name'].replace('.parquet', '_clean.parquet'),
         upcoming_fight_details,
         compression = 'ZSTD')

fp.write('./data/' + config['fight_results_file_name'].replace('.parquet', '_clean.parquet'),
         fight_results,
         compression = 'ZSTD')

fp.write('./data/' + config['fight_stats_file_name'].replace('.parquet', '_clean.parquet'),
         fight_stats_agg,
         compression = 'ZSTD')

fp.write('./data/' + config['fighter_odds_file_name'].replace('.parquet', '_clean.parquet'),
         fighter_odds,
         compression = 'ZSTD')

fp.write('./data/' + config['event_odds_file_name'].replace('.parquet', '_clean.parquet'),
         event_odds,
         compression = 'ZSTD')

fp.write('./data/' + config['fighter_names_file_name'],
         fighter_names,
         compression = 'ZSTD')