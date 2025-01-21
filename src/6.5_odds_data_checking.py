# Read in global objects and packages
exec(open('global.py').read())


# Read in the final data
df = concat_parquets_fastparquet('./data/final_dataset/')

# Read in a dataset that can be used to get the names of fighters as they appear in the raw data
df_for_names = fp.ParquetFile('./data/' + config['fight_stats_file_name']).to_pandas().sort_values('FIGHTER')['FIGHTER'].unique()

# Get the manual odds data
odds_urls = pd.read_csv(config['fighter_odds_urls'])

# Find which fighters have the most missing odds data
te = pd.concat([df[df['fighter1_open_odds'].isna()]['fighter1'], df[df['fighter2_open_odds'].isna()]['fighter2']]).value_counts()
