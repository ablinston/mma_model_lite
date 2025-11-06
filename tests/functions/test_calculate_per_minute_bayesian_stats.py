# Load all the packages and functions needed
exec(open('global.py').read())

# # Load what answers are expected
test_calculate_per_minute_bayesian_stats_correct = fp.ParquetFile('./tests/data/test_calculate_per_minute_bayesian_stats.parquet').to_pandas()

def test_calculate_per_minute_bayesian_stats_long_dataset():
    # Create 10 dummy fights
    n = 10  # number of bouts

    # one row per bout initially
    fight_stats_hist = pd.DataFrame({
        'event': [f'UFC{i}' for i in range(1, n + 1)],
        'bout': list(range(1, n + 1)),
        'date': [(datetime.today() + timedelta(days=i)).date() for i in range(n)],
    })
    
    # --- Create two fighters per bout with unique stats ---
    fighters = ['Fighter_A', 'Fighter_B']
    
    np.random.seed(12434)
    
    records = []
    for _, row in fight_stats_hist.iterrows():
        # randomize or differentiate stats for the two fighters
        for fighter, mult in zip(fighters, [1.0, np.random.uniform(0.7, 1.3)]):
            record = {
                'event': row['event'],
                'bout': row['bout'],
                'date': row['date'],
                'fighter': fighter,
                'round1_strikes': int(10 * mult + row['bout']),
                'round2_strikes': int(5 * mult + row['bout']),
                'round1_fight_minutes': 5,
                'round2_fight_minutes': 5,
                'allrounds_strikes': int(30 * mult + row['bout']),
                'allrounds_fight_minutes': 15,
            }
            records.append(record)
    
    fight_stats_hist_2fighters = pd.DataFrame(records)
    
    bayes_priors_df = bayesian_priors(
        fight_stats_df = fight_stats_hist_2fighters, 
        id_cols = ['event', 'bout', 'fighter'], 
        group_by = [], 
        fight_span = 3, 
        ignore_last_n_fights = 1)

    bayes_alpha_plus_beta_rebase = 5

    # Run the function
    result = calculate_per_minute_bayesian_stats(fight_stats_hist_2fighters, bayes_priors_df, bayes_alpha_plus_beta_rebase).drop(columns = 'date')

    # Remove date
    fight_stats_hist_2fighters.drop(columns = 'date', inplace = True)

    # ---- Assertions ----
    # Check the original columns are unchanged
    assert result[fight_stats_hist_2fighters.columns].equals(fight_stats_hist_2fighters)

    # Ensure the result is correct. I checked this manually in Excel replicating the calculations
    assert result.equals(test_calculate_per_minute_bayesian_stats_correct)

# fp.write('./tests/data/test_calculate_per_minute_bayesian_stats.parquet', result, compression = 'ZSTD')
