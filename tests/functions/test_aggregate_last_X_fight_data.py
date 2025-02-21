# Load all the packages and functions needed
exec(open('global.py').read())

# Load what answers are expected
test_aggregate_last_X_fight_data_correct = fp.ParquetFile('./tests/data/test_aggregate_last_X_fight_data_correct.parquet').to_pandas()

# This test looks at a specific URL for an historic event and checks that the results match what is expected
def test_aggregate_last_X_fight_data():
    
    # Creating a dataframe with multiple fighters and time series
    data = {
        'fighter_id': ['fighter_1', 'fighter_1', 'fighter_1', 'fighter_2', 'fighter_2', 'fighter_2'],
        'date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01', '2024-01-01', '2024-02-01', '2024-03-01']),
        'score': [10, np.nan, 20, 15, 5, np.nan],  # Contains NAs
        'hits': [5, 10, 15, 7, np.nan, 9],  # Contains NAs
        'opponent': ['fighter_2', 'fighter_2', 'fighter_2', 'fighter_1', 'fighter_1', 'fighter_1']  # Grouping by fighter
    }
    df = pd.DataFrame(data)

    # Define columns and parameters
    ignore_cols = ['fighter_id', 'date', 'opponent']  # Columns to ignore in aggregation
    group_by = ['fighter_id']  # Group by fighter_id
    prefix = 'cumulative_'  # Prefix for new columns

    # Apply the function
    result = aggregate_last_X_fight_data(df, ignore_cols, group_by, prefix, 1000, per_minute_calc = True, bayes_priors_df = fight_stats_bayes_priors)

    # Ensure that 'fighter_id' and 'date' columns are preserved and not included in the aggregation
    assert 'fighter_id' in result.columns
    assert 'date' in result.columns
    assert 'score' not in result.columns
    assert 'hits' not in result.columns

    # Expected cumulative sums
    # Fighter 1:
    # score: [0, NaN, 30], hits: [0, 10, 30]
    # Fighter 2:
    # score: [0, 5, NaN], hits: [0, 7, 16]
    expected_result = {
        'fighter_id': ['fighter_1', 'fighter_1', 'fighter_1', 'fighter_2', 'fighter_2', 'fighter_2'],
        'opponent': ['fighter_2', 'fighter_2', 'fighter_2', 'fighter_1', 'fighter_1', 'fighter_1'],
        'date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01', '2024-01-01', '2024-02-01', '2024-03-01']),
        'cumulative_score': [np.nan, 10, 10, np.nan, 15, 20],  # Expected cumulative sums with handling for NAs
        'cumulative_hits': [np.nan, 5, 15, np.nan, 7, 7]  # Expected cumulative sums with handling for NAs
    }
    expected_df = pd.DataFrame(expected_result)

    # Compare the result to the expected dataframe
    pd.testing.assert_frame_equal(result[['fighter_id', 'opponent', 'date', 'cumulative_score', 'cumulative_hits']], expected_df)
    
    
    #################
    # Now test Bayesian
    
    data = {
        "event": ["E" + str(i) for i in range(1, 25)],
        "date": pd.date_range(start="2024-01-01", periods=24, freq="D"),
        "bout": [f"b{i}" for i in range(1, 25)],
        "fighter": ["f" + str(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
                                           1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]],
        "weightclass": ["light"] * 6 + ["mid"] * 2 + ["light"] * 2 + ["mid"] * 2 + 
                       ["light"] * 2 + ["mid"] * 2 + ["light"] * 2 + ["mid"] * 2 + 
                       ["light"] * 2 + ["mid"] * 2,
        "allrounds_hit": [5, 2, 3, 4, 15, 48, 84, 5, 1, 25, 5, 4, 5, 1, 16, 5, 2, 1, 2, 85, 55, 3, 58, 4],
        "allrounds_hit_against": [2, 5, 4, 3, 48, 15, 5, 84, 25, 1, 4, 5, 1, 5, 5, 16, 1, 2, 85, 2, 3, 55, 4, 58],
        "allrounds_fight_minutes": [5] * 24
    }
    
    df = pd.DataFrame(data)


    bayes_result = aggregate_last_X_fight_data(
        df, 
        ignore_cols = ['event', 'bout', 'date', 'fighter', 'weightclass'], 
        group_by = ['fighter'], 
        prefix = 'test_', 
        X = 3, 
        per_minute_calc = True, 
        bayes_priors_df = bayesian_priors(
            df,
            id_cols = ['event', 'bout', 'fighter'], 
            group_by = ['weightclass'], 
            fight_span = 3,
            ignore_last_n = 1
            ))
    
    assert bayes_result.equals(test_aggregate_last_X_fight_data_correct)

# fp.write('./tests/data/test_aggregate_last_X_fight_data_correct.parquet', bayes_result, compression = 'ZSTD')
