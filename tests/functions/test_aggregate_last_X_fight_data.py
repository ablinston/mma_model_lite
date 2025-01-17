# Load all the packages and functions needed
exec(open('global.py').read())


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
    result = aggregate_last_X_fight_data(df, ignore_cols, group_by, prefix, 1000)

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
    
