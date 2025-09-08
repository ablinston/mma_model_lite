# Load all the packages and functions needed
exec(open('global.py').read())

# Load what answers are expected and inputs to the function
aggregate_last_3_fight_data_correct = fp.ParquetFile('./tests/data/test_aggregate_last_3_fight_data_correct.parquet').to_pandas()


# This test looks at a specific URL for an historic event and checks that the results match what is expected
def test_aggregate_last_3_fight_data():
    
    # Sample data to test
    data = {
        'event': range(1, 10),
        'fighter_id': [1, 1, 1, 1, 2, 2, 2, 2, 2],
        'date': ['2024-10-01', '2024-10-05', '2024-10-10', '2024-10-15', '2024-09-01', '2024-09-05', '2024-09-10', '2024-09-12', '2024-09-15'],
        'stat1': [10, 20, np.nan, 40, 15, 25, 35, 5, 10],
        'stat2': [5, 15, 25, 35, 10, 20, 30, 10, 5]
    }
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])

    # Define arguments for the function
    ignore_cols = ['event', 'date']
    group_by = ['fighter_id']
    prefix = 'last_3_'

    # Call the function
    result_df = aggregate_last_X_fight_data(df, ignore_cols, group_by, prefix, 3)
    
    assert result_df[['fighter_id', 'date', 'last_3_stat1', 'last_3_stat2']].equals(aggregate_last_3_fight_data_correct) == True
    

# fp.write('./tests/data/test_aggregate_last_3_fight_data_correct.parquet', result_df, compression = 'GZIP')
