
# Load all the packages and functions needed
exec(open('global.py').read())

# Load what answers are expected
test_bayesian_priors_correct = fp.ParquetFile('./tests/data/test_bayesian_priors_correct.parquet').to_pandas()

def test_bayesian_priors():
    # Create a larger mock dataset
    num_fights = 1000
    data = {
        'event': np.repeat([f'E{i}' for i in range(1, num_fights//2 + 1)], 2),
        'bout': np.repeat(np.arange(1, num_fights//2 + 1), 2),
        'fighter': [f'F{i}' for i in range(1, num_fights + 1)],
        'weightclass': np.random.choice(['Lightweight', 'Welterweight', 'Middleweight'], num_fights),
        'date': pd.date_range(start='2022-01-01', periods=num_fights, freq='D'),
        'stat1': [1] * num_fights,
        'stat2': [2, 2, 2, 4] * int(num_fights/4),
        'stat3': np.random.randint(1, 30, num_fights)
    }
    fight_stats_df = pd.DataFrame(data)
    fight_stats_df['bout'] = fight_stats_df['bout'].apply(str)
    
    id_cols = ['event', 'bout', 'fighter']
    group_by = ['weightclass']
    fight_span = 50  # Using a larger span for robust testing

    # Run the function
    result = bayesian_priors(fight_stats_df, id_cols, group_by, fight_span)
    
    # Assertions
    assert isinstance(result, pd.DataFrame), "Output should be a DataFrame"
    assert all(col.startswith('bayes_total_') for col in result.columns if col not in id_cols), "Columns should be prefixed with 'bayes_total_'"
    assert len(result) == len(fight_stats_df) / 2, "Output should have half the rows since duplicates are removed"
    assert test_bayesian_priors_correct.equals(result)
    
    
# fp.write('./tests/data/test_bayesian_priors_correct.parquet', result, compression = 'ZSTD')
