
# Load all the packages and functions needed
import fastparquet as fp
import numpy as np
import pandas as pd

from mma_functions.features import bayesian_priors

# Load what answers are expected
test_bayesian_priors_correct = fp.ParquetFile('./tests/data/test_bayesian_priors_correct.parquet').to_pandas()



def test_bayesian_priors():
    
    np.random.seed(123)
    
    # Create a larger mock dataset
    num_fights = 100
    data = {
        'event': np.repeat([f'E{i}' for i in range(1, num_fights//2 + 1)], 2),
        'bout': np.repeat(np.arange(1, num_fights//2 + 1), 2),
        'fighter': [f'F{i}' for i in range(1, num_fights + 1)],
        'weightclass': np.repeat(np.random.choice(['Lightweight', 'Welterweight', 'Middleweight'], int(num_fights / 2)), 2),
        'date': pd.date_range(start='2022-01-01', periods=num_fights, freq='D'),
        'stat1': [1] * num_fights,
        'stat2': [2, 2, 2, 4] * int(num_fights/4),
        'stat3': np.random.randint(1, 30, num_fights)
    }
    fight_stats_df = pd.DataFrame(data)
    fight_stats_df['bout'] = fight_stats_df['bout'].apply(str)
    
    id_cols = ['event', 'bout', 'fighter']
    group_by = ['weightclass']
    fight_span = 2  # Using a larger span for robust testing

    # Run the function
    result = bayesian_priors(fight_stats_df, id_cols, group_by, fight_span, ignore_last_n_fights = 1)
    
    # Assertions
    assert isinstance(result, pd.DataFrame)
    assert all(col.startswith('bayes_total_') for col in result.columns if col not in id_cols)
    assert len(result) == len(fight_stats_df) / 2
    assert test_bayesian_priors_correct.sort_values(['bout']).equals(result[test_bayesian_priors_correct.columns].sort_values(['bout']))
    
    
# fp.write('./tests/data/test_bayesian_priors_correct.parquet', result, compression = 'ZSTD')
