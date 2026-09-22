# Load all the packages and functions needed
import pandas as pd

from mma_functions.evaluation import metric_table_from_metrics_dictionary

# Sample data for testing
metrics_dict = {
    'col': {
        'group1': {'metric1': 0.5, 'metric2': 0.5},
        'group2': {'metric1': 0.25, 'metric2': 0.25}
    },
    'another_col': {
        'group1': {'metric1': 0.75, 'metric2': 0.75},
        'group2': {'metric1': 0.35, 'metric2': 0.35}
    }
}

# This test looks at a specific URL for an historic event and checks that the results match what is expected
def test_metric_table_from_metrics_dictionary():
    # Expected output DataFrame for 'col'
    expected_df = pd.DataFrame({
        'group': ['group1', 'group2'],
        'metric1': [0.5, 0.25],
        'metric2': [0.5, 0.25]
    })
    
    result_df = metric_table_from_metrics_dictionary(metrics_dict, 'col')
    
    # Assert the DataFrame is as expected
    pd.testing.assert_frame_equal(result_df, expected_df)
    
