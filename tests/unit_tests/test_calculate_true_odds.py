# Load all the packages and functions needed
import numpy as np
import pandas as pd

from mma_functions.odds import calculate_true_odds

def test_calculate_true_odds():
    # Test with example odds
    data = {
        "odds1": [2.5, 1.8, 3.2],
        "odds2": [1.5, 2.2, 1.8]
    }
    df = pd.DataFrame(data)

    # Expected results
    expected_true_odds1 = [2.6667, 1.8182, 2.7778]  # Pre-calculated values using online calculator
    expected_true_odds2 = [1.6000, 2.2222, 1.5625]  # Pre-calculated values using online calculator

    result = calculate_true_odds(df.copy(), "odds1", "odds2")

    # Assert the results are correct with a tolerance for floating-point comparisons
    assert np.round(result["col1"], 4).tolist() == expected_true_odds1
    assert np.round(result["col2"], 4).tolist() == expected_true_odds2

def test_input_dataframe_not_modified():
    # Test that the original DataFrame is not modified
    data = {
        "odds1": [2.5, 1.8, 3.2],
        "odds2": [1.5, 2.2, 1.8]
    }
    df = pd.DataFrame(data)
    df_original = df.copy()

    calculate_true_odds(df.copy(), "odds1", "odds2")

    # Assert the original DataFrame remains unchanged
    pd.testing.assert_frame_equal(df, df_original)
