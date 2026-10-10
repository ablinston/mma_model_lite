# Load all the packages and functions needed
from datetime import datetime
import numpy as np
import pandas as pd

from mma_functions.betfair import betfair_bout_get_chosen_odds

# Test cases for the trim_column_list function
def test_betfair_bout_get_chosen_odds():
    
    # Simulated bet data for two fighters across two days
    data = {
        "fighter": ["A"] * 3 + ["B"] * 3,
        "bout": ["A vs B"] * 6,
        "inplay": [False] * 6,
        "GMT_datetime": [
            datetime(2025, 4, 22, 15, 0),  # A - valid
            datetime(2025, 4, 22, 20, 0),  # A - valid, should be selected
            datetime(2025, 4, 23, 15, 0),  # A - invalid (after cutoff)
            datetime(2025, 4, 22, 14, 0),  # B - valid
            datetime(2025, 4, 22, 18, 0),  # B - valid, should be selected
            datetime(2025, 4, 23, 16, 30), # B - invalid (after cutoff)
        ],
        "last_price_traded": [2.0, 2.2, 2.4, 1.9, 2.1, 2.3]
    }

    df = pd.DataFrame(data)

    # Cutoff at 16:15 means we'll use odds before 2025-04-22 16:15 (i.e., prior day cutoff)
    
    result = betfair_bout_get_chosen_odds(df.copy(), "16:15", average_last_n=1)
    
    # Basic shape and columns
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 2  # One row per fighter
    assert set(result["fighter"]) == {"A", "B"}
    assert "betfair_exch_odds" in result.columns

    expected_odds = pd.Series([
        2.4,  
        2.1
    ])

    assert result["betfair_exch_odds"].equals(expected_odds)
    
    # Average last 3
    result_3 = betfair_bout_get_chosen_odds(df.copy(), "16:15", average_last_n=3)

    expected_odds_3 = pd.Series([
        2.2,  
        2.0
    ])
    
    assert np.allclose(result_3["betfair_exch_odds"], expected_odds_3)
