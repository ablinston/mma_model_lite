# Load all the packages and functions needed
import numpy as np
import pandas as pd

from mma_functions.odds import odds_to_probabilities

# Test cases for the trim_column_list function
def test_odds_to_probabilities():
   
    odds_fighter1 = pd.Series([1.333333, 1.2])
    odds_fighter2 = pd.Series([4, 5.99999])

    result = odds_to_probabilities(odds_fighter1, odds_fighter2, normalise = True).map(lambda x: np.round(x, 6))
    
    expected_result = pd.DataFrame({
        'L/W': [0.25, 0.166667],
        'OTH': [0.0, 0.0],
        'W/L': [0.75, 0.833333]})
    
    assert result.equals(expected_result)
