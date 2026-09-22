# Load all the packages and functions needed
import numpy as np
import pandas as pd

from mma_functions.modelling import get_base_margin

def test_get_base_margin():
    
    # Fake data
    df = pd.DataFrame({
        "odds_A": [2.0, 4.0],   # implied 0.5, 0.25
        "odds_B": [2.0, 1.5],   # implied 0.5, 0.6667
    })
    
    # Case 1: None → uniform prior with 3 classes
    bm = get_base_margin(df, None)
    assert isinstance(bm, np.ndarray)
    assert bm.shape[0] == 2 * 3  # 2 rows, 3 classes
    
    # Case 2: With odds columns
    bm = get_base_margin(df, ["odds_A", {"flat_prob": 0.01}, "odds_B"])
    assert bm.shape[0] == 2 * 3  # 2 rows, 3 classes
    p = np.exp(bm.reshape(2, 3))  # exponentiate back
    np.testing.assert_allclose(p.sum(axis=1), np.ones(2))  # probs sum to 1
    
