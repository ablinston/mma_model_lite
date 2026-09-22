# Load all the packages and functions needed
import numpy as np
import pandas as pd

from mma_functions.odds import calculate_target_odds

# Run a test on the target odds bu simulating 1,000,000 outcomes and calculate the actual EV

def test_calculate_target_odds():
    
    np.random.seed(42)

    # Setup
    n = 5000000
    p = 0.7  # fixed probability
    target_ev = 0.05
    commission = 0.0
    probabilities = pd.Series([p] * int(n))

    # Calculate target odds
    odds = calculate_target_odds(probabilities, target_ev, commission)

    # Simulate outcomes
    outcomes = np.random.binomial(1, p, n)
    profits = np.where(
        outcomes == 1,
        (odds - 1) * (1 - commission),
        -1.0
    )

    # Assert mean return ≈ target EV
    mean_profit = profits.mean()
    
    assert np.isclose(mean_profit, target_ev, atol=0.001), f"Expected EV ≈ {target_ev}, got {mean_profit}"

    p = 0.6  # fixed probability
    target_ev = 0.15
    commission = 0.02
    probabilities = pd.Series([p] * int(n))

    # Calculate target odds
    odds = calculate_target_odds(probabilities, target_ev, commission)

    # Simulate outcomes
    outcomes = np.random.binomial(1, p, n)
    profits = np.where(
        outcomes == 1,
        (odds - 1) * (1 - commission),
        -1.0
    )

    # Assert mean return ≈ target EV
    mean_profit = profits.mean()
    
    assert np.isclose(mean_profit, target_ev, atol=0.001), f"Expected EV ≈ {target_ev}, got {mean_profit}"
