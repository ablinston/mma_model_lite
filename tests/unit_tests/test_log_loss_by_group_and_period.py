# Load all the packages and functions needed
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from mma_functions.evaluation import log_loss_by_group_and_period

# Check each cell against log_loss on that cell, and the change between the periods

def test_log_loss_by_group_and_period():

    actuals = pd.DataFrame({
        "method_group": ["Decision"] * 4 + ["KO/TKO"] * 4,
        "period": ["1. Pre", "1. Pre", "2. Post", "2. Post"] * 2,
        "fighter1_backtesting_odds": [1.5, 2.0, 3.0, 1.8, 2.5, 1.6, 2.2, 3.5],
        "fighter2_backtesting_odds": [2.5, 2.0, 1.4, 2.1, 1.6, 2.4, 1.7, 1.3]
        })

    probabilities = np.array([
        [0.10, 0.10, 0.80],
        [0.20, 0.10, 0.70],
        [0.40, 0.10, 0.50],
        [0.45, 0.10, 0.45],
        [0.15, 0.05, 0.80],
        [0.25, 0.05, 0.70],
        [0.60, 0.10, 0.30],
        [0.70, 0.10, 0.20]
        ])

    target = np.array([2, 2, 2, 0, 2, 2, 0, 0])

    period_metrics = log_loss_by_group_and_period(actuals, probabilities, target, "method_group", "period", min_sample_size=1)

    # One row per group, in sorted order
    assert period_metrics.index.tolist() == ["Decision", "KO/TKO"]

    # Every cell should match log_loss run on that cell on its own
    for group in ["Decision", "KO/TKO"]:
        for period in ["1. Pre", "2. Post"]:

            row_filter = (actuals["method_group"] == group) & (actuals["period"] == period)

            expected = log_loss(target[row_filter], probabilities[row_filter], labels=np.array([0, 1, 2]))

            assert np.isclose(period_metrics.loc[group, "Logloss " + period], expected), \
                f"Expected {expected} for {group} {period}, got {period_metrics.loc[group, 'Logloss ' + period]}"

            assert period_metrics.loc[group, "Sample Size " + period] == 2

    # The change should be the later period less the earlier one
    for group in ["Decision", "KO/TKO"]:
        assert np.isclose(
            period_metrics.loc[group, "Change Logloss"],
            period_metrics.loc[group, "Logloss 2. Post"] - period_metrics.loc[group, "Logloss 1. Pre"])

        assert np.isclose(
            period_metrics.loc[group, "Change X Odds Logloss"],
            period_metrics.loc[group, "X Odds Logloss 2. Post"] - period_metrics.loc[group, "X Odds Logloss 1. Pre"])

# Check a group that only appears in one period leaves the other period and the change missing

def test_log_loss_by_group_and_period_group_in_one_period_only():

    actuals = pd.DataFrame({
        "method_group": ["Decision", "Decision", "Submission", "Submission"],
        "period": ["1. Pre", "2. Post", "1. Pre", "1. Pre"],
        "fighter1_backtesting_odds": [1.5, 2.0, 3.0, 1.8],
        "fighter2_backtesting_odds": [2.5, 2.0, 1.4, 2.1]
        })

    probabilities = np.array([
        [0.10, 0.10, 0.80],
        [0.20, 0.10, 0.70],
        [0.60, 0.10, 0.30],
        [0.45, 0.10, 0.45]
        ])

    target = np.array([2, 2, 0, 0])

    period_metrics = log_loss_by_group_and_period(actuals, probabilities, target, "method_group", "period", min_sample_size=1)

    assert period_metrics.index.tolist() == ["Decision", "Submission"]

    # Submission never happened after the split, so there is nothing to compare it against
    assert np.isnan(period_metrics.loc["Submission", "Logloss 2. Post"])
    assert np.isnan(period_metrics.loc["Submission", "Change Logloss"])

    # Decision has both, so it gets a real change
    assert not np.isnan(period_metrics.loc["Decision", "Change Logloss"])

# Check one period gives no change columns, since a change of zero would read as nothing having changed

def test_log_loss_by_group_and_period_single_period():

    actuals = pd.DataFrame({
        "method_group": ["Decision", "Decision", "KO/TKO", "KO/TKO"],
        "period": ["2. Post"] * 4,
        "fighter1_backtesting_odds": [1.5, 2.0, 3.0, 1.8],
        "fighter2_backtesting_odds": [2.5, 2.0, 1.4, 2.1]
        })

    probabilities = np.array([
        [0.10, 0.10, 0.80],
        [0.20, 0.10, 0.70],
        [0.60, 0.10, 0.30],
        [0.45, 0.10, 0.45]
        ])

    target = np.array([2, 2, 0, 0])

    period_metrics = log_loss_by_group_and_period(actuals, probabilities, target, "method_group", "period", min_sample_size=1)

    assert "Logloss 2. Post" in period_metrics.columns
    assert "Change Logloss" not in period_metrics.columns
    assert "Change X Odds Logloss" not in period_metrics.columns

# Check cells with too few fights are left out, since a log loss on two or three fights says nothing

def test_log_loss_by_group_and_period_min_sample_size():

    actuals = pd.DataFrame({
        "method_group": ["Decision"] * 4 + ["Other"] * 2,
        "period": ["1. Pre", "1. Pre", "2. Post", "2. Post", "1. Pre", "2. Post"],
        "fighter1_backtesting_odds": [1.5, 2.0, 3.0, 1.8, 2.5, 1.6],
        "fighter2_backtesting_odds": [2.5, 2.0, 1.4, 2.1, 1.6, 2.4]
        })

    probabilities = np.array([
        [0.10, 0.10, 0.80],
        [0.20, 0.10, 0.70],
        [0.40, 0.10, 0.50],
        [0.45, 0.10, 0.45],
        [0.15, 0.05, 0.80],
        [0.25, 0.05, 0.70]
        ])

    target = np.array([2, 2, 2, 0, 1, 1])

    period_metrics = log_loss_by_group_and_period(actuals, probabilities, target, "method_group", "period", min_sample_size=2)

    # Decision has two fights per cell and survives, Other has one per cell and drops out entirely
    assert period_metrics.index.tolist() == ["Decision"]

# Check the draw and no contest count per cell, since a couple of those can swamp a comparison

def test_log_loss_by_group_and_period_counts_other_outcomes():

    actuals = pd.DataFrame({
        "method_group": ["Decision"] * 4,
        "period": ["1. Pre", "1. Pre", "2. Post", "2. Post"],
        "fighter1_backtesting_odds": [1.5, 2.0, 3.0, 1.8],
        "fighter2_backtesting_odds": [2.5, 2.0, 1.4, 2.1]
        })

    probabilities = np.array([
        [0.10, 0.10, 0.80],
        [0.20, 0.10, 0.70],
        [0.40, 0.10, 0.50],
        [0.45, 0.10, 0.45]
        ])

    # One draw in the later period and none in the earlier one
    target = np.array([2, 2, 2, 1])

    period_metrics = log_loss_by_group_and_period(actuals, probabilities, target, "method_group", "period", min_sample_size=1)

    assert period_metrics.loc["Decision", "OTH Fights 1. Pre"] == 0
    assert period_metrics.loc["Decision", "OTH Fights 2. Post"] == 1
