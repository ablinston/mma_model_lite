# Load all the packages and functions needed
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from mma_functions.evaluation import rolling_log_loss
from mma_functions.odds import odds_to_probabilities

# Check that a window long enough to cover the whole sample gives the same answer as log_loss on the lot

def test_rolling_log_loss_full_window():

    actuals = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01", "2025-03-01", "2025-06-01", "2025-09-01", "2025-09-01"]),
        "fighter1_backtesting_odds": [1.5, 2.0, 3.0, 1.8, 2.5],
        "fighter2_backtesting_odds": [2.5, 2.0, 1.4, 2.1, 1.6]
        })

    probabilities = np.array([
        [0.10, 0.10, 0.80],
        [0.45, 0.10, 0.45],
        [0.70, 0.10, 0.20],
        [0.30, 0.10, 0.60],
        [0.55, 0.05, 0.40]
        ])

    target = np.array([2, 0, 0, 2, 0])

    # A window of a hundred years, so the last date covers every fight
    rolling = rolling_log_loss(actuals, probabilities, target, months=1200, min_sample_size=1)

    # One row per unique date, and the two fights on the same date share a window
    assert len(rolling) == 4
    assert rolling["Sample Size"].tolist() == [1, 2, 3, 5]

    whole_sample = log_loss(target, probabilities, labels=np.array([0, 1, 2]))

    assert np.isclose(rolling["Logloss"].iloc[-1], whole_sample), \
        f"Expected the last window to match log_loss on the whole sample of {whole_sample}, got {rolling['Logloss'].iloc[-1]}"

    # The odds column should match log_loss on the probabilities implied by the odds
    whole_sample_odds = log_loss(
        target,
        odds_to_probabilities(actuals["fighter1_backtesting_odds"], actuals["fighter2_backtesting_odds"]),
        labels=np.array([0, 1, 2]))

    assert np.isclose(rolling["X Odds Logloss"].iloc[-1], whole_sample_odds)

# Check a single date against a log loss worked out by hand

def test_rolling_log_loss_single_date():

    actuals = pd.DataFrame({
        "date": pd.to_datetime(["2025-05-10", "2025-05-10", "2025-05-10"]),
        "fighter1_backtesting_odds": [1.5, 2.0, 3.0],
        "fighter2_backtesting_odds": [2.5, 2.0, 1.4]
        })

    probabilities = np.array([
        [0.10, 0.10, 0.80],
        [0.20, 0.10, 0.70],
        [0.60, 0.10, 0.30]
        ])

    target = np.array([2, 2, 0])

    rolling = rolling_log_loss(actuals, probabilities, target, months=8, min_sample_size=1)

    # Log loss is the mean of minus the log of the probability given to the outcome that happened
    expected = -(np.log(0.80) + np.log(0.70) + np.log(0.60)) / 3

    assert len(rolling) == 1
    assert rolling["Sample Size"].iloc[0] == 3
    assert np.isclose(rolling["Logloss"].iloc[0], expected), \
        f"Expected {expected}, got {rolling['Logloss'].iloc[0]}"

# Check the window takes fights after its start and up to its end, so one exactly a window old drops out

def test_rolling_log_loss_window_edges():

    actuals = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01", "2025-09-01"]),
        "fighter1_backtesting_odds": [1.5, 2.0],
        "fighter2_backtesting_odds": [2.5, 2.0]
        })

    probabilities = np.array([
        [0.10, 0.10, 0.80],
        [0.20, 0.10, 0.70]
        ])

    target = np.array([2, 2])

    # Eight months before 2025-09-01 is 2025-01-01 exactly, so that fight sits on the excluded edge
    rolling = rolling_log_loss(actuals, probabilities, target, months=8, min_sample_size=1)

    assert rolling["Sample Size"].tolist() == [1, 1]

    # Nine months reaches back past it, so the later window holds both
    rolling = rolling_log_loss(actuals, probabilities, target, months=9, min_sample_size=1)

    assert rolling["Sample Size"].tolist() == [1, 2]

# Check windows holding too few fights are dropped

def test_rolling_log_loss_min_sample_size():

    actuals = pd.DataFrame({
        "date": pd.to_datetime(["2025-05-10", "2025-05-11", "2025-05-12"]),
        "fighter1_backtesting_odds": [1.5, 2.0, 3.0],
        "fighter2_backtesting_odds": [2.5, 2.0, 1.4]
        })

    probabilities = np.array([
        [0.10, 0.10, 0.80],
        [0.20, 0.10, 0.70],
        [0.60, 0.10, 0.30]
        ])

    target = np.array([2, 2, 0])

    rolling = rolling_log_loss(actuals, probabilities, target, months=8, min_sample_size=10)

    assert len(rolling) == 0

# Check an empty result still carries its columns, so the caller can look at it without a KeyError

def test_rolling_log_loss_empty_result_keeps_columns():

    actuals = pd.DataFrame({
        "date": pd.to_datetime(["2025-05-10"]),
        "fighter1_backtesting_odds": [1.5],
        "fighter2_backtesting_odds": [2.5]
        })

    probabilities = np.array([[0.10, 0.10, 0.80]])

    target = np.array([2])

    rolling = rolling_log_loss(actuals, probabilities, target, months=8, min_sample_size=10)

    assert len(rolling) == 0
    assert rolling.columns.tolist() == ["date", "Logloss", "X Odds Logloss", "Sample Size", "OTH Fights"]

# Check the draw and no contest count, since those are what make the metric jump about

def test_rolling_log_loss_counts_other_outcomes():

    actuals = pd.DataFrame({
        "date": pd.to_datetime(["2025-05-10", "2025-05-10", "2025-05-10"]),
        "fighter1_backtesting_odds": [1.5, 2.0, 3.0],
        "fighter2_backtesting_odds": [2.5, 2.0, 1.4]
        })

    probabilities = np.array([
        [0.10, 0.10, 0.80],
        [0.20, 0.10, 0.70],
        [0.60, 0.10, 0.30]
        ])

    # The middle fight was a draw or a no contest
    target = np.array([2, 1, 0])

    rolling = rolling_log_loss(actuals, probabilities, target, months=8, min_sample_size=1)

    assert rolling["OTH Fights"].iloc[0] == 1
