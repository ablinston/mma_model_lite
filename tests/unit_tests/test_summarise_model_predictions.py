# Load all the packages and functions needed
import numpy as np
import pandas as pd

from mma_functions.evaluation import summarise_model_predictions

# Run a test on the target odds bu simulating 1,000,000 outcomes and calculate the actual EV

def test_summarise_model_predictions():
    
    # Create a basic test dataset
    df = pd.DataFrame({
        "fighter1": ["A"],
        "fighter2": ["B"],
        "fighter1_backtesting_odds": [2.5],
        "fighter2_backtesting_odds": [1.5]
    })

    # Example probabilities: [fighter2_win, draw, fighter1_win]
    probs = np.array([[0.3, 0.1, 0.6]])

    # Run the function with original input
    result = summarise_model_predictions(
        actuals_data_df=df,
        prediction_probabilities=probs,
        min_prob_to_bet=0.5,
        min_ev_to_bet=0.0,
        min_bet_size=1,
        max_bet_size=1,
        exchange_commission=0.02,
        return_triggers=False
    )

    # Swap fighters and probs
    df_swapped = df.copy()
    df_swapped[["fighter1", "fighter2"]] = df_swapped[["fighter2", "fighter1"]]
    df_swapped[["fighter1_backtesting_odds", "fighter2_backtesting_odds"]] = df_swapped[["fighter2_backtesting_odds", "fighter1_backtesting_odds"]]
    swapped_probs = probs[:, [2, 1, 0]]

    result_swapped = summarise_model_predictions(
        actuals_data_df=df_swapped,
        prediction_probabilities=swapped_probs,
        min_prob_to_bet=0.5,
        min_ev_to_bet=0.0,
        min_bet_size=1,
        max_bet_size=1,
        exchange_commission=0.02,
        return_triggers=False
    )

    # Perform assertions
    assert result["Bet_trigger"].values[0] == "A"
    assert result_swapped["Bet_trigger"].values[0] == "A"

    assert np.isclose(result["bet_ev"].values[0], result_swapped["bet_ev"].values[0], atol=1e-6)
    assert np.isclose(result["fighter1_target_exchange_odds"].values[0], result_swapped["fighter2_target_exchange_odds"].values[0], atol=1e-6)
    assert np.isclose(result["fighter2_target_exchange_odds"].values[0], result_swapped["fighter1_target_exchange_odds"].values[0], atol=1e-6)

    print("All basic symmetry tests passed.")
    
    # Now do a test on odds that are slightly worse than probs
    
    # Create a basic test dataset
    df = pd.DataFrame({
        "fighter1": ["A"],
        "fighter2": ["B"],
        "fighter1_backtesting_odds": [1.99],
        "fighter2_backtesting_odds": [1.99]
    })

    # Example probabilities: [fighter2_win, draw, fighter1_win]
    probs = np.array([[0.5, 0, 0.5]])

    # Run the function with original input
    result = summarise_model_predictions(
        actuals_data_df=df,
        prediction_probabilities=probs,
        min_prob_to_bet=0.5,
        min_ev_to_bet=0.0,
        min_bet_size=1,
        max_bet_size=1,
        exchange_commission=0.02,
        return_triggers=False
    )

    assert result["Bet_trigger"].values[0] == "None"
