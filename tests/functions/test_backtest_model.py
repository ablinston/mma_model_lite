# Load all the packages and functions needed
exec(open('global.py').read())

def test_backtest_model():
    # Create dummy actuals data
    actuals_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=3),
        'fighter1': ['A', 'B', 'C'],
        'fighter2': ['D', 'E', 'F'],
        'outcome': ['W/L', 'L/W', 'W/L'],
        'fighter1_backtesting_odds': [2.0, 3.0, 5],
        'fighter2_backtesting_odds': [1.5, 1.3, 1.1],
    })

    # Scenario 1: Perfect predictions
    perfect_predictions = np.array([
        [0.0, 0.0, 1.0],  # Fighter 1 wins
        [1.0, 0.0, 0.0],  # Fighter 2 wins
        [0.0, 0.0, 1.0],  # Fighter 1 wins
    ])

    # Scenario 2: Completely wrong predictions
    wrong_predictions = np.array([
        [1.0, 0.0, 0.0],  # Fighter 1 predicted to lose, but won
        [0.0, 0.0, 1.0],  # Fighter 2 predicted to lose, but won
        [1.0, 0.0, 0.0],  # Fighter 1 predicted to lose, but won
    ])

    # Scenario 3: Predictions match odds exactly
    odds_match_predictions = np.array([
        [0.33, 0.34, 0.33],
        [0.25, 0.24, 0.51],
        [0.25, 0.24, 0.51],
    ])

    # Common parameters for backtesting
    breakdown_cols = []
    min_prob_to_bet = 0.5
    min_ev_to_bet = -1

    # Test each scenario
    results_perfect, tmp = backtest_model(
        actuals_data.copy(), perfect_predictions, breakdown_cols, min_prob_to_bet, min_ev_to_bet, output_data_for_debug = False
    )
    results_wrong, tmp = backtest_model(
        actuals_data.copy(), wrong_predictions, breakdown_cols, min_prob_to_bet, min_ev_to_bet, output_data_for_debug = False
    )
    results_odds_match, tmp = backtest_model(
        actuals_data.copy(), odds_match_predictions, breakdown_cols, min_prob_to_bet, min_ev_to_bet, output_data_for_debug = False
    )

    results_odds_match_non_constant, tmp = backtest_model(
        actuals_data.copy(), odds_match_predictions, breakdown_cols, min_prob_to_bet, min_ev_to_bet, min_bet_size = 0.1, max_bet_size = 0.5, constant_bets = False, output_data_for_debug = False
    )
    
    # Validate results
    # Scenario 1: Perfect predictions should result in maximum profit and all bets placed
    assert results_perfect['bets'].iloc[0] == len(actuals_data)  # All bets placed
    assert results_perfect['wins'].iloc[0] == results_perfect['bets'].iloc[0]  # All wins
    assert results_perfect['profit_loss'].iloc[0] == 5.3 # Win amount is odds

    # Scenario 2: Completely wrong predictions should result in maximum loss and no wins
    assert results_wrong['bets'].iloc[0] == len(actuals_data)  # All bets placed
    assert results_wrong['wins'].iloc[0] == 0  # No wins
    assert results_wrong['profit_loss'].iloc[0] == -results_wrong['bets'].iloc[0]  # Negative profit

    # Scenario 3: Predictions matching odds should place 2 bets but lose 1
    assert results_odds_match['bets'].iloc[0] == 2  # 2 bets placed
    assert results_odds_match['wins'].iloc[0] == 1  # 2 bets placed
    assert results_odds_match['profit_loss'].iloc[0] == 3 # Win amount is odds

    # Scenario 3B: Predictions matching odds, same as 3 but with proportional bets
    assert results_odds_match_non_constant['bets'].iloc[0] == 2  # 2 bets placed
    assert results_odds_match_non_constant['wins'].iloc[0] == 1  # 2 bets placed
    assert np.round(results_odds_match_non_constant['profit_loss'].iloc[0], 2) == 603.88 # Calculated by hand
