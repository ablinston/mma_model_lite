# Load all the packages and functions needed
exec(open('global.py').read())
# # Load what answers are expected
# test_aggregate_last_X_fight_data_correct = fp.ParquetFile('./tests/data/test_aggregate_last_X_fight_data_correct.parquet').to_pandas()



# Test function
def test_calculate_percentages():
    
    merged_data_df = pd.DataFrame({
        "event": ["E1", "E2", "E3", "E4"],
        "bout": ["B1", "B2", "B3", "B4"],
        "fighter": ["F1", "F2", "F1", "F2"],
        "fighter1_punch_attempt": [10, 20, 15, 30],
        "fighter1_punch_landed": [5, 10, 7, 15],
        "fighter1_kick_attempt": [8, 16, 12, 24],
        "fighter1_kick_landed": [4, 8, 6, 12]
    })
    
    fight_stats_bayes_priors = pd.DataFrame({
        "event": ["E1", "E2", "E3", "E4"],
        "bout": ["B1", "B2", "B3", "B4"],
        "bayes_total_punch_attempt": [20, 20, 30, 30],
        "bayes_total_punch_landed": [2, 2, 3, 3],
        "bayes_total_kick_attempt": [10, 10, 20, 20],
        "bayes_total_kick_landed": [1, 1, 2, 2]
    })
    
    result = calculate_percentages(merged_data_df, fight_stats_bayes_priors, bayes_alpha_plus_beta_rebase = 10)

    # Ensure the function runs and returns a DataFrame
    assert isinstance(result, pd.DataFrame), "Function must return a DataFrame"

    # Check row count consistency
    assert len(result) == len(merged_data_df), "Row count should remain the same"

    # Expected accuracy values
    expected_punch_acc = [0.5, 0.5, 0.4667, 0.5]  # landed/attempt
    expected_kick_acc = [0.5, 0.5, 0.5, 0.5]

    # Validate computed percentages (rounded for precision)
    assert np.allclose(result["fighter1_punch_accuracy_percent"], expected_punch_acc, atol=1e-3)
    assert np.allclose(result["fighter1_kick_accuracy_percent"], expected_kick_acc, atol=1e-3)

    # Expected Bayesian accuracy values
    expected_punch_bayes = [(5+2*10/20)/(10+10), (10+2*10/20)/(20+10), (7+3*10/30)/(15+10), (15+3*10/30)/(30+10)]
    expected_kick_bayes = [(4+1)/(8+10), (8+1)/(16+10), (6+2*10/20)/(12+10), (12+2*10/20)/(24+10)]

    # Validate Bayesian computed percentages
    assert np.allclose(result["fighter1_punch_bayes_accuracy_percent"], expected_punch_bayes, atol=1e-3)
    assert np.allclose(result["fighter1_kick_bayes_accuracy_percent"], expected_kick_bayes, atol=1e-3)

    # Ensure all percentages are between 0 and 1
    for col in ["fighter1_punch_accuracy_percent", "fighter1_kick_accuracy_percent", 
                "fighter1_punch_bayes_accuracy_percent", "fighter1_kick_bayes_accuracy_percent"]:
        assert (result[col] >= 0).all() and (result[col] <= 1).all(), f"Values in {col} must be between 0 and 1"


