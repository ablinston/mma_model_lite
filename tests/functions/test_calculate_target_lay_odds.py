# Load all the packages and functions needed
exec(open('global.py').read())

# Run a test on the target odds bu simulating 1,000,000 outcomes and calculate the actual EV

def test_calculate_target_lay_odds():
    
    np.random.seed(42)

    # Setup
    n = int(2e6)
    p = 0.4  # probability of event happening (i.e., lay bet loses)
    target_ev = 0.05
    commission = 0.05
    probabilities = pd.Series([p] * int(n))

    # Calculate lay odds that should give target EV
    lay_odds = calculate_target_lay_odds(probabilities, target_ev, commission)

    # Simulate outcomes: 1 if event happens (lay bet loses), 0 if it doesn't (lay bet wins)
    outcomes = np.random.binomial(1, p, n)

    # Compute profit:
    # - If event happens (lay bet loses): lose (lay_odds - 1)
    # - If event doesn't happen (lay bet wins): win 1 unit (commission charged on this)
    profits = np.where(
        outcomes == 1,
        -(lay_odds - 1),
        1 * (1 - commission)
    )

    mean_profit = profits.mean()

    # Assert the average profit is close to the target EV
    assert np.isclose(mean_profit, target_ev, atol=0.001), f"Expected EV ≈ {target_ev}, got {mean_profit}"

    p = 0.8  # probability of event happening (i.e., lay bet loses)
    target_ev = 0.15
    commission = 0.0
    probabilities = pd.Series([p] * int(n))

    # Calculate lay odds that should give target EV
    lay_odds = calculate_target_lay_odds(probabilities, target_ev, commission)

    # Simulate outcomes: 1 if event happens (lay bet loses), 0 if it doesn't (lay bet wins)
    outcomes = np.random.binomial(1, p, n)

    # Compute profit:
    # - If event happens (lay bet loses): lose (lay_odds - 1)
    # - If event doesn't happen (lay bet wins): win 1 unit (commission charged on this)
    profits = np.where(
        outcomes == 1,
        -(lay_odds - 1),
        1 * (1 - commission)
    )

    mean_profit = profits.mean()

    # Assert the average profit is close to the target EV
    assert np.isclose(mean_profit, target_ev, atol=0.001), f"Expected EV ≈ {target_ev}, got {mean_profit}"
