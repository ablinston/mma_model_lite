def odds_to_probabilities(odds_fighter1, odds_fighter2, normalise = True):
    """
    Convert the closing odds of two fighters into the probabilities of win, loss, or draw.

    Parameters:
    - odds_fighter1 (float): Closing odds for fighter 1 (decimal).
    - odds_fighter2 (float): Closing odds for fighter 2 (decimal).

    Returns:
    - probabilities (dict): Dictionary containing probabilities for fighter 1 win, fighter 2 win, and draw.
    """
    # Calculate implied probabilities
    prob_fighter1 = 1 / odds_fighter1
    prob_fighter2 = 1 / odds_fighter2
    
    # Total implied probability
    total_prob = prob_fighter1 + prob_fighter2

    if normalise:
        # Calculate normalized probabilities
        prob_fighter1 = prob_fighter1 / total_prob
        prob_fighter2 = prob_fighter2 / total_prob

    # Assuming there is a draw, calculate the draw probability (just for demonstration)
    # This could be based on additional logic or odds if available
    prob_draw = 1 - (prob_fighter1 + prob_fighter2)

    # Create a DataFrame to hold the results
    probabilities = pd.DataFrame({
        'L/W': np.round(prob_fighter2, 8),
        'OTH': np.round(prob_draw, 8),
        'W/L': np.round(prob_fighter1, 8)
    })

    return probabilities

