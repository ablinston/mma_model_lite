# This function takes a dataframe with two odds columns and normalises them so that there is no spread or gap between them. This creates the true odds


def calculate_true_odds(df, col1, col2):
    """
    Calculates the true odds in decimal form from two columns of decimal odds.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        col1 (str): The name of the first column with decimal odds.
        col2 (str): The name of the second column with decimal odds.

    Returns:
        pd.DataFrame: A DataFrame with the true odds for the specified columns.
    """
    # Calculate implied probabilities
    implied_prob_col1 = 1 / df[col1]
    implied_prob_col2 = 1 / df[col2]

    # Calculate the total implied probability (overround)
    overround = implied_prob_col1 + implied_prob_col2

    # Calculate true probabilities
    true_prob_col1 = implied_prob_col1 / overround
    true_prob_col2 = implied_prob_col2 / overround

    # Convert true probabilities back to decimal odds
    true_odds_col1 = 1 / true_prob_col1
    true_odds_col2 = 1 / true_prob_col2

    # Replace the original odds
    df[col1] = true_odds_col1
    df[col2] = true_odds_col2

    # Create and return a new DataFrame with the results
    return df
