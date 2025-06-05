def back_odds_to_lay_odds(back_odds_series):
    """
    
    A function that takes in a series or array of back odds for an outcome 1 and converts it into lay odds for outcome 2. ONLY WORKS FOR A 2 OUTCOME EVENT


    """
    
    
    return back_odds_series / (back_odds_series - 1)
