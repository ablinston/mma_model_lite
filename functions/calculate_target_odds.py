
def calculate_target_odds(probabilities, target_ev, commission = 0):
    
    """
    This function takes a series of outcome probabilities and a set of target EVs and calculates the odds required to achieve that with a given rate of commission on winnings
    
    Parameters
    -------
    probabilities : series
    A series of probabilities of events
    
    target_ev: float64
    The EV we want to target
    
    commission: float64
    A proportion between 0 and 1 for the brokers commission on winnings
    
    
    Returns
    -------
    Series of odds in decimal format

    """
    
    return (target_ev - commission * probabilities + 1) / (probabilities * (1 - commission))