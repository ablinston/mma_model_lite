
def calculate_target_lay_odds(probabilities, target_ev, commission = 0):
    
    """
    This function takes a series of outcome probabilities and a set of target EVs and calculates the LAY odds required to achieve that with a given rate of commission on winnings. NB The probabilities should be of the event happening i.e. the lay bet LOSING
    
    Parameters
    -------
    probabilities : series
    A series of probabilities of an event occuring
    
    target_ev: float64
    The EV we want to target
    
    commission: float64
    A proportion between 0 and 1 for the brokers commission on winnings
    
    
    Returns
    -------
    Series of lay odds to get to target EV in decimal format

    """
    
    return (1 - commission + commission * probabilities - target_ev) / probabilities
