# This function takes american style odds and converts them to decimals

def american_odds_to_decimal(american_odds):
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1