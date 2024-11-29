# This function takes a series of dates from BetFightOdds and converts them to pandas datetime

def betfightodds_dates_to_datetime(dates):
    
    clean_dates = dates.apply(lambda x: sub(r'(\d+)(st|nd|rd|th)', r'\1', x))

    return pd.to_datetime(clean_dates, format='%b %d %Y')
