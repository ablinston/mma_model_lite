# This function cleans the date columns on the fighters betting odds data. It takes in an odds data frame

# and returns a cleaned data frame

# AB: DO NOT USE THIS FUNCTION. THERE ARE LOTS OF EVENTS CALLED JUST 'UFC' AND THIS JUST AMALGAMATES THEM ALL TOGETHER

def clean_odds_dates(odds_df):
    
    cleaned_dates_df = odds_df.groupby('event').agg({'date': safe_mode})
    
    # Now merge this back onto the odds data
    odds_df.drop('date', axis=1, inplace = True)
    odds_df2 = odds_df.merge(cleaned_dates_df, on = 'event')

    del(cleaned_dates_df)
    
    return odds_df
