def betfair_bout_get_chosen_odds(bout_df, time_cutoff):
    """
    This function takes a data frame of bets for a given bout and returns the chosen odds to be used in modelling
    
    Args:
        df (DataFrame): a data frame from the betfair_stream_to_df function
        time_cutoff (time): A time cut off in the lead up to the event
    
    Returns:
        DataFrame: giving the bout information and the chosen betfair odds
    
    """

    # pdb.set_trace()
    
    df = bout_df.copy().sort_values('GMT_datetime')
    
    # Split out the datetime to allow easy analysis
    df['bet_date'] = df['GMT_datetime'].dt.date
    df['bet_time'] = df['GMT_datetime'].dt.time
    df['event_date'] = df['bet_date'].max()

    # Check if we're passed the cutoff time on the given day when the bout starts
    if datetime.strptime(time_cutoff, "%H:%M").time() < df.iloc[-1]["bet_time"]:
        cutoff_datetime = datetime.strptime(str(df['event_date'].iloc[0]) + ":" + time_cutoff, "%Y-%m-%d:%H:%M")
    else:
        cutoff_datetime = datetime.strptime(str(df['event_date'].iloc[0] - timedelta(days = 1)) + ":" + time_cutoff, "%Y-%m-%d:%H:%M")
    
    # Now filter the data after the cutoff time
    df = df[df['GMT_datetime'] < cutoff_datetime]
    
    # And finally only keep one record per fighter
    output_df = df.groupby('fighter').agg({'bout': 'first', 'event_date': 'first', 'last_price_traded': 'last'}).reset_index()

    output_df.rename(columns = {'last_price_traded': 'betfair_exch_odds'}, inplace = True)

    # Check the length is 2 so we have both fighters
    if len(output_df) != 2:
        print("Error: Data is not the length expected")
        pdb.set_trace()

    return output_df