
def betfair_clean_fighter_names(betfair_df, event_details_df, stats_df, fuzz_threshold = 70, day_threshold = 2):
    
    """
    This function makes the fighter names consistent with the names on the stats data
    
    
    """
    
    # Find the closest name matches
    
    # pdb.set_trace()
    
    # Merge the event date onto the stats
    merged_df = stats_df[['event', 'fighter']].merge(event_details_df[['event', 'date']], how = 'left', on = 'event')
    merged_df.rename(columns = {'date': 'event_date'}, inplace = True)
    
    # Find the closest matches to the fighter
    matched_names = betfair_df[['fighter', 'event_date']].apply(lambda row: get_best_match_with_date(
        row,
        merged_df,
        date_col='event_date',
        text_col='fighter',
        threshold=fuzz_threshold,
        max_days=day_threshold
    ), axis = 1)
     
    # Replace the fighter name where we have a close match
    betfair_df['fighter'] = np.where(matched_names.isna(), betfair_df['fighter'], matched_names)

    