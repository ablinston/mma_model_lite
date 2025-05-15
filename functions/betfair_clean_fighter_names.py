
def betfair_clean_fighter_names(betfair_df, event_details_df, stats_df, fuzz_threshold = 70, day_threshold = 2, surname_fuzz_threshold = 90):
    
    """
    This function makes the fighter names consistent with the names on the stats data. It also adds a cleaned opponent column
    
    This requires a betfair_df that has the fighter surname and opponent surname within it.
    
    """
    
    # Find the closest name matches
    
    # pdb.set_trace()
    
    # Merge the event date onto the stats
    merged_df = stats_df[['event', 'fighter', 'bout']].drop_duplicates().merge(event_details_df[['event', 'date']], how = 'left', on = 'event')
    merged_df.rename(columns = {'date': 'event_date'}, inplace = True)
    
    if "opponent" not in betfair_df.columns:
        add_opponent_column(betfair_df)
    
    if "opponent" not in merged_df.columns:
        add_opponent_column(merged_df)
    
    # Find the closest matches to the fighter
    matched_names = betfair_df[['fighter', 'event_date']].apply(lambda row: get_best_match_with_date(
        row,
        merged_df,
        date_col='event_date',
        text_col='fighter',
        threshold=fuzz_threshold,
        max_days=day_threshold
    ), axis = 1)
    
    # Now do the same for the opponent. Do a simple rename of fighter to opponent on the UFC data as we only need names
    matched_opponent_names = betfair_df[['opponent', 'event_date']].apply(lambda row: get_best_match_with_date(
        row,
        merged_df,
        date_col='event_date',
        text_col='opponent',
        threshold=fuzz_threshold,
        max_days=day_threshold
    ), axis = 1)
     
    # Replace the fighter name where we have a close match
    betfair_df['fighter'] = np.where(matched_names.isna(), betfair_df['fighter'], matched_names)
    betfair_df['opponent'] = np.where(matched_opponent_names.isna(), betfair_df['opponent'], matched_opponent_names)

    # Now look at fighters that have no match. Try to find matches based on the fights
    betfair_df_surname_matches = betfair_df.copy()
    
    # First clean the surname columns
    betfair_df.loc[:, ['fighter_surname', 'opponent_surname']] = clean_text_columns(betfair_df[['fighter_surname', 'opponent_surname']])
    
    # Find fights on similar dates where both fighters surnames are in the bout on UFC data
    betfair_df['event_date_m1'] = betfair_df['event_date'] + timedelta(days = -1)
    betfair_df['event_date_p1'] = betfair_df['event_date'] + timedelta(days = 1)
    
    # Prepare merged data for matching
    merged_df_to_match = merged_df.rename(columns = {'event': 'event_ufc', 'fighter' : 'fighter_ufc', 'bout': 'bout_ufc', 'opponent': 'opponent_ufc', 'event_date': 'event_date_ufc'})
    
    # Merge all bouts and fighters that take place within 1 day of the UFC event date
    merged_all = pd.concat([
        # 1 day prior
        betfair_df.merge(merged_df_to_match, left_on = 'event_date_m1', right_on = 'event_date_ufc'),
        betfair_df.merge(merged_df_to_match, left_on = 'event_date', right_on = 'event_date_ufc'),
        betfair_df.merge(merged_df_to_match, left_on = 'event_date_p1', right_on = 'event_date_ufc')
        ], axis = 0)
    
    # Find matches for the bouts
    
    merged_all['strict_match'] = merged_all.apply(lambda row: 1 if (
        row['fighter_surname'] in row['bout_ufc']) and (
            row['opponent_surname'] in row['bout_ufc']) and (
                row['fighter_surname'] in row['fighter_ufc']) 
                else 0, axis = 1)
    
    # Use a fuzzy contains rather than strict
    merged_all['match'] = merged_all.apply(lambda row: 1 if (
        fuzzy_contains(row['fighter_surname'], row['bout_ufc'], surname_fuzz_threshold)) and (
            fuzzy_contains(row['opponent_surname'], row['bout_ufc'], surname_fuzz_threshold)) and (
                fuzzy_contains(row['fighter_surname'], row['fighter_ufc'], surname_fuzz_threshold)) 
                else 0, axis = 1)
    
    # Find matches for the bouts
    surname_matches_df = merged_all[merged_all['match'] == 1].sort_values('strict_match', ascending = False)[['fighter', 'bout', 'event_date', 'fighter_ufc', 'bout_ufc', 'opponent_ufc']].drop_duplicates(subset = ['fighter', 'bout', 'event_date'])
    
    # Merge back onto the original Betfair data
    betfair_merged_ufc = betfair_df.merge(surname_matches_df, on = ['fighter', 'bout', 'event_date'], how = 'left')

    # Check we have the right number of rows
    stop_if_not(len(betfair_merged_ufc) == len(betfair_df), 'Surname match row number check')
        
    # For QA, can check the non-matches
    qa = betfair_merged_ufc[(betfair_merged_ufc['fighter'] != betfair_merged_ufc['fighter_ufc']) & ~(betfair_merged_ufc['fighter_ufc'].isna()) & (matched_names.reset_index(drop = True).isna())]
    qa.to_csv('./data/qa_betfair_fuzzy_matched_ufc_names.csv')
    
    
    # Now replace fighter names where we don't have the matches from the original fuzzy match
    betfair_df['fighter'] = np.where(
        matched_names.reset_index(drop = True).isna() & 
        ~betfair_merged_ufc['fighter_ufc'].reset_index(drop = True).isna(), 
        betfair_merged_ufc['fighter_ufc'], 
        betfair_df['fighter'])
    
    betfair_df['opponent'] = np.where(
        matched_opponent_names.reset_index(drop = True).isna() & 
        ~betfair_merged_ufc['opponent_ufc'].reset_index(drop = True).isna(), 
        betfair_merged_ufc['opponent_ufc'], 
        betfair_df['opponent'])
    
    # Remove columns no longer needed
    betfair_df.drop(columns = ['fighter_surname', 'opponent_surname', 'event_date_m1', 'event_date_p1'], inplace = True)
