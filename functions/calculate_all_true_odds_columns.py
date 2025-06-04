def calculate_all_true_odds_columns(merged_data, impute_missing_betfair, odds_suffix_for_true_odds, impute_one_sided_missing_odds):
    
    """
    This function calculates all the required true odds columns on the processed merged dataset
    
    Parameters
    -------
    merged_data : data frame
    merged data with fighter1 and fighter2 columns
    
    impute_missing_betfair: bool
    whether missing betfair odds will be imputed with best_closing_odds
    
    odds_suffix_for_true_odds: string
    the column suffix for the odds column to be used in calculating true odds
    
    model_hyper_params: dict
    the model hyperparameters dictionary
    
    Returns
    -------
    Series of odds in decimal format
    
    """
    
    # pdb.set_trace()
    
    # Impute missing odds if they are missing for one fighter but not the other
    if impute_one_sided_missing_odds:
        for odds in ['betfair_exch_odds', 'best_closing_odds', 'open_odds']:
            merged_data['fighter1_' + odds] = np.where(merged_data['fighter1_' + odds].isna(), merged_data['fighter2_' + odds] / (merged_data['fighter2_' + odds] - 1), merged_data['fighter1_' + odds])
            merged_data['fighter2_' + odds] = np.where(merged_data['fighter2_' + odds].isna(), merged_data['fighter1_' + odds] / (merged_data['fighter1_' + odds] - 1), merged_data['fighter2_' + odds])
    
    # Impute Betfair odds if required
    if impute_missing_betfair:
        merged_data['fighter1_betfair_exch_odds_imputed'] = np.where(merged_data['fighter1_betfair_exch_odds'].isna(), merged_data['fighter1_best_closing_odds'], merged_data['fighter1_betfair_exch_odds'])
        merged_data['fighter2_betfair_exch_odds_imputed'] = np.where(merged_data['fighter2_betfair_exch_odds'].isna(), merged_data['fighter2_best_closing_odds'], merged_data['fighter2_betfair_exch_odds'])
        
        # If we're using these in true odds, use the imputed
        if odds_suffix_for_true_odds == 'betfair_exch_odds':
            odds_suffix_for_true_odds = 'betfair_exch_odds_imputed'
    
    # Change odds into true odds
    merged_data[['fighter1_true_closing_odds', 'fighter2_true_closing_odds']] = calculate_true_odds(merged_data, 'fighter1_' + odds_suffix_for_true_odds, 'fighter2_' + odds_suffix_for_true_odds)
    merged_data[['fighter1_true_open_odds', 'fighter2_true_open_odds']] = calculate_true_odds(merged_data, 'fighter1_open_odds', 'fighter2_open_odds')
    
    # Recalculate change
    merged_data['fighter1_true_odds_change'] = merged_data['fighter1_true_closing_odds'] - merged_data['fighter1_true_open_odds']
    merged_data['fighter2_true_odds_change'] = merged_data['fighter2_true_closing_odds'] - merged_data['fighter2_true_open_odds']
    
    # Get the true odds scrape date
    for fighter in ['fighter1', 'fighter2']:
        if 'betfair_exch_odds' in odds_suffix_for_true_odds:
            merged_data[fighter + '_notformodel_true_odds_asat_datetime'] = np.where(
                merged_data[fighter + '_betfair_exch_odds'].isna(),
                merged_data[fighter + '_notformodel_date_fighter_odd_scrape'],
                merged_data[fighter + '_notformodel_betfair_odds_datetime'])
        else:
            merged_data[fighter + '_notformodel_true_odds_asat_datetime'] = merged_data[fighter + '_notformodel_date_fighter_odd_scrape']
    
    return merged_data