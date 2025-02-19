# This function takes the final dataset (after processing) and logs some key metrics so they can be tracked with each update

def log_final_data_stats(df, log_csv_path, config):
    
    # ## For debug
    # df = forecasting_data.copy()
    # log_csv_path = 
    
    # pdb.set_trace()
    
    # Define which key columns we want to track NAs for
    # For all these cols, don't need to put fighter1_ fighter2_ prefix, 'diff' as they will be added later
    na_cols = ['open_odds', 'best_closing_odds', 'mean_closing_odds', 'start_elo', 'reach', 'hist_times_knocked_out', 'hist_round3_body_landed_against_per_minute', 'hist_last_3_fights_round1_distance_accuracy_percent', 'hist_wins_per_fight', 'hist_opp_stance_round1_leg_accuracy_percent_against', 'hist_opp_type_allrounds_sig.str._landed_against', 'date_minus_dob', 'subs_per_attempt']

    # Start a new data frame for the final data stats
    results_df = pd.DataFrame({'Date': [datetime.today()]}, columns = ['Date'])

    for fighter in ['fighter1_', 'fighter2_']:  
        
        # Choose only fights that are after validation checks date, and where we have a fight history to avoid NAs
        test_df = df[(df['date'] >= config['start_date_validation_checks']) & (df[fighter + 'hist_fights'] > 0)][[fighter + col for col in na_cols]]
    
        # Calculate the missing values
        missing_values_percentage = test_df.isna().sum() / len(test_df) * 100
        # Calculate column means (ignoring NaNs)
        column_means = test_df.mean(skipna = True)
        
        # Convert the result into a DataFrame and transpose
        results_df = pd.concat([
                results_df,
                pd.DataFrame([missing_values_percentage.tolist() + column_means.tolist()], 
                             columns = [col + '_NA_Percent' for col in missing_values_percentage.index.tolist()] + [col + '_Mean' for col in column_means.index.tolist()])
            ],
            axis = 1)

    # Add this onto the existing csv
    existing_results = pd.read_csv(log_csv_path)
    
    results_df = pd.concat([results_df, existing_results])
    
    # Now overwrite the results
    results_df.to_csv(log_csv_path, index=False)

    return None
