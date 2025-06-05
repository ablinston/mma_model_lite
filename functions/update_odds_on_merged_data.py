# This function takes new cleaned data and adds the odds to the existing merged data, recalculating differentials where needed

# It returns the data frame with new odds data

def update_odds_on_merged_data(
        cleaned_fight_odds_data_path: str,
        cleaned_event_odds_data_path: str,
        cleaned_betfair_odds_data_path: str,
        final_data_folder: str,
        average_odds_model_path,
        config,
        data_processing_config
        ):
    
    # #########
    # cleaned_fight_odds_data_path = './data/' + config['fighter_odds_file_name'].replace('.parquet', '_clean.parquet')
    # cleaned_event_odds_data_path = './data/' + config['event_odds_file_name'].replace('.parquet', '_clean.parquet')
    # final_data_folder = './data/final_dataset/'
    # average_odds_model_path = './results/average_odds_model.joblib'
    # #########
    
    # Load in the datasets
    fight_odds = fp.ParquetFile(cleaned_fight_odds_data_path).to_pandas()
    event_odds = fp.ParquetFile(cleaned_event_odds_data_path).to_pandas()
    betfair_odds = fp.ParquetFile(cleaned_betfair_odds_data_path).to_pandas()
    merged_data = pq.ParquetDataset(final_data_folder).read().to_pandas()
    average_odds_model = joblib.load(average_odds_model_path)
    
    # Save original shape for later
    original_shape = merged_data.shape
    
    # pdb.set_trace()
    
    # Assume that the event and bout have been added correctly as part of the data cleaning process.
    # Process the data
    fight_odds = process_odds_data(fight_odds, event_odds, betfair_odds, odds_scrape_config, average_odds_model)
    
    # Remove the columns that will be replaced
    cols_to_remove = [col for col in merged_data.columns if col.replace('fighter1_', '').replace('fighter2_', '') in fight_odds.columns]
    cols_to_remove = [col for col in cols_to_remove if col not in ['event', 'bout', 'date']]

    # Remove old columns
    merged_data.drop(columns = cols_to_remove, inplace = True)
    
    # Merge on the new odds
    merged_data = merge_df_both_fighters(
        left_df =  merged_data,
        df_list = [
            fight_odds
        ]
    ).reset_index(drop = True)

    # Recalculate all true odds
    merged_data = calculate_all_true_odds_columns(merged_data, impute_missing_betfair = data_processing_config['impute_missing_betfair'], odds_suffix_for_true_odds = data_processing_config['odds_suffix_for_true_odds'], impute_one_sided_missing_odds = data_processing_config['impute_one_sided_missing_odds'])

    # Add what odds are to be used in model training and backtesting
    merged_data['fighter1_backtesting_odds'] = merged_data['fighter1_' + model_hyper_params['odds_to_use']].copy()
    merged_data['fighter2_backtesting_odds'] = merged_data['fighter2_' + model_hyper_params['odds_to_use']].copy()
    
    # Recalculate differentials
    merged_data = add_fighter_differentials(merged_data)
    
    # Check on data
    cols_that_should_be_in_original = [col for col in merged_data.columns if 'notformodel_true_odds_asat_datetime' not in col]
    stop_if_not(len(merged_data) == original_shape[0])
    stop_if_not(len(cols_that_should_be_in_original) == original_shape[1])

    return merged_data
