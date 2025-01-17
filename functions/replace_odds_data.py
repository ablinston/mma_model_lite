# This function replaces the odds data on the final dataset with new cleaned odds data

def replace_odds_data(final_data, cleaned_odds_path):
    
    ## Debug
    final_data = forecasting_data.copy()
    cleaned_odds_path = './data/' + config['fighter_odds_file_name'].replace('.parquet', '_clean.parquet')
    ###
    
    # Read in the new odds data
    fight_odds = fp.ParquetFile(cleaned_odds_path).to_pandas()
    fight_data = final_data.copy()
    
    # Rename the odds cols
    fight_odds = process_odds_data(fight_odds)
    
    # Get all the odds columns
    odds_cols = [col for col in final_data.columns if 'odd' in col]
    
    # Merged with the fight data with the odds columns removed
    fight_data = merge_df_both_fighters(
        left_df =  final_data[[col for col in final_data.columns if col not in odds_cols]],
        df_list = [
            fight_odds
        ]
    ).reset_index(drop = True)
    
    # Add on the new fighter1 and fighter2 odds
    fighter1_data = fight_odds.copy().add_prefix('fighter1_')
    fighter2_data = fight_odds.copy().add_prefix('fighter2_')
    
    # Merge the fighter data
    
    return 

