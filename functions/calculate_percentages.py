# This function takes the merged data and calculates percentages

def calculate_percentages(merged_data_df, fight_stats_bayes_priors):

    merged_data = merged_data_df.copy()
    
    # Calculate percentages for those columns with attempts and lands
    percent_columns = [col for col in merged_data.columns if '_attempt' in col and 'per_minute' not in col]
    landed_columns = [col for col in merged_data.columns if '_landed' in col and 'per_minute' not in col]
    percent_columns.sort()
    landed_columns.sort()
    
    # Check we have all the corresponding columns
    if ([string.replace('_attempt', '') for string in percent_columns] == [string.replace('_landed', '') for string in landed_columns]):
        print("Percent column correspodence check PASSED")
    else:
        print("Percent column correspodence check FAILED")
        raise SystemExit
    
    # Now calculate the percentages
    new_percent_cols = [col.replace('_attempt', '_accuracy_percent') for col in percent_columns]
    merged_data[landed_columns].index = merged_data[percent_columns].index
    
    # Shapes where strange so had to use numpy
    new_data = pd.DataFrame(safe_divide_np(merged_data[landed_columns].to_numpy(), merged_data[percent_columns].to_numpy(), zero_zero_fill = np.nan), columns = new_percent_cols)
    
    # Remove percentages for sample sizes below minimum thresholds, excluding takedowns
    percent_columns_no_td = [col for col in percent_columns if "_td_" not in col]
    new_percent_cols_no_td = [col for col in new_percent_cols if "_td_" not in col]
    new_data.reset_index(drop = True, inplace = True)
    new_data[new_percent_cols_no_td] = np.where(merged_data[percent_columns_no_td] < config['min_attempts'], np.nan, new_data[new_percent_cols_no_td])
    
    # Combine with data
    merged_data = pd.concat([merged_data, new_data], axis = 1)

    # pdb.set_trace()

    ##########################################
    # Now calculate the bayes percentages
    
    
    if fight_stats_bayes_priors is not None:
        
        # Add on the bayes variables
        merged_data_bayes = merged_data.merge(fight_stats_bayes_priors, on = ['event', 'bout'], how = 'left')
        
        # Find what numerators and denominators are needed
        bayes_numerators = [col.replace('fighter1_', 'bayes_total_').replace('fighter2_', 'bayes_total_').replace('hist_', '').replace('opp_type_', '').replace('opp_stance_', '').replace('_against', '').replace('_last_3_fights', '') for col in landed_columns]
        
        bayes_denominators = [col.replace('fighter1_', 'bayes_total_').replace('fighter2_', 'bayes_total_').replace('hist_', '').replace('opp_type_', '').replace('opp_stance_', '').replace('_against', '').replace('_last_3_fights', '') for col in percent_columns]
        
        # Check these are in the bayes data
        stop_if_not(all(item in merged_data_bayes.columns for item in bayes_numerators + bayes_denominators), "Bayes priors check in data")
        
        # Calculate the bayes percentage accuracies
        new_bayes_percent_cols = [col.replace('_attempt', '_bayes_accuracy_percent') for col in percent_columns]
        
        new_data = pd.DataFrame(safe_divide_np(
            merged_data_bayes[landed_columns].to_numpy() + merged_data_bayes[bayes_numerators].to_numpy(), 
            merged_data_bayes[percent_columns].to_numpy() + merged_data_bayes[bayes_denominators].to_numpy(), 
            zero_zero_fill = np.nan
            ), 
            columns = new_bayes_percent_cols
        )
        
        # Check there are non > 0
        stop_if_not(new_data[new_data>1].sum().sum() == 0, "Bayes percentages less than 1")
        stop_if_not(new_data[new_data<0].sum().sum() == 0, "Bayes percentages greater than 0")
        
        # Merge the data
        merged_data = pd.concat([merged_data, new_data], axis = 1)
    
    # Check we have the correct number of rows
    stop_if_not(len(merged_data_df) == len(merged_data), "Dataset length")
    
    del([new_data, percent_columns, landed_columns, new_percent_cols])
    
    return merged_data