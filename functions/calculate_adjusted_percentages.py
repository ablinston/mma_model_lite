# This function takes the fight stats and fight stats history and calculates percentages (with bayes if needed)
# and adjusted percentages (with bayes if needed)

# It returns the original data frame with the new percentage columns added

def calculate_adjusted_percentages(
        fight_stats, 
        fight_stats_hist, # This must contain both against and normal versions of stats
        fight_stats_bayes_priors,
        data_processing_config):

    # pdb.set_trace()
    
    # Calculate percentages for each fight
    fight_stats_percentages = calculate_percentages(
        fight_stats,
        fight_stats_bayes_priors, 
        bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_percent']
    )
    
    # Calculate percentages for each fighters history up to that point
    fight_stats_hist_percentages = calculate_percentages(
        fight_stats_hist,
        fight_stats_bayes_priors, 
        bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_percent']
    )
    
    # Add on adjusted percentages if required
    if data_processing_config['add_adj_percentage_columns']:
        
        fight_stats_percentages = add_adjusted_stats(
            stats_df = fight_stats_percentages, 
            hist_df = fight_stats_hist_percentages,
            columns_with = ['accuracy_percent'],
            exclude_columns_with = ['bayes']
            )
    
    # Add on adjusted bayes percentages if required
    if data_processing_config['add_adj_percentage_bayes_columns']:
        
        fight_stats_percentages = add_adjusted_stats(
            stats_df = fight_stats_percentages, 
            hist_df = fight_stats_hist_percentages,
            columns_with = ['bayes_accuracy_percent'],
            exclude_columns_with = ['_adj']
            )
        
    # Check all the row numbers are correct
    stop_if_not(
        len(fight_stats) == len(fight_stats_percentages),
        "Percentages data check")
        
    del(fight_stats_hist_percentages)
    
    return fight_stats_percentages