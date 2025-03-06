# This function adds some random noise to odds columns using a normal distribution. It does this with a multiplicative factor

def add_noise_to_odds(df, norm_mean = 1, norm_sd = 0.1):
        
    # Get a list of the odds columns required. Don't include backtesting odds
    odds_cols = [col for col in df.columns if 'odds' in col and 'backtesting' not in col]
    
    # Create data with just the odds
    odds_df = df[odds_cols].copy()
    
    # Add randomness to the odds. Loop through so we can check each column
    for col in ['fighter1_open_odds', 'fighter1_mean_closing_odds', 'fighter1_best_closing_odds', 'fighter2_open_odds', 'fighter2_mean_closing_odds', 'fighter2_best_closing_odds']:
        if col in odds_cols:
            odds_df[col] = (odds_df[col] - 1) * np.random.normal(norm_mean, norm_sd) + 1
            
    # Calculate all odds cols
    if 'fighter1_mean_closing_odds' in odds_cols:
        odds_df['fighter1_odds_change'] = odds_df['fighter1_mean_closing_odds'] - odds_df['fighter1_open_odds']
        odds_df['fighter2_odds_change'] = odds_df['fighter2_mean_closing_odds'] - odds_df['fighter2_open_odds']
        
        # Add on a flag to say whether the odds have changed by more than 0.1
        odds_df['fighter1_odds_change_gt_0'] = np.where(odds_df['fighter1_odds_change'] < 0, -1, np.where(odds_df['fighter1_odds_change'] > 0, 1, 0))
        
        # Add on a flag to say whether the odds have changed by more than 0.1
        odds_df['fighter2_odds_change_gt_0'] = np.where(odds_df['fighter2_odds_change'] < 0, -1, np.where(odds_df['fighter2_odds_change'] > 0, 1, 0)) 
    
    # Recalulate true odds
    odds_df = add_true_odds(odds_df, 'mean_closing_odds')

    # Add diff columns
    odds_df = add_fighter_differentials(odds_df) 
    
    # Now merge the new odds with the old data
    new_df = pd.concat([
        df.drop(columns = odds_cols),
        odds_df[odds_cols]
    ], axis = 1)
    
    # check the data is the right shape
    stop_if_not(new_df.shape == df.shape, "Add noise - shape check")
    
    return new_df
    
    