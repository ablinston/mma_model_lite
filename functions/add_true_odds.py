# This function adds true odds based on a column suffix
def add_true_odds(df, col_suffix = 'mean_closing_odds'):
    
    # Change odds into true odds
    df[['fighter1_true_closing_odds', 'fighter2_true_closing_odds']] = calculate_true_odds(df, 'fighter1_' + col_suffix, 'fighter2_' + col_suffix)
    df[['fighter1_true_open_odds', 'fighter2_true_open_odds']] = calculate_true_odds(df, 'fighter1_open_odds', 'fighter2_open_odds')

    # Recalculate change
    df['fighter1_true_odds_change'] = df['fighter1_true_closing_odds'] - df['fighter1_true_open_odds']
    df['fighter2_true_odds_change'] = df['fighter2_true_closing_odds'] - df['fighter2_true_open_odds']
    
    return df
