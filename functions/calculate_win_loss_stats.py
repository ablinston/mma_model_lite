# This function takes the fighter results data (obtained after converting the fight results to be one row per fighter, not per fight) and derives various columns for win/loss stats and adds them to the data

def calculate_win_loss_stats(fighter_results, finishes = ['KO/TKO', 'Submission'], last_x_fights = 10):

    # pdb.set_trace()    

    ###########################################################################################
    # Calculate number of previous wins/losses
    win_loss_stats = fighter_results.copy()
    win_loss_stats['date_win'] = np.where(win_loss_stats['result'] == 'W', win_loss_stats['date'], np.datetime64('NaT'))
    win_loss_stats['date_loss'] = np.where(win_loss_stats['result'] == 'L', win_loss_stats['date'], np.datetime64('NaT'))

    win_loss_stats.sort_values(by = ['fighter', 'date'], inplace = True)

    # Calculate date of last win and loss
    win_loss_stats['date_last_win'] = win_loss_stats.groupby(['fighter'])['date_win'].shift(1)
    win_loss_stats['date_last_loss'] = win_loss_stats.groupby(['fighter'])['date_loss'].shift(1)
    # Fill in the places where there is NAs because of losses and wins between each other
    win_loss_stats['date_last_win'] = win_loss_stats.groupby(['fighter'])['date_last_win'].ffill()
    win_loss_stats['date_last_loss'] = win_loss_stats.groupby(['fighter'])['date_last_loss'].ffill()

    # Count number of historic wins and losses
    win_loss_stats['win'] = np.where(win_loss_stats['result'] == "W", 1, 0)
    win_loss_stats['loss'] = np.where(win_loss_stats['result'] == "L", 1, 0)
    win_loss_stats['hist_wins'] = win_loss_stats.groupby(['fighter'])['win'].cumsum() - win_loss_stats['win']
    win_loss_stats['hist_losses'] = win_loss_stats.groupby(['fighter'])['loss'].cumsum() - win_loss_stats['loss']
    win_loss_stats['win_finish'] = np.where((win_loss_stats['result'] == "W") & (win_loss_stats['method'].isin(finishes)), 1, 0)
    win_loss_stats['loss_finish'] = np.where((win_loss_stats['result'] == "L") & (win_loss_stats['method'].isin(finishes)), 1, 0)
    
    # Begin a count of finishes to be populated as we go through the methods
    win_loss_stats['hist_wins_finishes'] = 0
    win_loss_stats['hist_losses_finishes'] = 0
    
    # Get a unique list of methods
    unique_method_list = win_loss_stats['method'].dropna().unique()
    
    # Now calculate the wins and losses by method
    for method in unique_method_list:
        # Create a binary flag to say whether a win or loss was by a method
        win_loss_stats['W_' + method] = (win_loss_stats['method'] == method).astype(int) * win_loss_stats['win']
        win_loss_stats['L_' + method] = (win_loss_stats['method'] == method).astype(int) * win_loss_stats['loss']
        
        # Now count the historical wins and losses by that method
        win_loss_stats['hist_wins_' + method] = win_loss_stats.groupby(['fighter'])['W_' + method].cumsum() - win_loss_stats['W_' + method]
        win_loss_stats['hist_losses_' + method] = win_loss_stats.groupby(['fighter'])['L_' + method].cumsum() - win_loss_stats['L_' + method]
        
        # If this method is a finish, then add to the finish count
        if method in finishes:
            win_loss_stats['hist_wins_finishes'] += win_loss_stats['hist_wins_' + method]
            win_loss_stats['hist_losses_finishes'] += win_loss_stats['hist_losses_' + method]

    # Now calculate the win rate for the last X fights
    for x in range(1, (last_x_fights + 1)):
        win_loss_stats[f'hist_win_percent_{x}_fights'] = (
            win_loss_stats.groupby('fighter')['win']
            .transform(lambda s: s.shift().rolling(x, min_periods=x).mean())
        )
        
        win_loss_stats[f'hist_win_finish_percent_{x}_fights'] = (
            win_loss_stats.groupby('fighter')['win_finish']
            .transform(lambda s: s.shift().rolling(x, min_periods=x).mean())
        )
        
        win_loss_stats[f'hist_loss_finish_percent_{x}_fights'] = (
            win_loss_stats.groupby('fighter')['loss_finish']
            .transform(lambda s: s.shift().rolling(x, min_periods=x).mean())
        )

    # Add a flag to say whether a fighter has moved a weightclass since their last 2 fights (adding it into the win loss data because it has it in the format required already)
    win_loss_stats['weightclass_m1'] = win_loss_stats.groupby('fighter')['weightclass'].shift(1)
    win_loss_stats['weightclass_m2'] = win_loss_stats.groupby('fighter')['weightclass'].shift(2)

    win_loss_stats['moved_weightclass'] = np.where(
        (~win_loss_stats['weightclass_m2'].isna()) & 
        (~win_loss_stats['weightclass_m2'].isna()) &
        (win_loss_stats['weightclass_m1'] == win_loss_stats['weightclass_m2']) & 
        (win_loss_stats['weightclass_m2'] != win_loss_stats['weightclass']),
        1,
        0
    )

    ############################
    # Add per fight stats
    numerator_strings = ['hist_wins', 'hist_losses']

    # Calculate denominator
    win_loss_stats['hist_fights'] = win_loss_stats.groupby('fighter').cumcount()
    
    # Get the column names
    cols = []
    for i in range(0, len(numerator_strings)):
        cols = cols + [col for col in win_loss_stats.columns if numerator_strings[i] in col]
    
    new_cols = {col: col + '_per_fight' for col in cols}

    win_loss_stats = win_loss_stats.assign(
        **{new_col: win_loss_stats[old_col] / win_loss_stats['hist_fights'] 
           for old_col, new_col in new_cols.items()}
    )

    # Deal with any infinities
    win_loss_stats.loc[:, [col + '_per_fight' for col in cols]] = win_loss_stats[[col + '_per_fight' for col in cols]].replace([np.inf, -np.inf], np.nan)

    ############################
    # Clean up columns
    win_loss_stats.drop(columns = ['hist_fights', 'weightclass', 'weightclass_m1', 'weightclass_m2', 'method', 'round', 'time format', 'date', 'rounds', 'result', 'win', 'loss', 'win_finish', 'loss_finish', 'date_win', 'date_loss'] + ('W_' + unique_method_list).tolist() + ('L_' + unique_method_list).tolist(), inplace = True)

    # Merge back onto the fighter results data
    output_data = fighter_results.merge(win_loss_stats, on = ['event', 'bout', 'fighter'], how = 'left')
    
    # Do a check
    if len(output_data) == len(fighter_results):
        return output_data.copy()
    else:
        print("Error: win loss stats couldn't be merged")
        return None