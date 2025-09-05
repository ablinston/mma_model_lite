def calculate_hist_opp_stance_stats(fight_stats, fighters, fight_stats_bayes_priors, data_processing_config):
    
    # pdb.set_trace()
    
    # Get the fight stats of each fighter against only specific types of fighters
    stances = fighters['stance'].str.replace(' ', '').unique().tolist()    
    
    # Merge on opponent stance
    fight_stats = fight_stats.merge(
        fighters[['fighter', 'stance']],
        left_on='opponent',
        right_on='fighter',
        how='left',
        suffixes=('', '_opp')
    )
    
    # Cleanup columns
    fight_stats = fight_stats.rename(columns = {'stance': 'opponent_stance'}).drop(columns = ['fighter_opp'])

    # Get the historic records of each fighter for each opponent stance
    fight_stats_hist_opponent_stance = aggregate_last_X_fight_data(
        fight_stats,
        id_cols = ['date', 'event', 'bout', 'fighter', 'opponent', 'opponent_stance'],
        group_by = ['fighter', 'opponent_stance'],
        prefix = 'hist_opp_stance_',
        X = 100,
        per_minute_calc = True, 
        bayes_priors_df = fight_stats_bayes_priors,
        bayes_alpha_plus_beta_rebase = data_processing_config['bayes_alpha_plus_beta_rebase_per_minute']
    ).drop(columns = ['date', 'opponent', 'opponent_stance'])

    # Check the row numbers are correct
    stop_if_not(len(fight_stats) == len(fight_stats_hist_opponent_stance), "Opponent data length check")
    
    return(fight_stats_hist_opponent_stance)