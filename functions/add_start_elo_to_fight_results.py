def add_start_elo_to_fight_results(fight_results, fight_stats, config, elo_config):
    
    """
    This function takes the fight results data and adds the start ELO to all the bouts
    
    """
    # pdb.set_trace()
    
    # Get the starting data
    start_elo, elo_fight_data = prep_stats_for_elo_calc(fight_results, fight_stats, config, start_date = '2000-01-01', remove_nas = False)

    # Dderive the ELO for the params in the config
    elo_data, elo_fight_data_welo = derive_start_elos(start_elo, elo_fight_data, elo_config)

    # Create two versions to be added onto the results data
    fighter1_elo = elo_data.copy().to_pandas().drop(['fight_number'], axis = 1).rename(columns = {'fighter': 'fighter1', 'start_elo': 'fighter1_start_elo'}).sort_values(['date']).reset_index(drop=True)
    fighter2_elo = elo_data.copy().to_pandas().drop(['fight_number'], axis = 1).rename(columns = {'fighter': 'fighter2', 'start_elo': 'fighter2_start_elo'}).sort_values(['date']).reset_index(drop=True)

    # Merge this onto the fight results data, because this is a merge to the nearest date before or on the fight date. Can't merged elo_fight_data because draws etc were filtered out.

    fight_results = fight_results.sort_values(['date']).reset_index(drop=True)

    fight_results = pd.merge_asof(
        fight_results,
        fighter1_elo,
        on = 'date',
        by = 'fighter1',
        direction = 'backward'
    )
    fight_results = pd.merge_asof(
        fight_results,
        fighter2_elo,
        on = 'date',
        by = 'fighter2',
        direction = 'backward'
    )

    return fight_results
            