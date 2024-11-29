def convert_fight_stats_to_df(clean_fighter_stats, totals_column_names,
    significant_strikes_column_names):
    """
    convert a list of fighter stats from organise_fight_stats() into a structured dataframe
    check if list of stats is empty, there are old fights that do not have stats
    if fight has no stats, then fill stat columns with nans
    if fight has stats continue and get number of rounds in the fight
    for each round in fight, get stats for totals and significant strikes
    the summary of stats for the fights are ignored
    merge totals and significant stike stats together and return as one df

    arguments:
    clean_fighter_stats (list): list of fighter stats from organise_fight_stats()
    totals_column_names (list): list of column names for totals type stats
    significant_strikes_column_names (list): list of column names for significant strike type stats

    returns:
    a dataframe of fight stats
    """
    totals_df = pd.DataFrame(columns=totals_column_names)
    significant_strikes_df = pd.DataFrame(columns=
        significant_strikes_column_names)
    if len(clean_fighter_stats) == 0:
        totals_df.loc[len(totals_df)] = [np.nan] * len(list(totals_df))
        significant_strikes_df.loc[len(significant_strikes_df)] = [np.nan
            ] * len(list(significant_strikes_df))
    else:
        number_of_rounds = int((len(clean_fighter_stats) - 2) / 2)
        totals_df = pd.DataFrame(columns=totals_column_names)
        significant_strikes_df = pd.DataFrame(columns=
            significant_strikes_column_names)
        for round in range(number_of_rounds):
            totals_df.loc[len(totals_df)] = ['Round ' + str(round + 1)
                ] + clean_fighter_stats[round + 1]
            significant_strikes_df.loc[len(significant_strikes_df)] = [
                'Round ' + str(round + 1)] + clean_fighter_stats[round + 1 +
                int(len(clean_fighter_stats) / 2)]
    fighter_stats_df = totals_df.merge(significant_strikes_df, how='inner')
    return fighter_stats_df
