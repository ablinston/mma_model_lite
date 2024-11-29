def parse_organise_fight_results_and_stats(soup, url,
    fight_results_column_names, totals_column_names,
    significant_strikes_column_names):
    """
    parse and organise fight results and fight stats from soup
    this function combines other functions that parse fight results and stats into one
    and returns two dfs, one for fight results and the other for fight stats

    arguments:
    soup (html): output of get_soup() parser
    url (str): url of fight
    fight_results_df (df): an df
    fight_results_column_names (list): list of column names for fight results
    fight_stats_df (df):
    totals_column_names (list): list of column names for totals type stats
    significant_strikes_column_names (list): list of column names for significant strike type stats

    returns:
    two dfs for fight results and stats
    """
    fight_results = parse_fight_results(soup)
    fight_results.append('URL:' + url)
    fight_results_df = organise_fight_results(fight_results,
        fight_results_column_names)
    fighter_a_stats, fighter_b_stats = parse_fight_stats(soup)
    fighter_a_stats_clean = organise_fight_stats(fighter_a_stats)
    fighter_b_stats_clean = organise_fight_stats(fighter_b_stats)
    fighter_a_stats_df = convert_fight_stats_to_df(fighter_a_stats_clean,
        totals_column_names, significant_strikes_column_names)
    fighter_b_stats_df = convert_fight_stats_to_df(fighter_b_stats_clean,
        totals_column_names, significant_strikes_column_names)
    fight_stats_df = combine_fighter_stats_dfs(fighter_a_stats_df,
        fighter_b_stats_df, soup)
    return fight_results_df, fight_stats_df
