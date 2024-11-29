def combine_fighter_stats_dfs(fighter_a_stats_df, fighter_b_stats_df, soup):
    """
    concat both fighter's stats into one df
    create new event and bout column as a key
    results in a dataframe of stats for both fighters for a fight

    arguments:
    fighter_a_stats_df (df): a df output from convert_fight_stats_to_df()
    fighter_b_stats_df (df): a df output from convert_fight_stats_to_df()
    soup (html): output of get_soup() parser

    returns
    a dataframe of stats for the fight
    """
    fight_stats = pd.concat([fighter_a_stats_df, fighter_b_stats_df])
    fight_stats['EVENT'] = soup.find('h2', class_='b-content__title'
        ).text.strip()
    fighters_names = []
    for tag in soup.find_all('a', class_='b-link b-fight-details__person-link'
        ):
        fighters_names.append(tag.text.strip())
    fight_stats['BOUT'] = ' vs. '.join(fighters_names)
    fight_stats = move_columns(fight_stats, ['EVENT', 'BOUT'], 'ROUND',
        'before')
    return fight_stats
