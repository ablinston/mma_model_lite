def organise_fight_results(results_from_soup, fight_results_column_names):
    """
    organise list of fight results
    fighters' names should be from index 1 and 2
    fight outcome should be from index 3 and 4
    other results includes from index 5 onwards
    weightclass, method, round, time, time format, referee, and details, should be 
    append all results into list and convert to a df

    arguments:
    results_from_soup (list): list of results from parse_fight_results()
    fight_results_column_names (list): list of column names for fight results

    returns:
    an organised list of fight results
    """
    fight_results_clean = []
    fight_results_clean.append(results_from_soup[0])
    fight_results_clean.append(' vs. '.join(results_from_soup[1:3]))
    fight_results_clean.append('/'.join(results_from_soup[3:5]))
    fight_results_clean.extend([re.sub('^(.+?): ?', '', text) for text in
        results_from_soup[5:]])
    fight_result_df = pd.DataFrame(columns=fight_results_column_names)
    fight_result_df.loc[len(fight_result_df)] = fight_results_clean
    return fight_result_df
