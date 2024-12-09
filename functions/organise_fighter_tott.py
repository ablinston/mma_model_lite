def organise_fighter_tott(tott_from_soup, fighter_tott_column_names, url):
    """
    organise list of fighter tale of the tape
    remove label of tale of the tape using regex
    e.g. 'Height:5'7"' to '5'7"
    convert and return list as df

    arguments:
    tott_from_soup (list): list of fighter tale of the tale from parse_fighter_tott()
    fighter_tott_column_names (list): list of column names for fighter tale of the tape
    url (str): url of fighter

    results:
    a df of fighter tale of the tape
    """
    # pdb.set_trace()
    fighter_tott_clean = [re.sub('^(.+?): ?', '', text) for text in
        tott_from_soup]
    fighter_tott_clean.append(url)
    fighter_tott_df = pd.DataFrame(columns=fighter_tott_column_names)
    fighter_tott_df.loc[len(fighter_tott_df)] = fighter_tott_clean
    return fighter_tott_df
