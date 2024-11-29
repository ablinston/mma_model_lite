def parse_fighter_details(soup, fighter_details_column_names):
    """
    parse fighter details from soup
    fighter details include first name, last name, nickname, and url
    returns dataframe with first, last, nickname, url

    arguments:
    soup (html): output of get_soup() parser

    returns:
    a dataframe of fighter details
    """
    fighter_names = []
    for tag in soup.find_all('a', class_='b-link b-link_style_black'):
        fighter_names.append(tag.text)
    fighter_urls = []
    for tag in soup.find_all('a', class_='b-link b-link_style_black'):
        fighter_urls.append(tag['href'])
    fighter_details = list(zip(fighter_names[0::3], fighter_names[1::3],
        fighter_names[2::3], fighter_urls[0::3]))
    fighter_details_df = pd.DataFrame(fighter_details, columns=
        fighter_details_column_names)
    return fighter_details_df
