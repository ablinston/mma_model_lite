def get_fighter_data(soup, url, **kwargs):
    """
    get fighter data and organise from a url

    arguments:
    soup: a soup output from the get_soup function
    url: the url that the soup is from

    returns:
    data frame
    """
    return organise_fighter_tott(parse_fighter_tott(soup), **kwargs, url = url)
