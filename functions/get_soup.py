def get_soup(url):
    """
    get soup from url using beautifulsoup

    arguments:
    url (str): url of page to parse

    returns:
    soup
    """
    # get_website_soup drops to a browser itself if the fast fetch gets blocked
    return get_website_soup(url, fast = True, time_delay = True,
                            chrome_binary = config['chrome_binary_location'])
