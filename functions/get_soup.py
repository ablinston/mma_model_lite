def get_soup(url):
    """
    get soup from url using beautifulsoup

    arguments:
    url (str): url of page to parse

    returns:
    soup
    """
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    return soup
