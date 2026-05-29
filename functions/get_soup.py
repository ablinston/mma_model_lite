def get_soup(url):
    """
    get soup from url using beautifulsoup

    arguments:
    url (str): url of page to parse

    returns:
    soup
    """
    # page = requests.get(url)
    # soup = BeautifulSoup(page.content, 'html.parser')
    # pdb.set_trace()
    if not os.name == 'nt':
        soup = get_website_soup(url, fast = False, time_delay = True)
    else:
        soup = get_website_soup(url, fast = False, chrome_binary = 'C:\Program Files\Google\Chrome Beta\Application\chrome.exe', time_delay = True)
    
    return soup
