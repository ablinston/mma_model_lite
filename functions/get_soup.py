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
    
    # If we didn't get a good result
    for attempt in range(0, 6):
        
        if soup.find(string = re.compile('Checking your browser')) is None:
            break
        else:
            soup = get_website_soup(url, fast = False, time_delay = True)
        
    return soup
