def generate_alphabetical_urls():
    """
    generate a list of alphabetical urls for fighter details
    fighter urls are split by their last name and categorised alphabetically
    loop through each character in the alphabet from a to z to parse all the urls
    return all fighter urls as a list

    arguments:
    none

    returns:
    a list of urls of fighter details
    """
    list_of_alphabetical_urls = []
    for character in list(string.ascii_lowercase):
        list_of_alphabetical_urls.append(
            'http://ufcstats.com/statistics/fighters?char=' + character +
            '&page=all')
    return list_of_alphabetical_urls
