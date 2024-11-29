# This function takes the soup from a fighter's bestfightodds.com webpage and a parsed date.
# It looks to match the date to the listed events the fighter has been in and returns the url
# of the event page if it finds it.

def get_event_odds_url(soup, date):

    # Find the table in the soup object
    table = soup.find('table')
    
    # Specify the target date to search for in the correctly formatted way for the website
    target_date = date.strftime('%b ') + f'{date.day}' + get_day_suffix(date.day) + ' ' + f'{date.year}'
    
    # Find all relevant <td> elements
    tds = soup.find_all('td', colspan="8", scope="row")
    
    # Initialize variable to hold the href
    found_href = None
    
    # Iterate through the cells to find the target date
    for td in tds:
        # Check if the target date is in the cell's text
        if target_date in td.get_text():
            # If found, get the previous <a> href
            link_tag = td.find('a')
            if link_tag:
                found_href = link_tag.get('href')  # Extract href
            break  # Exit the loop after finding the first match
    
    return found_href
