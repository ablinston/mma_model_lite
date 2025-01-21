# This function scrapes the fighter odds data for a fighter.

# It returns a data frame with the odds for all the fights those fighters have been involved with.

# This fast function requires cookies so can be more temperamental. That's why there is an alternate version sans '_fast'

def scrape_fighter_odds_fast(
        fighter_name, 
        cookies_dict, 
        odds_urls_overide_df):

    # pdb.set_trace()

    try:
        
        # Get headers for the website request. This is a copy and paste from a manual visit
        headers = {'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Encoding':'gzip, deflate, br, zstd',
            'Accept-Language':'en-US,en;q=0.5',
            'Connection':'keep-alive',
            'Host':'www.bestfightodds.com',
            'Priority':'u=0, i',
            'Referer':'https://www.bestfightodds.com/events/ufc-3570',
            'Sec-Fetch-Dest':'document',
            'Sec-Fetch-Mode':'navigate',
            'Sec-Fetch-Site':'same-origin',
            'Upgrade-Insecure-Requests':'1',
            'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:134.0) Gecko/20100101 Firefox/134.0'}
        
        # Navigate to the website
        response = requests.get('https://www.bestfightodds.com/search?query=' + fighter_name, headers=headers, cookies=cookies_dict)
        
        # Extract the page source and convert to xml
        soup = BeautifulSoup(response.text, 'html.parser')

        # skip if no possible matches found
        if soup.find('table') == None:
            print(f"Error scraping {fighter_name}: {e}")
            return pd.DataFrame()

        fighter_name_parts = fighter_name.lower().split()
    
        # Create placeholder for the 
        link_url = ''
        found = False
        exact_find = False
        
        # See if url is in override
        if fighter_name in odds_urls_overide_df['fighter']:
            link_url = odds_urls_overide_df[[odds_urls_overide_df['fighter'] == fighter_name]]['url'].iloc[0]
            exact_find = True
        
        # Then try to find the fighter in the table
        # Search through all tables in the soup
        for table in soup.find_all('table'):
            # Iterate over all rows in the table
            for row in table.find_all('tr'):
                # Iterate over each cell in the row
                for cell in row.find_all('td'):
                    # Check if the cell contains the exact search string
                    if (clean_name_for_odds_search(fighter_name) == clean_name_for_odds_search(cell.get_text()) and not exact_find:
                        # Find the <a> tag inside the cell
                        link_tag = cell.find('a')
                        if link_tag and not exact_find:
                            # Extract the href attribute (the link)
                            link_url = 'https://www.bestfightodds.com' + link_tag.get('href')
                            print(f'Found exact string: {fighter_name}, Link: {link_url}')
                            # Take only the first entry, so flag as found
                            exact_find = True
                            found = True
                    # Check if the cell contains the search string
                    elif all(clean_name_for_odds_search(part) in clean_name_for_odds_search(cell.get_text()) for part in fighter_name_parts) and not found:
                        # Find the <a> tag inside the cell
                        link_tag = cell.find('a')
                        if link_tag and not found:
                            # Extract the href attribute (the link)
                            link_url = 'https://www.bestfightodds.com' + link_tag.get('href')
                            print(f'Found string: {fighter_name}, Link: {link_url}')
                            # Take only the first entry, so flag as found
                            found = True

        # Check if a valid output has been found or if we have gone directly to the fighters odds page
        if link_url == '' and (fighter_name not in soup.title.string):
            print(f"Error scraping {fighter_name}: {e}")
            return pd.DataFrame()

        # Navigate to the website if we're not already there
        if fighter_name not in soup.title.string:
            # Navigate to the website
            response = requests.get(link_url, headers=headers, cookies=cookies_dict)

        # Get the html table from the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the table in the soup object
        table = soup.find('table')

        # Convert the table to a data frame and add the rows to the results
        odds_df = betfightodds_table_to_df(table, fighter_name)
    
    except Exception as e:
        print(f"Error scraping {fighter_name}: {e}")
        return pd.DataFrame()

    return odds_df
