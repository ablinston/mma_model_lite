# This function scrapes the fighter odds data for a fighter.

# It returns a data frame with the odds for all the fights those fighters have been involved with.

# This fast function requires cookies so can be more temperamental. That's why there is an alternate version sans '_fast'

# event_urls_only tells the function whether to only return the urls of all events for a fighter

def scrape_event_odds_fast(
        event_details, # A tuple with the event url and date together
        cookies_dict):

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
        response = requests.get('https://www.bestfightodds.com' + event_details[1]['event_url'], headers=headers, cookies=cookies_dict)
        
        # Extract the page source and convert to xml
        soup = BeautifulSoup(response.text, 'html.parser')

        # skip if no possible matches found
        if soup.find('table') == None:
            print(f"Error scraping {event_details[1]['event_url']}: {e}")
            return pd.DataFrame()

        # divs = soup.find_all('div', {'class': 'table-div'})
        # Find the table in the soup object
        tables = soup.find_all('table')
        
        # Check if table is found
        if len(tables) == 2:
        
            # Get the odds value tables
            table = tables[1]  # Selecting the second table

            rows = table.find_all('tr')[1:]  # Skip header row

            # Construct a list of the rows searching cells by class
            data = [
                [
                    row.find('th', {'scope': 'row'}).text,  # Fighter name
                    *map(lambda td: td.find('span').text.strip(), row.find_all('td', class_=odds_scrape_config['odds_cell_class'])),  # Odds
                    *map(lambda td: td.text.strip(), (td for td in row.find_all('td') if td.get('class', []) != [odds_scrape_config['odds_cell_class']]))  # Other columns
                ]
                for row in rows
            ]

            # Create column names dynamically
            odds_count = max(len(row) - 1 for row in data)  # First column is 'fighter'
            columns = ['fighter'] + [f'Odd {i+1}' for i in range(odds_count)] + [f'Missing {i+1}' for i in range(len(data[0]) - odds_count - 1)]

            # Convert to DataFrame
            df = pd.DataFrame(data, columns=columns)
            
            # Remove prop bet columns
            exclude_strings = ['distance', ' is ', 'Fight', ' other ', ' round ', ' rounds', 'wins by ', 'has more ', 'by decision', ' has ', ' points ', 'FOTN']
            df = df[~df['fighter'].str.contains('|'.join(exclude_strings), na=False, case=False)]
    
            # Drop any columns not recognised
            df = df.iloc[:, 0:(len(odds_scrape_config['events_colnames']) + 1)]
            df.columns = ['fighter'] + odds_scrape_config['events_colnames']
            
            # Add on the date
            df['date'] = event_details[1]['event_date']
            
        else:
            print("Table not found")
            return pd.DataFrame()
    
    except Exception as e:
        print(f"Error scraping {fighter_name}: {e}")
        return pd.DataFrame()

    return df
