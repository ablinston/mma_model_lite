# This function scrapes the fighter odds data for a fighter.

# It returns a data frame with the odds for all the fights those fighters have been involved with.

# This fast function requires cookies so can be more temperamental. That's why there is an alternate version sans '_fast'

# event_urls_only tells the function whether to only return the urls of all events for a fighter

def scrape_fighter_odds_fast(
        fighter_name, 
        cookies_dict, 
        odds_urls_overide_df,
        event_urls = False,
        fast = True,
        playwright = False,
        firefox_binary = '', 
        gecko_driver = '',
        driver = None):

    # pdb.set_trace()
    
    if fast:
        # time.sleep(random.uniform(1, 3))  # Sleep for a random amount of time between requests to avoid being detected
        pass
    
    try:
                
        if playwright == True:
            # Use parallel scraping within playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                playwright_context = browser.new_context()
        else:
            playwright_context = None
            
    
        # Extract the page source and convert to xml
        soup = get_website_soup(
            url = 'https://www.bestfightodds.com/search?query=' + fighter_name,
            fast = fast,
            playwright_context = playwright_context,
            cookies = cookies_dict, # Only needed for fast
            firefox_binary = firefox_binary, # Only needed for NOT fast
            gecko_driver = gecko_driver,
            driver = driver
        )            
        
        # skip if no possible matches found
        if soup.find('table') == None:
            print(f"Error scraping {fighter_name} with {headers['User-Agent']}")
            return pd.DataFrame()

        link_url = find_fighter_odds_url_in_html(soup, fighter_name, odds_urls_overide_df)

        # Check if a valid output has been found or if we have gone directly to the fighters odds page
        if link_url == '' and (fighter_name not in soup.title.string):
            print(f"Error scraping {fighter_name}: {e}")
            return pd.DataFrame()

        # Navigate to the website if we're not already there
        if fighter_name not in soup.title.string:
            # Navigate to the website
            soup = get_website_soup(
                url = link_url,
                fast = fast,
                playwright_context = playwright_context,
                # headers = headers, # Only needed for fast
                cookies = cookies_dict, # Only needed for fast
                firefox_binary = firefox_binary, # Only needed for NOT fast
                gecko_driver = gecko_driver,
                driver = driver
            )

        if playwright:
            # Close the browser
            browser.close()
            
        # pdb.set_trace()
                    
        # Find the table in the soup object
        table = soup.find('table')

        # Convert the table to a data frame and add the rows to the results
        odds_df = betfightodds_table_to_df(table, fighter_name)
        
        # See if we want to return odds data or event urls
        if event_urls:
            # Collect the event urls
            # Extract all event URLs and their corresponding dates efficiently
            events = [(a['href'], a.next_sibling.strip()) for a in soup.select('a[href^="/events/"]') if a.next_sibling]
            
            # Convert to DataFrame
            events_df = pd.DataFrame(events, columns=['event_url', 'event_date'])
            
            # Convert date
            events_df['event_date'] = betfightodds_dates_to_datetime(events_df['event_date'])
            
        else:
            events_df = pd.DataFrame()
    
    except Exception as e:
        print(f"Error scraping {fighter_name}: {e}")
        return pd.DataFrame(), pd.DataFrame()

    return odds_df, events_df
