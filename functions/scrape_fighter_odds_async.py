# This function scrapes the fighter odds data for a fighter.

# It returns a data frame with the odds for all the fights those fighters have been involved with.

# This fast function requires cookies so can be more temperamental. That's why there is an alternate version sans '_fast'

# event_urls_only tells the function whether to only return the urls of all events for a fighter

async def scrape_fighter_odds_async(
        fighter_name,
        odds_urls_overide_df,
        event_urls_only = False,
        playwright_context = None):

    # pdb.set_trace()

    try:
                
        # Extract the page source and convert to xml
        soup = get_website_soup_async(
            url = 'https://www.bestfightodds.com/search?query=' + fighter_name,
            fast = fast,
            playwright_context = playwright_context
        )            

        # skip if no possible matches found
        if soup.find('table') == None:
            print(f"Error scraping {fighter_name}: {e}")
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
                headers = headers, # Only needed for fast
                cookies = cookies_dict, # Only needed for fast
                firefox_binary = firefox_binary, # Only needed for NOT fast
                gecko_driver = gecko_driver
            )

        # pdb.set_trace()
        
        # See if we want to return odds data or event urls
        if event_urls_only:
            # Collect the event urls
            # Extract all event URLs and their corresponding dates efficiently
            events = [(a['href'], a.next_sibling.strip()) for a in soup.select('a[href^="/events/"]') if a.next_sibling]
            
            # Convert to DataFrame
            odds_df = pd.DataFrame(events, columns=['event_url', 'event_date'])

        else:
            # Find the table in the soup object
            table = soup.find('table')
    
            # Convert the table to a data frame and add the rows to the results
            odds_df = betfightodds_table_to_df(table, fighter_name)
    
    except Exception as e:
        print(f"Error scraping {fighter_name}: {e}")
        return pd.DataFrame()

    return odds_df
