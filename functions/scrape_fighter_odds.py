# This function scrapes the fighter odds data for a fighter.

# It returns a data frame with the odds for all the fights those fighters have been involved with.

def scrape_fighter_odds(
        fighter_name, 
        firefox_binary = 'C:/Program Files/Mozilla Firefox/firefox.exe', 
        gecko_driver = './data/firefox_driver_32bit/geckodriver.exe'):

    # pdb.set_trace()
    
    ###########
    # Setup headless Firefox for getting odds information
    options = Options()
    options.headless = True
    options.accept_insecure_certs = True
    # Create a Service object with the path to Firefox driver
    options.binary_location = firefox_binary # firefox_binary
    service = Service(gecko_driver)
    # Create a WebDriver instance
    driver = webdriver.Firefox(service=service, options=options)
    ###########

    try:

        # Navigate to the website
        driver.get('https://www.bestfightodds.com/search?query=' + fighter_name)
        
        # Extract the page source and convert to xml
        soup = BeautifulSoup(driver.page_source, 'lxml')

        # skip if no possible matches found
        if soup.find('table') == None:
            # Quit the browser
            driver.quit()
            print(f"Error scraping {fighter_name}: {e}")
            return pd.DataFrame()

        # Create placeholder for the 
        link_url = ''
        found = False
        exact_find = False
        
        # Then try to find the fighter in the table
        # Search through all tables in the soup
        for table in soup.find_all('table'):
            # Iterate over all rows in the table
            for row in table.find_all('tr'):
                # Iterate over each cell in the row
                for cell in row.find_all('td'):
                    # Check if the cell contains the exact search string
                    if (fighter_name.lower() == cell.get_text().lower()) and not exact_find:
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
                    elif (fighter_name.lower() in cell.get_text().lower()) and not found:
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
            # Quit the browser
            driver.quit()
            print(f"Error scraping {fighter_name}: {e}")
            return pd.DataFrame()

        # Navigate to the website if we're not already there
        if fighter_name not in soup.title.string:
            driver.get(link_url)

        # Get the html table from the page
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Find the table in the soup object
        table = soup.find('table')

        # Convert the table to a data frame and add the rows to the results
        odds_df = betfightodds_table_to_df(table, fighter_name)
    
    except Exception as e:
        print(f"Error scraping {fighter_name}: {e}")
        # Quit the browser
        driver.quit()
        return pd.DataFrame()

    finally:

        # Quit the browser
        driver.quit()

    return odds_df
