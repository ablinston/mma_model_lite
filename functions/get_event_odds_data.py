# This function takes a list of fighters fighting on a card and the parsed date of the event and finds the url on bestfightodds.com
# for the event they were competing in

## AB: THIS FUNCTION ISN'T FINISHED. I ABANDONED IT BECAUSE THE EVENT DATA IS TOO FULL OF RUBBISH

def get_event_odds_data(fighter_list, date, firefox_location, gecko_driver):

    ###########
    # Setup headless Firefox for getting odds information
    options = Options()
    options.headless = True
    options.accept_insecure_certs = True
    # Create a Service object with the path to Firefox driver
    options.binary_location = firefox_location # firefox_binary
    service = Service(gecko_driver)
    # Create a WebDriver instance
    driver = webdriver.Firefox(service=service, options=options)
    ###########

    # First, loop through the fighters to try to find the event url
    for fighter_name in fighter_list:
        try:
    
            # Navigate to the website
            driver.get('https://www.bestfightodds.com/search?query=' + fighter_name)
            
            # Extract the page source and convert to xml
            soup = BeautifulSoup(driver.page_source, 'lxml')
    
            # skip if no possible matches found
            if soup.find('table') == None:
                # Quit the browser
                driver.quit()
                return pd.DataFrame()
    
            # Create placeholder for the 
            link_url = ''
            found = False
            # Then try to find the fighter in the table
            # Search through all tables in the soup
            for table in soup.find_all('table'):
                # Iterate over all rows in the table
                for row in table.find_all('tr'):
                    # Iterate over each cell in the row
                    for cell in row.find_all('td'):
                        # Check if the cell contains the search string
                        if (fighter_name in cell.get_text()) and not found:
                            # Find the <a> tag inside the cell
                            link_tag = cell.find('a')
                            if link_tag and not found:
                                # Extract the href attribute (the link)
                                link_url = 'https://www.bestfightodds.com' + link_tag.get('href')
                                print(f'Found string: {fighter_name}, Link: {link_url}')
                                # Take only the first entry, so flag as found
                                found = True
    
            # Check if a valid output has been found
            if link_url == '':
                # Quit the browser
                driver.quit()
                # Try the next fighter
                break
    
            # Navigate to the website
            driver.get(link_url)
    
            # Get the html table from the page
            soup = BeautifulSoup(driver.page_source, 'html.parser')
    
            # Find the url of the event from the fighter table
            event_url = get_event_odds_url(soup, date)
            
            # Check if it found one
            if event_url == None:
                # Quit the browser
                driver.quit()
                # Try the next fighter
                break
            
            # Otherwise, navigate to the event url
            driver.get('https://www.bestfightodds.com' + event_url)
            
            # Get the html table from the page
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # Read the first table and convert it to a data frame
            event_df = html_table_to_dataframe(driver.page_source)
            
            ## AB: ABANDONED THIS FUNCTION HERE. THE IMPORTED TABLES ARE TOO FULL OF CRAP
            
        except Exception as e:
            print(f"Error scraping {fighter_name}: {e}")
            return pd.DataFrame()
    
        finally:
    
            # Quit the browser
            driver.quit()
    

    return soup
