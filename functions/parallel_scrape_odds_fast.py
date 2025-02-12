# This function is specifically for parallel scraping odds data

def parallel_scrape_odds_fast(
        fighter_list, 
        output_df = pd.DataFrame(), 
        config = [],
        num_threads = 1, 
        firefox_location = '', 
        gecko_driver = './data/firefox_driver_32bit/geckodriver.exe',
        odds_urls_overide_df = None,
        event_urls_only = False):

    
    # Keep counters to track progress
    total_fighters = len(fighter_list)
    completed_count = 0

    scrape_results = []
        
    # Split the data into chunks to stop the cookies expiring
    batch_size = 200
    
    # Loop through slices of the DataFrame
    for start in range(0, len(fighter_list), batch_size):
        chunk = fighter_list[start:(start + batch_size)]
        
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
        
        # Get the cookies for the odds webpages using a Selenium call
        driver.get("https://www.bestfightodds.com/events/ufc-3570")
    
        # Get cookies from Selenium
        selenium_cookies = driver.get_cookies()
        cookies_dict = {cookie['name']: cookie['value'] for cookie in selenium_cookies}
        # Quit the browser
        driver.quit()
        
        with ThreadPoolExecutor(max_workers = num_threads) as executor:
            future_to_url = {executor.submit(scrape_fighter_odds_fast, fighter, cookies_dict, odds_urls_overide_df, event_urls_only): fighter for fighter in chunk}
    
            # Collect results as they complete
            for future in as_completed(future_to_url):
                try:
                    result = future.result()
                    scrape_results.append(result)  # Collect the result (url, title)
                except Exception as e:
                    print(f"Error for URL {url}: {e}")
    
                # Increment the completed count
                completed_count += 1
                print(f"Completed {completed_count} out of {total_fighters} tasks")
            
    # This ensures that all threads are done
    executor.shutdown(wait=True)

    # Combine the results
    return pd.concat(scrape_results)
