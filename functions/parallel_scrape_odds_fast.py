# This function is specifically for parallel scraping odds data

def parallel_scrape_odds_fast(
        fighter_list, 
        output_df = pd.DataFrame(), 
        config = [],
        num_threads = 1, 
        firefox_location = '', 
        gecko_driver = './data/firefox_driver_32bit/geckodriver.exe',
        odds_urls_overide_df = None,
        event_urls_only = False,
        fast = True,
        playwright = False):

    
    # Keep counters to track progress
    total_fighters = len(fighter_list)
    completed_count = 0

    scrape_results = []
            
    # If fast, we will use response.get. There is selenium code in here as well from old runs that still works
    if fast:
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
        driver.get("https://www.bestfightodds.com/")
    
        # Get cookies from Selenium
        selenium_cookies = driver.get_cookies()
        cookies_dict = {cookie['name']: cookie['value'] for cookie in selenium_cookies}
        # Quit the browser
        driver.quit()
        
        scrape_func = partial(
            scrape_fighter_odds_fast,
            cookies_dict = cookies_dict,
            odds_urls_overide_df = odds_urls_overide_df,
            event_urls_only = event_urls_only,
            playwright = playwright,
            fast = fast,
            firefox_binary = firefox_location,
            gecko_driver = gecko_driver
        )
        
        with ThreadPoolExecutor(max_workers = num_threads) as executor:
            future_to_url = {executor.submit(scrape_func, fighter): fighter for fighter in fighter_list}
    
            # Collect results as they complete
            for future in as_completed(future_to_url):
                try:
                    result = future.result()
                    scrape_results.append(result)  # Collect the result (url, title)
                except Exception as e:
                    print(f"Error for fighter {fighter}: {e}")
    
                # Increment the completed count
                completed_count += 1
                print(f"Completed {completed_count} out of {total_fighters} tasks")
                
        # This ensures that all threads are done
        executor.shutdown(wait=True)
    
    # Else it's not fast and we use playwright
    else:
        
        # Use partial to prefill arguments in the function. the map below only passes one argument.
        scrape_func = partial(
            scrape_fighter_odds_fast,
            cookies_dict = {}, # only needed for fast run
            odds_urls_overide_df = odds_urls_overide_df,
            event_urls_only = event_urls_only,
            fast = False,
            playwright = playwright
        )

        # with ThreadPoolExecutor(max_workers = num_threads) as executor:
        #     scrape_results = list(executor.map(scrape_func, fighter_list))
        
        with ThreadPoolExecutor(max_workers = num_threads) as executor:
            futures = {executor.submit(scrape_func, fighter_name): fighter_name for fighter_name in fighter_list}

            scrape_results = []
            for future in as_completed(futures):
                try:
                    scrape_results.append(future.result())  # Get the result as soon as it's ready
                except Exception as e:
                    print(f"Error scraping {futures[future]}: {e}")  # Handle failures
                    
                # Increment the completed count
                completed_count += 1
                print(f"Completed {completed_count} out of {total_fighters} tasks")

    

    # Combine the results
    return pd.concat(scrape_results)
