# This function is specifically for parallel scraping odds data

def parallel_scrape_odds_fast(
        fighter_list, 
        output_df = pd.DataFrame(), 
        config = [],
        num_threads = 1, 
        firefox_location = '',
        gecko_driver = '',
        odds_urls_overide_df = None,
        event_urls = False,
        fast = True,
        playwright = False):

    
    # Keep counters to track progress
    total_fighters = len(fighter_list)
    completed_count = 0

    odds_scrape_results = []
    event_scrape_results = []
            
    # ###########
    # # Setup headless Firefox for getting odds information
    # options = Options()
    # # options.headless = True
    # options.accept_insecure_certs = True
    # options.set_preference("general.useragent.override",
    #                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    #                "AppleWebKit/537.36 (KHTML, like Gecko) "
    #                "Chrome/121.0.0.0 Safari/537.36")
    # # Create a Service object with the path to Firefox driver
    # options.binary_location = firefox_location # firefox_binary
    # service = Service(gecko_driver)
    # options.set_preference("dom.webdriver.enabled", False)
    # options.set_preference("useAutomationExtension", False)
    # options.set_preference("dom.webnotifications.enabled", False)
    # # Create a WebDriver instance
    # driver = webdriver.Firefox(service=service, options=options)
    # ###########
    
    # # Get the cookies for the odds webpages using a Selenium call
    # driver.get("https://www.bestfightodds.com/")
    # pdb.set_trace()
    # Get cookies from Selenium
    # selenium_cookies = driver.get_cookies()
            
    # Look up the true global config (this function is defined via exec() into the
    # caller's namespace, so globals() here is that namespace, not the local
    # 'config' parameter above, which defaults to [] and is otherwise unused)
    _global_config = globals().get('config')
    chrome_binary_location = _global_config.get('chrome_binary_location') if isinstance(_global_config, dict) else None

    # Run in a window rather than headless so the Cloudflare checkbox can be clicked
    browser = start_browser(
        "https://www.bestfightodds.com/",
        chrome_binary = chrome_binary_location,
        headless = False)

    # Clear the checkbox, falling back to a delay to allow it to be manually done
    clear_captcha(browser)

    cookies = browser.get_all_cookies()

    cookies_dict = {cookie.name: cookie.value for cookie in cookies}
    # Quit the browser unless we are going through one at a time
    if num_threads > 1:
        browser.stop()
    
    scrape_func = partial(
        scrape_fighter_odds_fast,
        cookies_dict = cookies_dict,
        odds_urls_overide_df = odds_urls_overide_df,
        event_urls = event_urls,
        playwright = playwright,
        fast = fast,
        firefox_binary = firefox_location,
        gecko_driver = gecko_driver
    )
    
    # Check if we have more than 1 thread
    if num_threads > 1:
        with ThreadPoolExecutor(max_workers = num_threads) as executor:
            future_to_url = {executor.submit(scrape_func, fighter): fighter for fighter in fighter_list}
    
            # Collect results as they complete
            for future in as_completed(future_to_url):
                try:
                    odds_result, events_result = future.result()
                    odds_scrape_results.append(odds_result)  # Collect the result (url, title)
                    event_scrape_results.append(events_result)  # Collect the result (url, title)
                    
                except Exception as e:
                    print(f"Error for fighter {future}: {e}")
    
                # Increment the completed count
                completed_count += 1
                print(f"Completed {completed_count} out of {total_fighters} tasks")
                
        # This ensures that all threads are done
        executor.shutdown(wait=True)
    
    else:
        for fighter in fighter_list:
            
            odds_result, events_result = scrape_fighter_odds_fast(
                fighter,
                cookies_dict = cookies_dict,
                odds_urls_overide_df = odds_urls_overide_df,
                event_urls = event_urls,
                playwright = playwright,
                fast = fast,
                firefox_binary = firefox_location,
                gecko_driver = gecko_driver,
                browser = browser)
            
            odds_scrape_results.append(odds_result)  # Collect the result (url, title)
            event_scrape_results.append(events_result)  # Collect the result (url, title)
        
    # pdb.set_trace()    
    
    # Combine into a df
    if len(odds_scrape_results) > 0:
        all_fighter_odds = pd.concat(odds_scrape_results)
    else:
        all_fighter_odds = pd.DataFrame()
    
    browser.stop()

    # Combine the results
    if event_urls:
        
        # Get rid of duplicate events and keep the most common date for the event
        if len(all_fighter_odds) > 0:
            event_urls_odds_agg = pd.concat(event_scrape_results).groupby('event_url', as_index=False)['event_date'].apply(lambda x: x.mode().iloc[0])
        else:
            event_urls_odds_agg = pd.DataFrame()
        
        return all_fighter_odds, event_urls_odds_agg
    else:
        return all_fighter_odds
