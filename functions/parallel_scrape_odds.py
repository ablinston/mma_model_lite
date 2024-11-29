import pandas as pd

# This function is specifically for parallel scraping odds data

def parallel_scrape_odds(
        fighter_list, 
        output_df = pd.DataFrame(), 
        config = [],
        num_threads = 1, 
        firefox_location = '', 
        gecko_driver = './data/firefox_driver_32bit/geckodriver.exe'):

    # Keep counters to track progress
    total_fighters = len(fighter_list)
    completed_count = 0

    scrape_results = []
    with ThreadPoolExecutor(max_workers = num_threads) as executor:
        future_to_url = {executor.submit(scrape_fighter_odds, fighter, firefox_location, gecko_driver): fighter for fighter in fighter_list}

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
