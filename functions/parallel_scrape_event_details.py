# This function is specifically for parallel scraping event details for a list of events

def parallel_scrape_event_details(
        url_list,
        num_threads = 1):

    # Keep counters to track progress
    total_urls = len(url_list)
    completed_count = 0

    scrape_results = []
    with ThreadPoolExecutor(max_workers = num_threads) as executor:
        future_to_url = {executor.submit(scrape_event_details, [url]): url for url in url_list}

        # Collect results as they complete
        for future in as_completed(future_to_url):
            try:
                result = future.result()
                scrape_results.append(result)  # Collect the result (url, title)
            except Exception as e:
                print(f"Error for URL {url}: {e}")

            # Increment the completed count
            completed_count += 1
            print(f"Completed {completed_count} out of {total_urls} tasks")
            
    # This ensures that all threads are done
    executor.shutdown(wait=True)

    # Combine the results
    return pd.concat(scrape_results)
