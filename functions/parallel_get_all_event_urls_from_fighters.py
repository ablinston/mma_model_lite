
# This function gets all event urls for UFCstats.com given a list of fighter urls

def parallel_get_all_event_urls_from_fighters(fighter_list_urls, fighter_result_headers, num_threads = 3):

    # Keep counters to track progress
    total_fighters = len(fighter_list_urls)
    completed_count = 0

    scrape_results = []
    with ThreadPoolExecutor(max_workers = num_threads) as executor:
        future_to_url = {executor.submit(extract_event_urls, fighter_url, fighter_result_headers): fighter_url for fighter_url in fighter_list_urls}

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
    return [item for sublist in scrape_results for item in sublist]
