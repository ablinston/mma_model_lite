# This function takes a data frame of fight results and a list of new UFC results urls. It then
# scrapes for the UFC events and updates the data. It does this using parallel scrapes

def update_fight_results_and_stats_data(existing_fight_results_data, existing_fight_stats_data, new_fights_url_list, config, num_threads = 1):

    # Create empty list for results
    scraped_fight_results = []
    scraped_fight_stats = []
    
    # Run a parallel scrape of the new fights data
    with ThreadPoolExecutor(max_workers = num_threads) as executor:
        
        # Get the html contents for the url
        future_to_url = {executor.submit(get_soup, url): url for url in new_fights_url_list}

        # Now iterate through the functions
        for future in as_completed(future_to_url):
            # Get the url used
            url = future_to_url[future]
            try:
                # Parse the data and get the results
                fight_result, fight_stats = parse_organise_fight_results_and_stats(
                    future.result(),
                    url,
                    config['fight_results_column_names'],
                    config['totals_column_names'],
                    config['significant_strikes_column_names']      
                )
                
                # Save these results to the results list then remove
                scraped_fight_results.append(fight_result)
                scraped_fight_stats.append(fight_stats)
                del([fight_result, fight_stats])
                
            except Exception as e:
                print(f"Error for URL {url}: {e}")
    
    # Now concatenate the new scrape results into a data frame
    if (len(scraped_fight_results) == 0):
        print("Error: No new fight details returned")
        sys.exit()
    elif (len(scraped_fight_results) > 1):
        new_fight_results_data = pd.concat(scraped_fight_results)
        new_fight_stats_data = pd.concat(scraped_fight_stats)
    else:
        new_fight_results_data = scraped_fight_results[0]
        new_fight_stats_data = scraped_fight_stats[0]
    
    del([scraped_fight_results, scraped_fight_stats])    
    
    # Now attach the new results to the original data
    new_fight_results_data = pd.concat([new_fight_results_data, existing_fight_results_data])
    new_fight_stats_data = pd.concat([new_fight_stats_data, existing_fight_stats_data])

    return new_fight_results_data, new_fight_stats_data