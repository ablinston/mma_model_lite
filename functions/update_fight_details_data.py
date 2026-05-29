# This function takes a data frame of fight details and a list of new UFC events. It then
# scrapes for the UFC events and updates the data

def update_fight_details_data(existing_fights_data, new_events_url_list, config, num_threads = 1, debug = False):

    # Create empty list for results
    scrape_results = []

    # Run a parallel scrape of the new fights data
    with ThreadPoolExecutor(max_workers = num_threads) as executor:
        future_to_url = {executor.submit(parse_fight_details_url, url, config): url for url in new_events_url_list}

        # Now iterate through the functions
        for future in as_completed(future_to_url):
            try:
                result = future.result()
                scrape_results.append(result)  # Collect the result (url, title)
            except Exception as e:
                url = future_to_url[future]  # Get the URL associated with the failing future
                print(f"Error for URL {url}: {e}")

    # If we want to debug, return the raw scrape results, otherwise clean the results
    if debug:
        return scrape_results

    # Now concatenate the new scrape results into a data frame
    if (len(scrape_results) == 0):
        print("Error: No new fight details returned")
        sys.exit()
    elif (len(scrape_results) > 1):
        new_fights_data = pd.concat(scrape_results)
    else:
        new_fights_data = scrape_results[0]
    
    # Check we have the correct columns
    if (list(new_fights_data.columns) == config['fight_details_column_names']):
        # If they are, add on the rows to the existing data
        parsed_fight_details_df = pd.concat([new_fights_data, existing_fights_data])
        del([existing_fights_data, new_fights_data])
    else:
        print("Error: new parsed fight details have different columns")
        print("Got:")
        print(list(new_fights_data.columns))
        print("Expecting:")
        print(config['fight_details_column_names'])
        sys.exit()

    return parsed_fight_details_df
