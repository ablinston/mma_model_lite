def parallel_scrape(url_list, parse_function, output_df = pd.DataFrame(), num_threads = 1, **kwargs):
    """
    scrape from a list of urls

    arguments:
    url_list: a list of url strings
    parse_function: the function to be run on the data scraped from the page
    output_df: the output pandas df that the parsed results
    num_threads: the number of CPU threads to use
    **kwargs: any other keyword arguments to pass to the parse_function

    """
    #pdb.set_trace()
    with ThreadPoolExecutor(max_workers = num_threads) as executor:
        future_to_url = {executor.submit(get_soup, url): url for url in url_list}

        # Now iterate through the functions
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                # Check to see whether we need the url in the function and input it if needed
                if 'url' in inspect.signature(parse_function).parameters:
                    result = parse_function(future.result(), **kwargs, url = url)
                else:
                    result = parse_function(future.result(), **kwargs)
                output_df = pd.concat([output_df, result])
                print(f'Scraped url: {url}')
            except Exception as exc:
                print(f'Error scraping {url}: {exc}')

    return output_df