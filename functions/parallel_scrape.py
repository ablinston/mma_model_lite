
# def parallel_scrape(
#         url_list, 
#         scrape_function, 
#         output_df = pd.DataFrame(), 
#         config = [], 
#         num_threads = 1, 
#         firefox_location = '', 
#         gecko_driver = './data/firefox_driver_32bit/geckodriver.exe',
#         odds_urls_overide_df = None,
        # event_urls_only = False,
#         fast = True, 
#         **kwargs):
    
#     """
#     scrape from a list of urls

#     arguments:
#     url_list: a list of url strings
#     parse_function: the function to be run on the data scraped from the page
#     output_df: the output pandas df that the parsed results
#     num_threads: the number of CPU threads to use
#     **kwargs: any other keyword arguments to pass to the parse_function

#     """
#     #pdb.set_trace()
        
#     # If we're doing it fast, then just use response.get
#     if fast:
#         with ThreadPoolExecutor(max_workers = num_threads) as executor:
#             future_to_url = {executor.submit(scrape_function, url, fast = fast, **kwargs): url for url in url_list}
    
#             # Now iterate through the functions
#             for future in as_completed(future_to_url):
#                 url = future_to_url[future]
#                 try:
#                     output_df = pd.concat([output_df, result])
#                     print(f'Scraped url: {url}')
#                 except Exception as exc:
#                     print(f'Error scraping {url}: {exc}')
#     # Otherwise, use playwright
#     else:
        
#         with sync_playwright() as p:
#         browser = p.chromium.launch(headless=True)
#         context = browser.new_context()

#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             results = list(executor.map(lambda url: scrape_page(context, url, scrape_function, **kwargs), urls))

#         browser.close()
        
#     return output_df
