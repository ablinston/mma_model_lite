# # This function takes the data frames for the event odds urls and event odds data along with a new data frame of fighters



# def update_event_urls_and_odds(existing_event_urls, existing_event_odds, new_fighter_list, config):

#     pdb.set_trace()    

#     # Read the fighter odds urls
#     fighter_odds_urls = pd.read_csv(config['fighter_odds_urls'])    

#     # Get all the event urls for full odds
#     event_urls_odds = parallel_scrape_odds_fast(
#         new_fighter_list, 
#         num_threads = config['num_threads'], 
#         firefox_location = config['firefox_location'], 
#         gecko_driver = config['gecko_driver_location'],
#         odds_urls_overide_df = fighter_odds_urls,
#         playwright = False,
#         event_urls_only = True,
#         fast = True)

#     # Get rid of duplicate events and keep the most common date for the event
#     event_urls_odds_agg = event_urls_odds.groupby('event_url', as_index=False)['event_date'].apply(lambda x: x.mode().iloc[0])

#     # Convert the date
#     event_urls_odds_agg['event_date'] = pd.to_datetime(event_urls_odds_agg['event_date'].str.replace(r'(\d+)(st|nd|rd|th)', r'\1', regex=True), format = '%b %d %Y')

#     # Scrape the odds of each event after 2017 (odds only really start to show after then)
#     event_odds = parallel_scrape_event_odds_fast(
#         event_urls_odds_agg[event_urls_odds_agg['event_date'] > '01-01-2018'], 
#         num_threads = config['num_threads'], 
#         firefox_location = config['firefox_location'], 
#         gecko_driver = './data/firefox_driver_32bit/geckodriver0_35.exe',
#         fast = False)

#     return new_fight_results_data, new_fight_stats_data