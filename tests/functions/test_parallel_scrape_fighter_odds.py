
# Load all the packages and functions needed
exec(open('global.py').read())

# Load what answers are expected
parallel_scrape_odds_data_correct = fp.ParquetFile('./tests/data/parallel_scrape_odds_data_correct.parquet').to_pandas()

# This test looks at a specific fighter list and scrapes their odds to ensure these common errors don't occur again. One has a missing capital letter, the other isn't the first entry on the page, and the other goes directly to their page rather than a search results page
def test_parallel_scrape_odds():
    
    # Set the list of fighters to test, where problems have occurred in the past
    fighter_list = ['Joshua Van', 'Zygimantas Ramaska', 'Vanessa Demopoulos']
    
    # Scrape the odds data
    odds_df = parallel_scrape_odds(
        fighter_list,
        config = config,
        num_threads = config['num_threads'], 
        firefox_location = config['firefox_location'],
        gecko_driver = './data/firefox_driver_32bit/geckodriver.exe')
    

    # Filter out any new fights that might have occurred after the correct data was scraped
    odds_df = odds_df[odds_df['date'] < '2024-09-25']
    
    # Then sort by fighter in case the parallel runs did a different order
    odds_df.sort_values(by = ['fighter', 'date'], inplace = True)
    
    assert odds_df.reset_index(drop = True).equals(parallel_scrape_odds_data_correct.reset_index(drop = True)) == True
    
