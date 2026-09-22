
# Load all the packages and functions needed
import fastparquet as fp
import pandas as pd

from mma_functions.odds_scraping import parallel_scrape_odds_fast
from mma_functions.config import config


# This test looks at a specific fighter list and scrapes their odds to ensure these common errors don't occur again. One has a missing capital letter, the other isn't the first entry on the page, and the other goes directly to their page rather than a search results page
def test_parallel_scrape_odds_slow():
    
    # Load what answers are expected
    parallel_scrape_odds_data_correct = fp.ParquetFile('./tests/data/parallel_scrape_odds_data_correct.parquet').to_pandas()
    
    # Set the list of fighters to test, where problems have occurred in the past
    fighter_list = ['Joshua Van', 'Zygimantas Ramaska']
    
    # Read the fighter odds urls
    fighter_odds_urls = pd.read_csv(config['fighter_odds_urls'])
    
    # Scrape the odds data
    odds_df = parallel_scrape_odds_fast(
        fighter_list,
        config = config,
        num_threads = config['num_threads_odds_scrape'], 
        odds_urls_overide_df = fighter_odds_urls,
        fast = False)
    

    # Filter out any new fights that might have occurred after the correct data was scraped
    odds_df = odds_df[odds_df['date'] < '2024-09-25']
    
    # Then sort by fighter in case the parallel runs did a different order
    odds_df.sort_values(by = ['fighter', 'date'], inplace = True)
    
    parallel_scrape_odds_data_correct = parallel_scrape_odds_data_correct[parallel_scrape_odds_data_correct['fighter'].isin(fighter_list)]
    
    assert odds_df.reset_index(drop = True).equals(parallel_scrape_odds_data_correct.reset_index(drop = True)) == True
    
