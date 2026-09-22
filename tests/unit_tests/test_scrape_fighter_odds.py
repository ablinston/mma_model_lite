# Load all the packages and functions needed
from ablinston_ml.scraping import clear_captcha, start_browser
import fastparquet as fp
import pandas as pd

from mma_functions.odds_scraping import scrape_fighter_odds_fast
from mma_functions.config import config

# Read the fighter odds urls
fighter_odds_urls = pd.read_csv(config['fighter_odds_urls'])

# Load what answers are expected
scrape_fighter_odds_data_correct = fp.ParquetFile('./tests/data/scrape_fighter_odds_data_correct.parquet').to_pandas()

# This test looks at a specific URL for an historic event and checks that the results match what is expected
def test_scrape_fighter_odds():
    
    # Start the browser
    browser = start_browser(
        "https://www.bestfightodds.com",
        chrome_binary = config['chrome_binary_location'],
        headless = False)

    # Allow captcha solve
    clear_captcha(browser)

    result1, result2 = scrape_fighter_odds_fast(
        'Georges St-Pierre',
        cookies_dict = {},
        odds_urls_overide_df = pd.DataFrame({'fighter':[]}),
        fast = False,
        browser = browser)

    browser.stop()
    
    assert result1.equals(scrape_fighter_odds_data_correct) == True
    
# te = scrape_fighter_odds_fast('Georges St-Pierre', cookies_dict = {}, odds_urls_overide_df = fighter_odds_urls, fast = False)
