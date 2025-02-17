# Load all the packages and functions needed
exec(open('global.py').read())

# Read the fighter odds urls
fighter_odds_urls = pd.read_csv(config['fighter_odds_urls'])

# Load what answers are expected
scrape_fighter_odds_data_correct = fp.ParquetFile('./tests/data/scrape_fighter_odds_data_correct.parquet').to_pandas()

# This test looks at a specific URL for an historic event and checks that the results match what is expected
def test_scrape_fighter_odds():
    assert scrape_fighter_odds_fast('Georges St-Pierre', cookies_dict = {}, odds_urls_overide_df = pd.DataFrame(), fast = False).equals(scrape_fighter_odds_data_correct) == True
    
# te = scrape_fighter_odds_fast('Georges St-Pierre', cookies_dict = {}, odds_urls_overide_df = fighter_odds_urls, fast = False)
