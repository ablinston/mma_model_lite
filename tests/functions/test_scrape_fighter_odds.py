# Load all the packages and functions needed
exec(open('global.py').read())

# Load what answers are expected
scrape_fighter_odds_data_correct = fp.ParquetFile('./tests/data/scrape_fighter_odds_data_correct.parquet').to_pandas()

# This test looks at a specific URL for an historic event and checks that the results match what is expected
def test_scrape_fighter_odds():
    assert scrape_fighter_odds('Georges St-Pierre', config['firefox_location']).equals(scrape_fighter_odds_data_correct) == True
    
