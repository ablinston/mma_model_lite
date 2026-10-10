# Load all the packages and functions needed
from datetime import datetime
import fastparquet as fp
import pandas as pd

from mma_functions.odds_scraping import update_odds_data
from mma_functions.config import config

# Load what answers are expected and inputs to the function
update_fighter_odds_data_input = fp.ParquetFile("./tests/data/update_fighter_odds_data_input.parquet").to_pandas()
update_fighter_odds_data_correct1 = fp.ParquetFile("./tests/data/update_fighter_odds_data_data_correct.parquet").to_pandas().reset_index(drop=True)
update_fighter_odds_data_correct2 = fp.ParquetFile("./tests/data/update_fighter_odds_data_data_correct2.parquet").to_pandas().reset_index(drop=True)
update_fighter_odds_data_correct3 = fp.ParquetFile("./tests/data/update_fighter_odds_data_data_correct3.parquet").to_pandas().reset_index(drop=True)

# This test looks at a specific URL for an historic event and checks that the results match what is expected
def test_update_odds_data():
        
    result1, result2, result3 = update_odds_data(
        existing_odds_data=update_fighter_odds_data_input, 
        existing_event_urls=pd.DataFrame({"event_date":[], "event_url":[]}), 
        existing_event_odds=pd.DataFrame(),
        new_fighter_list=["Steve Garcia", "Gilbert Burns", "Kyle Nelson", "T J Dillashaw"], 
        new_event_date_list=[datetime.strptime("08-09-2024", "%d-%m-%Y"), datetime.strptime("30-12-2016", "%d-%m-%Y")], 
        config=config, 
        fast_scrape=True
    )
    
    # REMOVED EVENT SCRAPE FROM UPDATE SO COMMENTING OUT THESE CHECKS NOW
    ## Remove events after a certain date
    # result2 = result2[result2['event_date'] < '2024-01-01']
    # result3 = result3[result3['date'] < '2024-01-01']
    
    # Drop the scrape date
    result1.drop(columns=["notformodel_date_fighter_odd_scrape"], inplace=True)
    
    # Sort results
    result1 = result1.sort_values(by=["fighter", "date"]).reset_index(drop=True)
    # result2 = result2.sort_values(by = ['event_date']).reset_index(drop = True)
    # result3 = result3.sort_values(by = ['fighter', 'date']).reset_index(drop = True)

    # Sort by fighter because parallel runs can cause different ordering
    assert result1.equals(update_fighter_odds_data_correct1) == True
    # assert result2.equals(update_fighter_odds_data_correct2) == True
    # assert result3.astype(str).compare(update_fighter_odds_data_correct3.astype(str)) == True #  Not currently used in model but fix for errors if used.

# fp.write('./tests/data/update_fighter_odds_data_data_correct3.parquet', result3, compression = 'ZSTD')
