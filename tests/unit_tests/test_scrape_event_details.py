# Load all the packages and functions needed
import fastparquet as fp

from mma_functions.scraping import scrape_event_details

# Load what answers are expected
scrape_event_details_data_correct = fp.ParquetFile("./tests/data/scrape_event_details_data_correct.parquet").to_pandas()

# This test looks at a specific URL for an historic event and checks that the results match what is expected
def test_scrape_event_details():
    assert scrape_event_details([
        "http://ufcstats.com/event-details/c2a7623f398d9bd7",
        "http://ufcstats.com/event-details/e4770d5ed44a3970"
    ]).equals(scrape_event_details_data_correct) == True
    
