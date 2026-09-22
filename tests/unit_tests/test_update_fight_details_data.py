# Load all the packages and functions needed
import fastparquet as fp

from mma_functions.parsing import update_fight_details_data
from mma_functions.config import config

# Load what answers are expected
existing_fight_details_data = fp.ParquetFile('./tests/data/update_fight_details_data_input.parquet').to_pandas()
update_fight_details_data_correct = fp.ParquetFile('./tests/data/update_fight_details_data_correct.parquet').to_pandas()

# This test ensures that the update to fight details data correctly adds the data to an existing data frame
def test_update_fight_details_data():
    
    result = update_fight_details_data(
        existing_fights_data = existing_fight_details_data,
        new_events_url_list = ['http://ufcstats.com/event-details/0c01568b1ac77bff'],
        config = config
    )
    
    assert result[['EVENT', 'BOUT', 'URL']].equals(update_fight_details_data_correct) == True
    
