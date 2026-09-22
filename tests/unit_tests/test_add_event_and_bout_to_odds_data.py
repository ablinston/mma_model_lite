# Load all the packages and functions needed
import fastparquet as fp

from mma_functions.odds import add_event_and_bout_to_odds_data

# Load what answers are expected
test_add_event_and_bout_to_odds_data_correct = fp.ParquetFile('./tests/data/test_add_event_and_bout_to_odds_data_correct.parquet').to_pandas()
test_add_event_and_bout_to_odds_data_input_odds = fp.ParquetFile('./tests/data/test_add_event_and_bout_to_odds_data_input_odds.parquet').to_pandas()
test_add_event_and_bout_to_odds_data_input_stats = fp.ParquetFile('./tests/data/test_add_event_and_bout_to_odds_data_input_stats.parquet').to_pandas()
test_add_event_and_bout_to_odds_data_input_events = fp.ParquetFile('./tests/data/test_add_event_and_bout_to_odds_data_input_events.parquet').to_pandas()

# This test looks at a specific URL for an historic fight and checks that the results  and stats match what is expected
def test_add_event_and_bout_to_odds_data():
    
    new_odds = add_event_and_bout_to_odds_data(
        test_add_event_and_bout_to_odds_data_input_odds,
        test_add_event_and_bout_to_odds_data_input_events,
        test_add_event_and_bout_to_odds_data_input_stats).sort_values(['fighter', 'date', 'opponent']).reset_index(drop = True)
        
    assert new_odds.shape == test_add_event_and_bout_to_odds_data_correct.shape
    assert (new_odds[test_add_event_and_bout_to_odds_data_correct.columns].equals(test_add_event_and_bout_to_odds_data_correct.sort_values(['fighter', 'date', 'opponent']).reset_index(drop = True))) == True
    

# fp.write('./tests/data/test_add_event_and_bout_to_odds_data_correct.parquet', new_odds, compression = 'ZSTD')
