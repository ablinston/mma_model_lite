# Load all the packages and functions needed
exec(open('global.py').read())

# Load what answers are expected and inputs to the function
update_fighter_odds_data_input = fp.ParquetFile('./tests/data/update_fighter_odds_data_input.parquet').to_pandas()
update_fighter_odds_data_correct = fp.ParquetFile('./tests/data/update_fighter_odds_data_data_correct.parquet').to_pandas()

# This test looks at a specific URL for an historic event and checks that the results match what is expected
def test_update_fighter_odds_data():
    
    result = update_fighter_odds_data(
        existing_odds_data = update_fighter_odds_data_input, 
        new_fighter_list = ['Steve Garcia', 'Gilbert Burns', 'Kyle Nelson'], 
        new_event_date_list = [datetime.strptime('08-09-2024', '%d-%m-%Y')], 
        config = config, 
        num_threads = config['num_threads']
    )
    
    # Sort by fighter because parallel runs can cause different ordering
    
    assert result.sort_values(by = 'fighter').equals(update_fighter_odds_data_correct) == True
    

