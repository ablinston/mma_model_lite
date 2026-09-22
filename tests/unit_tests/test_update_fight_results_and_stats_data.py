# Load all the packages and functions needed
exec(open('global.py').read())

# Load what answers are expected
existing_fight_results_data = fp.ParquetFile('./tests/data/update_fight_results_data_input.parquet').to_pandas()
update_fight_results_data_correct = fp.ParquetFile('./tests/data/update_fight_results_data_correct.parquet').to_pandas()

existing_fight_stats_data = fp.ParquetFile('./tests/data/update_fight_stats_data_input.parquet').to_pandas()
update_fight_stats_data_correct = fp.ParquetFile('./tests/data/update_fight_stats_data_correct.parquet').to_pandas()

# This test ensures that the update to fight details data correctly adds the data to an existing data frame
def test_update_fight_results_and_stats_data():
    
    # Define which new fights to add
    fights_urls = ['http://ufcstats.com/fight-details/ee9ade72b44dfa06', 'http://ufcstats.com/fight-details/bcfaf1dd78f12f97']
    
    fight_results, fight_stats = update_fight_results_and_stats_data(
        existing_fight_results_data, 
        existing_fight_stats_data, 
        fights_urls, 
        config, 
        num_threads = config['num_threads'])
    
    # Order the data because the parallel runs can sometimes result in different orders
    fight_results.sort_values(by = 'BOUT', inplace = True)
    fight_stats.sort_values(by = ['BOUT', 'ROUND', 'FIGHTER'], inplace = True)
    
    assert ((fight_results.equals(update_fight_results_data_correct)) & (fight_stats.equals(update_fight_stats_data_correct))) == True

