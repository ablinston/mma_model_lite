def clean_all_raw_data_and_save(config):
    
    """
    This function takes all the raw scraped data and cleans them, then saves to file
    
    Returns: nothing
    
    """
    # pdb.set_trace()
        
    # Read in the data
    events = fp.ParquetFile('./data/' + config['event_details_file_name']).to_pandas()
    upcoming_events = fp.ParquetFile('./data/' + config['upcoming_event_details_file_name']).to_pandas()
    upcoming_fight_details = fp.ParquetFile('./data/' + config['upcoming_fight_details_file_name']).to_pandas()
    fighters = fp.ParquetFile('./data/' + config['fighter_tott_file_name']).to_pandas()
    fight_results = fp.ParquetFile('./data/' + config['fight_results_file_name']).to_pandas()
    fight_stats = fp.ParquetFile('./data/' + config['fight_stats_file_name']).to_pandas()
    fighter_odds = fp.ParquetFile('./data/' + config['fighter_odds_file_name']).to_pandas()
    event_odds = fp.ParquetFile('./data/' + config['event_odds_file_name']).to_pandas()
    betfair_odds = fp.ParquetFile('./data/' + config['betfair_odds_file_name']).to_pandas()
    betfair_upcoming_odds = fp.ParquetFile('./data/' + config['betfair_upcoming_odds_file_name']).to_pandas()
    
    # Data clearning
    cleaned_dict = clean_all_raw_data(events, upcoming_events, upcoming_fight_details, fighters, fight_results, fight_stats, fighter_odds, event_odds, betfair_odds, betfair_upcoming_odds)
    
    ###########################################
    
    # Save the outputs
    
    fp.write('./data/' + config['event_details_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['events'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['upcoming_event_details_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['upcoming_events'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['fighter_tott_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['fighters'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['upcoming_fight_details_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['upcoming_fight_details'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['fight_results_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['fight_results'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['fight_stats_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['fight_stats_agg'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['fighter_odds_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['fighter_odds'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['event_odds_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['event_odds'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['betfair_odds_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['betfair_odds'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['betfair_upcoming_odds_file_name'].replace('.parquet', '_clean.parquet'),
             cleaned_dict['betfair_upcoming_odds'],
             compression = 'ZSTD')
    
    fp.write('./data/' + config['fighter_names_file_name'],
             cleaned_dict['fighter_names'],
             compression = 'ZSTD')

    return None