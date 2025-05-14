def update_betfair_odds(existing_betfair_odds_df, betfair_login_config, download_folder, decompressed_output_dir, decompressed_temp_output_dir):
    
    """
    This function updates the raw betfair odds historic data by downloading new files
    
    """
    # pdb.set_trace()
            
    # Get the latest date of data we have available
    # latest_date = pd.to_datetime("2025-04-25")
    latest_date = existing_betfair_odds_df['event_date'].max()
    
    # Get the date to get data until. Betfair takes 5 days to become available
    new_date = datetime.today() - timedelta(days = 5)
    
    if new_date > latest_date:
        
        ############################################
        ## HISTORIC DATA DOWNLOAD
        
        # Open the connection with Betfair. This requires the betfair config which is not tracked by git because it contains passwords
        
        # ChatGPT says data is generally available 5 days after and event
        
        try:
            # get file list
            file_list = trading.historic.get_file_list(
                "Other Sports",
                "Basic Plan",
                from_day = latest_date.day,
                from_month = latest_date.month,
                from_year = latest_date.year,
                to_day = new_date.day,
                to_month = new_date.month,
                to_year = new_date.year,
                market_types_collection=["MATCH_ODDS"],
                # countries_collection=["GB"],
                file_type_collection=["L", "M"],
            )
        except:
            print(f"No betfair files found to download. Check that the current month, {new_date.month}, has been 'purchased'.")
            # file_list = []
        
        # Check there are files to parse
        if len(file_list) > 0:
            
            # download the files
            for file in file_list:
                # print(file)
                download = trading.historic.download_file(file_path = file, store_directory = download_folder)
                # print(download)
            
            ## Tried parallel running but ends up with corrupted files
            # # download the files
            # completed = parallel_run_function(
            #     func = trading.historic.download_file,
            #     input_list = file_list,
            #     max_workers = config['num_threads'],
            #     collapse_output_to_df = False,
            #     store_directory = download_folder)

            ############################################
            ## PROCESS DOWNLOADED FILES
            
            bz2_file_log = betfair_filter_bz2_for_fights(download_folder, decompressed_temp_output_dir)
            
            
            ############################################
            ## STREAMING HISTORIC DATA DOWNLOADS     

            # Get a list of all the decompressed files
            list_of_decompressed_filepaths = [decompressed_temp_output_dir + file for file in os.listdir(decompressed_temp_output_dir)]
            
            betfair_odds = parallel_run_function(
                    func = betfair_file_to_final_odds,
                    input_list = list_of_decompressed_filepaths,
                    max_workers = config['num_threads'],
                    collapse_output_to_df = True,
                    odds_cutoff_time = betfair_data_config['odds_cutoff_time'])
            
            
            # Change the date to a proper date to avoid parquet errors
            betfair_odds['event_date'] = pd.to_datetime(betfair_odds['event_date'], errors = 'coerce')
            
            # Clear out the downloads folder
            clear_directory(download_folder)
            
            # Move the processed decompressed files                    
            for filename in os.listdir(decompressed_temp_output_dir):
                src = os.path.join(decompressed_temp_output_dir, filename)
                dst = os.path.join(decompressed_output_dir, filename)
            
                if os.path.isfile(src):
                    shutil.move(src, dst)  # Will overwrite if file exists
            
            # Join with the old odds
            final_odds = pd.concat([betfair_odds, existing_betfair_odds_df], axis = 0)
            
            return final_odds.drop_duplicates(subset = ['fighter', 'bout', 'event_date'])
    else:
        print("No new Betfair data to parse")
        return existing_betfair_odds_df