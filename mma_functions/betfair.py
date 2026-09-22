import betfairlightweight
from betfairlightweight import StreamListener
from betfairlightweight.filters import market_filter, price_projection, streaming_market_filter, streaming_market_data_filter

from datetime import datetime, date, timedelta, timezone
import pandas as pd
import pdb

def betfair_add_upcoming_bouts_details(betfair_df, upcoming_bouts_df, threshold = 75):
    """
    This function takes a data frame of the latest betfair odds for upcoming bouts, and matches them with upcoming bouts information from UFC stats to get a cleaned data frame
    
    Parameters:
    
    betfair_df: the upcoming betfair df data frame created
    upcoming_bouts_df: the upcoming events
    threshold: the % match minimum in order to match with the betfair exch odds
    
    """

    # pdb.set_trace()
    
    df = betfair_df.copy()
    
    # Find the closest names for the fighter
    df['fighter'] = df['fighter'].apply(lambda x: get_best_match(
        x,
        choices = upcoming_bouts_df['fighter1'].tolist() + upcoming_bouts_df['fighter2'].tolist(),
        threshold = threshold  # use token_sort_ratio for good name matching
    ))
    
    # Remove those not matched
    df = df[~df['fighter'].isna()]
    
    # Now get the event details
    fighter_events = pd.concat([
        upcoming_bouts_df[['fighter1', 'event', 'bout']].rename(columns = {'fighter1': 'fighter'}),
        upcoming_bouts_df[['fighter2', 'event', 'bout']].rename(columns = {'fighter2': 'fighter'})
        ])
    
    # Keep the date as the date and time of the odds
    df.rename(columns = {'gmt_datetime': 'notformodel_betfair_odds_datetime'}, inplace = True)
    
    # Merge these onto the odds data
    df = df[['fighter', 'last_price_traded', 'notformodel_betfair_odds_datetime']].merge(fighter_events, on = 'fighter', how = 'inner')
    
    # Clean names of column
    df.rename(columns = {'last_price_traded': 'betfair_exch_odds'}, inplace = True)

    return df

def betfair_bout_get_chosen_odds(bout_df, time_cutoff, average_last_n):
    """
    This function takes a data frame of bets for a given bout and returns the chosen odds to be used in modelling
    
    Args:
        df (DataFrame): a data frame from the betfair_stream_to_df function
        time_cutoff (time): A time cut off in the lead up to the event
        average_last_n (integer): the number of bets before cutoff to be averaged
    
    Returns:
        DataFrame: giving the bout information and the chosen betfair odds
    
    """

    # pdb.set_trace()
    
    df = bout_df.copy().sort_values('GMT_datetime')
    
    # Split out the datetime to allow easy analysis
    df['bet_date'] = df['GMT_datetime'].dt.date
    df['bet_time'] = df['GMT_datetime'].dt.time
    df['event_date'] = df['bet_date'].max()

    # Check if we're passed the cutoff time on the given day when the bout starts
    if datetime.strptime(time_cutoff, "%H:%M").time() < df.iloc[-1]["bet_time"]:
        cutoff_datetime = datetime.strptime(str(df['event_date'].iloc[0]) + ":" + time_cutoff, "%Y-%m-%d:%H:%M")
    else:
        cutoff_datetime = datetime.strptime(str(df['event_date'].iloc[0] - timedelta(days = 1)) + ":" + time_cutoff, "%Y-%m-%d:%H:%M")
    
    # Now filter the data after the cutoff time
    df_cutoff = df[df['GMT_datetime'] < cutoff_datetime]
    
    # Check whether that has returned any rows. If not, take the earliest bets. This is when there are last minute replacement fights and bets don't seem to happen until on the day.
    if len(df_cutoff) == 0 and len(df) > 0:
        df_cutoff = df.drop_duplicates(subset = ['bout', 'fighter'])
    
    # And finally only keep one record per fighter. Make it the average of the last 3 bets
    output_df = df_cutoff.groupby('fighter').tail(average_last_n).groupby('fighter').agg({'bout': 'first', 'event_date': 'first', 'last_price_traded': 'mean'}).reset_index()

    output_df.rename(columns = {'last_price_traded': 'betfair_exch_odds'}, inplace = True)          

    # Now get the best odds after the cutoff time
    df_best_odds_after_cutoff = df[(df['GMT_datetime'] >= cutoff_datetime) & (df['inplay'] == False)].groupby('fighter').agg({'last_price_traded': 'max'}).rename(columns = {'last_price_traded': 'notformodel_best_betfair_odds_after_cutoff'})
    
    # Add on the best odds
    output_df = output_df.merge(df_best_odds_after_cutoff, on = 'fighter', how = 'left')

    # Check the length is 2 so we have both fighters
    if len(output_df) > 2:
        print("Error: Data is not the length expected")
        pdb.set_trace()
        
    elif len(output_df) == 1:
        
        # If only have one fighter, then create the opposing odds by reversing the existing fighter's odds
        fighters = output_df['bout'].str.replace("@", "v").str.split(' v ')[0]
        
        # Check if we have yes and no runners
        if output_df['fighter'][0].replace('.', '').lower() in ['yes', 'no']:
            if output_df['fighter'][0].replace('.', '').lower() == 'no':
                output_df.loc[0, 'fighter'] = fighters[1]
            else:
                output_df.loc[0, 'fighter'] = fighters[0]
        
        # Initiate the missing fighter name
        missing_fighter = ''
        
        # Check for surnames only
        if fighters[0].split(' ')[-1] in output_df['fighter'][0] and fighters[0] != output_df['fighter'][0]:
            missing_fighter = fighters[1]
        elif fighters[1].split(' ')[-1] in output_df['fighter'][0] and fighters[1] != output_df['fighter'][0]:
            missing_fighter = fighters[0]
            
        # Check the fighter we have is in the list
        if missing_fighter == '' and output_df['fighter'][0] in fighters:
            # Find the name of the missing fighter
            missing_fighter = fighters[0] if output_df['fighter'][0] == fighters[1] else fighters[1]
            
        if missing_fighter != '':
            
            # Create the record for them
            output_df = pd.concat([output_df, output_df], axis = 0).reset_index(drop = True)
            output_df.loc[1, 'fighter'] = missing_fighter
            output_df.loc[1, 'betfair_exch_odds'] = 1 / (1 - 1 / output_df.loc[0, 'betfair_exch_odds'])
            
        else:
            print("Fighter name not matching")
            # pdb.set_trace()
            return pd.DataFrame() # After checking, they seem to be 'Decision' or 'Submission' and not relevant
        
        return output_df
    
    # Fix the bout name if it needs fixing
    if len(output_df) > 0:
        
        try:
        
            # Some bouts have @ instead of v
            output_df['bout'] = output_df['bout'].str.replace("@", "v")
            
            # Get a list of fighters
            fighters = output_df['bout'].iloc[0].split(" v ")
            
            if len(fighters) == 2:
                
                # First check we don't have 'yes' 'no' runners as fighters
                if fighters[0].lower() != "no" and fighters[1].lower() != "no":
                    
                    # If the bout doesn't contain the fighter
                    if output_df['fighter'].iloc[0] not in fighters:
                        
                        for i in range(0, len(output_df)):
                            for j in range(0, len(fighters)):
                                # If the fighter from bout is contained in the fighter name e.g. surname only, set the fighter name to full name
                                if fighters[j].lower() in output_df['fighter'].iloc[i].lower():
                                    fighters[j] = output_df['fighter'].iloc[i]
                
                        # Set the bout to the correct names
                        output_df['bout'] = fighters[0] + " v " + fighters[1]
        except:
            pdb.set_trace()
            
    # Fill in any missing best odds?    
    
    return output_df

def betfair_clean_fighter_names(betfair_df, event_details_df, stats_df, fuzz_threshold = 70, day_threshold = 2, surname_fuzz_threshold = 90):
    
    """
    This function makes the fighter names consistent with the names on the stats data. It also adds a cleaned opponent column
    
    This requires a betfair_df that has the fighter surname and opponent surname within it.
    
    """
    
    # Find the closest name matches
    
    # pdb.set_trace()
    
    # Merge the event date onto the stats
    merged_df = stats_df[['event', 'fighter', 'bout']].drop_duplicates().merge(event_details_df[['event', 'date']], how = 'left', on = 'event')
    merged_df.rename(columns = {'date': 'event_date'}, inplace = True)
    
    if "opponent" not in betfair_df.columns:
        add_opponent_column(betfair_df)
    
    if "opponent" not in merged_df.columns:
        add_opponent_column(merged_df)
    
    # Find the closest matches to the fighter
    matched_names = betfair_df[['fighter', 'event_date']].apply(lambda row: get_best_match_with_date(
        row,
        merged_df,
        date_col='event_date',
        text_col='fighter',
        threshold=fuzz_threshold,
        max_days=day_threshold
    ), axis = 1)
    
    # Now do the same for the opponent. Do a simple rename of fighter to opponent on the UFC data as we only need names
    matched_opponent_names = betfair_df[['opponent', 'event_date']].apply(lambda row: get_best_match_with_date(
        row,
        merged_df,
        date_col='event_date',
        text_col='opponent',
        threshold=fuzz_threshold,
        max_days=day_threshold
    ), axis = 1)
     
    # Replace the fighter name where we have a close match
    betfair_df['fighter'] = np.where(matched_names.isna(), betfair_df['fighter'], matched_names)
    betfair_df['opponent'] = np.where(matched_opponent_names.isna(), betfair_df['opponent'], matched_opponent_names)

    # Now look at fighters that have no match. Try to find matches based on the fights
    betfair_df_surname_matches = betfair_df.copy()
    
    # First clean the surname columns
    betfair_df.loc[:, ['fighter_surname', 'opponent_surname']] = clean_text_columns(betfair_df[['fighter_surname', 'opponent_surname']])
    
    # Find fights on similar dates where both fighters surnames are in the bout on UFC data
    betfair_df['event_date_m1'] = betfair_df['event_date'] + timedelta(days = -1)
    betfair_df['event_date_p1'] = betfair_df['event_date'] + timedelta(days = 1)
    
    # Prepare merged data for matching
    merged_df_to_match = merged_df.rename(columns = {'event': 'event_ufc', 'fighter' : 'fighter_ufc', 'bout': 'bout_ufc', 'opponent': 'opponent_ufc', 'event_date': 'event_date_ufc'})
    
    # Merge all bouts and fighters that take place within 1 day of the UFC event date
    merged_all = pd.concat([
        # 1 day prior
        betfair_df.merge(merged_df_to_match, left_on = 'event_date_m1', right_on = 'event_date_ufc'),
        betfair_df.merge(merged_df_to_match, left_on = 'event_date', right_on = 'event_date_ufc'),
        betfair_df.merge(merged_df_to_match, left_on = 'event_date_p1', right_on = 'event_date_ufc')
        ], axis = 0)
    
    # Find matches for the bouts
    
    merged_all['strict_match'] = merged_all.apply(lambda row: 1 if (
        row['fighter_surname'] in row['bout_ufc']) and (
            row['opponent_surname'] in row['bout_ufc']) and (
                row['fighter_surname'] in row['fighter_ufc']) 
                else 0, axis = 1)
    
    # Use a fuzzy contains rather than strict
    merged_all['match'] = merged_all.apply(lambda row: 1 if (
        fuzzy_contains(row['fighter_surname'], row['bout_ufc'], surname_fuzz_threshold)) and (
            fuzzy_contains(row['opponent_surname'], row['bout_ufc'], surname_fuzz_threshold)) and (
                fuzzy_contains(row['fighter_surname'], row['fighter_ufc'], surname_fuzz_threshold)) 
                else 0, axis = 1)
    
    # Find matches for the bouts
    surname_matches_df = merged_all[merged_all['match'] == 1].sort_values('strict_match', ascending = False)[['fighter', 'bout', 'event_date', 'fighter_ufc', 'bout_ufc', 'opponent_ufc']].drop_duplicates(subset = ['fighter', 'bout', 'event_date'])
    
    # Merge back onto the original Betfair data
    betfair_merged_ufc = betfair_df.merge(surname_matches_df, on = ['fighter', 'bout', 'event_date'], how = 'left')

    # Check we have the right number of rows
    stop_if_not(len(betfair_merged_ufc) == len(betfair_df), 'Surname match row number check')
        
    # For QA, can check the non-matches
    qa = betfair_merged_ufc[(betfair_merged_ufc['fighter'] != betfair_merged_ufc['fighter_ufc']) & ~(betfair_merged_ufc['fighter_ufc'].isna()) & (matched_names.reset_index(drop = True).isna())]
    qa.to_csv('./data/qa_betfair_fuzzy_matched_ufc_names.csv')
    
    
    # Now replace fighter names where we don't have the matches from the original fuzzy match
    betfair_df['fighter'] = np.where(
        matched_names.reset_index(drop = True).isna() & 
        ~betfair_merged_ufc['fighter_ufc'].reset_index(drop = True).isna(), 
        betfair_merged_ufc['fighter_ufc'], 
        betfair_df['fighter'])
    
    betfair_df['opponent'] = np.where(
        matched_opponent_names.reset_index(drop = True).isna() & 
        ~betfair_merged_ufc['opponent_ufc'].reset_index(drop = True).isna(), 
        betfair_merged_ufc['opponent_ufc'], 
        betfair_df['opponent'])
    
    # Remove columns no longer needed
    betfair_df.drop(columns = ['fighter_surname', 'opponent_surname', 'event_date_m1', 'event_date_p1'], inplace = True)

def betfair_extract_bz2(bz2_file_path, output_folder):
    
    """
    This function takes a path to a bz2 file and extracts it into a new folder
    
    """

    filename = os.path.basename(bz2_file_path).replace(".bz2", "")
    output_file_path = os.path.join(output_folder, filename).replace("\\", "/")
    
    # === Decompress to output directory ===
    with bz2.open(bz2_file_path, 'rb') as f_in:
        with open(output_file_path, 'wb') as f_out:
            f_out.write(f_in.read())
            
    return None
            

def betfair_file_to_final_odds(file_path, odds_cutoff_time, average_last_n, trading):
    
    """
    This function takes a path to a betfair file that has been extracted from bz2 and turns it into a dataframe
    with chosen odds given a set cutoff date
    
    """
    # pdb.set_trace()
    
    # create listener
    listener = StreamListener(max_latency=None)
    
    # Create historical stream (update file_path to your file location)
    stream = trading.streaming.create_historical_generator_stream(
        file_path = file_path,
        listener = listener,
    )

    # Convert the list of series into a data frame
    betfair_odds_df = betfair_stream_to_df(stream = stream)
    
    if len(betfair_odds_df) > 0:
        # Get the last odds before the chosen cutoff time
        betfair_odds_summary_df = betfair_bout_get_chosen_odds(betfair_odds_df, odds_cutoff_time, average_last_n)
    
        return betfair_odds_summary_df
    else:
        return pd.DataFrame()
            

def betfair_filter_bz2_for_fights(input_folder, output_folder):
    
    """
    This function takes a folder containing raw downloaded betfair bz2 files within subfolders and looks in each one to see if it's a fight.
    It will extract the files from the bz2 compression for those files that are fights and output them in the output folder
    
    Returns: DataFrame of the filenames and whether they are fights or not
    
    """
    
    # pdb.set_trace()
    
    # file_list = []
    # for path, subdirs, files in os.walk(input_folder):
    #     for name in files:
    #         file_list.append(path.replace('\\', '/') + '/' + name)
    file_list = [str(p.resolve()) for p in Path(input_folder).rglob("*.bz2")]

    # new_files_df = pd.DataFrame({'filepath': file_list, 'fight_flag': np.nan})

    # # Loop through the files to check which are fights and flag them on the data
    # for i in range(0, len(new_files_df)):
        
    #     if i % 100 == 0:
    #         print('Read in ' + str(i) + ' of ' + str(len(new_files_df)))
            
    #     with bz2.open(new_files_df['filepath'][i], "rt", encoding="utf-8") as f:
    #         for line in f:
    #             if "Fight Result" in line:
    #                 new_files_df.loc[i, 'fight_flag'] = 1
    #                 break
            
    #             new_files_df.loc[i, 'fight_flag'] = 0
    
    # Get the flags as to whether the file is a fight or not
    fight_flags = parallel_run_function(
        func = betfair_flag_bz2_as_fight,
        input_list = file_list,
        max_workers = config['num_threads'],
        collapse_output_to_df = False)
            
    fight_flags = pd.DataFrame(fight_flags)
    
    # fp.write('./data/betfair_fileflags.parquet', new_files_df)

    fight_files = fight_flags[fight_flags['fight_flag'] == 1]

    # for i in range(0, len(fight_files)):
        
    #     if i % 100 == 0:
    #         print('Written ' + str(i) + ' of ' + str(len(fight_files)))
        
    #     bz2_file_path = fight_files['filepath'].iloc[i]
        
    #     # === Create output filename by stripping .bz2 extension ===
    #     filename = os.path.basename(bz2_file_path).replace(".bz2", "")
    #     output_file_path = os.path.join(output_folder, filename).replace("\\", "/")
        
    #     # === Decompress to output directory ===
    #     with bz2.open(bz2_file_path, 'rb') as f_in:
    #         with open(output_file_path, 'wb') as f_out:
    #             f_out.write(f_in.read())
    
    # For each file, extract the bz2 into an output folder
    parallel_run_function(
        func = betfair_extract_bz2,
        input_list = fight_files['filepath'].str.replace("\\", "/").tolist(),
        max_workers = config['num_threads'],
        collapse_output_to_df = False,
        output_folder = output_folder)
                
    return fight_flags

def betfair_flag_bz2_as_fight(bz2_path):
    
    """
    This function takes a path to a bz2 file and flags whether it's a fight or not, returning dictionary indicating 1 or 0
    
    """
    # pdb.set_trace()
    try:
        with bz2.open(bz2_path, "rt", encoding="utf-8") as f:
            for line in f:
                if "Fight Result" in line:
                    print("found")
                    return {'filepath': bz2_path, 'fight_flag': 1}
                    break
    except:
        print(f"File not able to be opened: {bz2_path}")
        # pdb.set_trace()
        
    return {'filepath': bz2_path, 'fight_flag': 0}
            

# This function takes a Betfair stream and saves the data to a data frame

def betfair_get_all_upcoming_odds(betfair_login_config, betfair_data_config):
    
    """
    This function will get all the odds from upcoming MMA fights from betfair
    
    
    """
    
    # pdb.set_trace()
    
    
    ############################################
    ## Betfair standard setup
    
    # Open the connection with Betfair. This requires the betfair config which is not tracked by git because it contains passwords
    trading = betfairlightweight.APIClient(betfair_login_config['username'], betfair_login_config['password'], app_key = betfair_login_config['delay_feed_app_key'], certs = betfair_login_config['certs_location'])
    
    trading.login()
    
    # create queue
    output_queue = queue.Queue()
    
    # create stream listener
    listener = betfairlightweight.StreamListener(output_queue=output_queue)
    
    time.sleep(5)
    
    # create stream
    stream = trading.streaming.create_stream(listener=listener)
    
    time.sleep(5)
    
    ############################################
    ## Get market catalogue to get runner names and event names
    
    # Build your market filter
    market_catalogue_filter = {
        "eventTypeIds": [betfair_data_config['betfair_mma_event_id']],
        "marketTypeCodes": ["MATCH_ODDS"]
    }
    
    # Attempt to get the catalogue
    for attempt in range(10):
        try:
            market_catalogues = trading.betting.list_market_catalogue(
                filter = market_catalogue_filter,
                market_projection=["RUNNER_DESCRIPTION", "MARKET_DESCRIPTION", "EVENT"],
                max_results=100
            )
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Attempt {attempt + 1} for getting catalogue failed. Retrying.")
            time.sleep(5)
    
    
    
    ############################################
    ## STREAMING MMA
    
    # create filters
    market_filter = streaming_market_filter(
        event_type_ids=[betfair_data_config['betfair_mma_event_id']], market_types=["MATCH_ODDS"]
    )
    market_data_filter = streaming_market_data_filter(
        fields=["EX_LTP", "EX_MARKET_DEF", "EX_BEST_OFFERS_DISP"], ladder_levels = 3 # Last traded price and market definitions
    )
    
    # Try to get the stream
    for attempt in range(10):
        
        try:
            # subscribe
            streaming_unique_id = stream.subscribe_to_markets(
                market_filter=market_filter,
                market_data_filter=market_data_filter,
                conflate_ms=10000,  # send update every 1000ms
            )
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Attempt {attempt + 1} for getting odds stream failed. Retrying.")
            time.sleep(5)
    
    # start stream in a new thread (in production would need err handling)
    t = threading.Thread(target=stream.start, daemon=True)
    t.start()
    
    # Give a short delay for the connection
    time.sleep(5)
    
    # Try to get the snapshot
    for attempt in range(20):
        
        try:
            # Take a snapshot of the market right now
            market_books = listener.snap()
            
            # Check we retreived the market books
            if len(market_books) > 0:
                break
            else:
                error()
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Attempt {attempt + 1} for getting snapshot failed. Retrying.")
            # stream.stop()
            # stream = trading.streaming.create_stream(listener=listener)
            time.sleep(5)
            
    # pdb.set_trace()
    # Stop the stream now we have the snapshot
    stream.stop()
    
    # Save the results to a df
    betfair_latest_future_odds = betfair_stream_to_df(
        stream_snapshot = market_books, 
        market_catalogues = market_catalogues, 
        upcoming_min_best_offer = betfair_data_config['upcoming_min_best_offer'], 
        upcoming_odds_to_use = betfair_data_config['upcoming_odds_to_use'])
    
    # fp.write('./data/' + config['betfair_upcoming_odds_file_name'],
    #          betfair_latest_future_odds,
    #          compression = 'ZSTD')
  

    # Return the dataframe of odds
    return betfair_latest_future_odds

def betfair_get_runner_info_from_marketcatalogue(marketcatalogue):
    
    """
    This function takes a marketbook and extracts the information market name and runner ids with runner names
    
    Returns a dictionary of selection_id and runner names
    
    """
    # pdb.set_trace()
    
    # Initiate a runner dictionary
    runner_dict = {}
    
    for runner in marketcatalogue.runners:
        runner_dict.update({runner.selection_id: runner.runner_name})
    
    return runner_dict

# This function takes a Betfair stream and saves the data to a data frame

def betfair_stream_to_df(stream = None, stream_snapshot = None, inplay = False, include_prop_bets = False, market_catalogues = None, upcoming_min_best_offer = 0, upcoming_odds_to_use = 'last_price_traded'):
    
    # pdb.set_trace()
    
    # Initial checks
    if stream is None and stream_snapshot is None:
        print("Need a valid input stream")
        return None
    
    if stream_snapshot is not None and market_catalogues is None:
        print("stream_snapshot input needs a market catalogue for runner and market names")
        return None
    
    # Process the market catalogues if present
    if market_catalogues is not None:
        runners_dict = {k: v for d in map(betfair_get_runner_info_from_marketcatalogue, market_catalogues) for k, v in d.items()}
    
    if stream is not None:
        # create generator
        gen = stream.get_generator()
        all_market_books = gen()
    else:
        all_market_books = [stream_snapshot]
    
    # Create an empty odds object to be populated with the odds info
    betfair_odds_list = []
    
    # Create a file for each marketbook
    for market_books in all_market_books:
        for market_book in market_books:
                   
            try:
                # Get a list of the runners names and put them into a dictionary if we don't have the market catalogue
                if market_catalogues is None:
                    runners_dict = {}
                    for runner in market_book.market_definition.runners: # Runner names are contained in the marketbook
                        runners_dict = runners_dict | {runner.selection_id: runner.name}
                
                # Now loop through the runners and get the odds data
                for runner in market_book.runners:
                    
                    # pdb.set_trace()
                    
                    # Get the odds to use
                    if upcoming_odds_to_use == 'best_offer':
                        
                        # Use the minium to filter out offers
                        offers = [item for item in runner.ex.available_to_back if item.size > upcoming_min_best_offer]
                        
                        # Best offer is the first in the list
                        best_odds = offers[0].price
                        
                    else:
                        best_odds = runner.last_price_traded
                    
                    # Add a dictionary to the odds list for this runner
                    betfair_odds_list.append({
                        'GMT_datetime': market_book.publish_time, # is technically in UTC but that is the same as GMT in 24H clock
                        'market_id': market_book.market_id,
                        'bout': market_book.market_definition.event_name,
                        'status': market_book.status,
                        'inplay': market_book.inplay,
                        'runner': runners_dict[runner.selection_id],
                        'last_price_traded': best_odds
                    })
            except:
                # pdb.set_trace()
                pass
    
    # Get the date of the event
    output_df = pd.DataFrame(betfair_odds_list)
    
    # Get rid of NA bets
    output_df = output_df[~output_df['last_price_traded'].isna()]
    
    # # Remove any bets after the cutoff time ## THE LIVE STREAMS ONLY GIVE ONE SET OF ODDS PER FIGHT
    # if cutoff_time is not None:
    #     cutoff_date = datetime.strptime(str(output_df['GMT_datetime'].dt.date.max()) + "-" + cutoff_time, "%Y-%m-%d-%H:%M")
    #     output_df = output_df[output_df['GMT_datetime'] <= cutoff_date]
        
    # pdb.set_trace()
    
    # If no prop bets should be included, filter them out by looking for the fighters only in the runner column
    if not include_prop_bets:
        
        try:
            # Exclude common strings from prop bets
            exclusions = output_df['runner'].apply(lambda x: 1 if 'under' in x.lower() or 'over' in x.lower() or 'points' in x.lower() or '/' in x or ':' in x or ' round ' in x.lower() or ' by ' in x.lower() or 'draw ' in x.lower() or x.lower() == 'draw' else 0)
            output_df = output_df[exclusions == 0]
            
            # If this isn't a live stream, each marketbook will have only 2 fighters, so take the two most common
            if stream_snapshot is None:
                # Get the two most common bets. Assume these are the fighters
                two_modes = output_df['runner'].value_counts().head(2).index.tolist()
                
                # Filter only the two most common bets
                output_df = output_df[output_df['runner'].isin(two_modes)]
                
            output_df.rename(columns = {'runner': 'fighter'}, inplace = True)
        except:
            pdb.set_trace()
    
    # Filter out in play
    if not inplay:
        output_df = output_df[(output_df['inplay'] == False) & (output_df['status'] == "OPEN")]
            
    # Convert the list of series into a data frame
    return output_df

def betfair_yesno_to_names(df):
    
    """
    This function takes a data frame of betfair odds and bouts and changes 'yes' or 'no' fighters into the names via the bout column
    
    """
    
    # Split the bout
    fighter = df['bout'].str.split(" v ", expand = True)
    
    # Replace any yes and nos with these names
    df['fighter'] = np.where(df['fighter'].str.lower().isin(['yes', 'yes.']), fighter[0],
                             np.where(df['fighter'].str.lower().isin(['no', 'no.']), fighter[1], df['fighter']))
    
    return df
            

def update_betfair_odds(existing_betfair_odds_df, betfair_login_config, download_folder, decompressed_output_dir, decompressed_temp_output_dir):
    
    """
    This function updates the raw betfair odds historic data by downloading new files
    
    """
    # pdb.set_trace()
    
    # Initiate the trading object
    trading = betfairlightweight.APIClient(betfair_login_config['username'], betfair_login_config['password'], app_key = betfair_login_config['delay_feed_app_key'], certs = betfair_login_config['certs_location'])

    trading.login()
            
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
        
        # Try to get the stream
        for attempt in range(10):
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
                if len(file_list) > 0:
                    break
            except:
                print(f"No betfair files found to download. Check that the current month, {new_date.month}, has been 'purchased'.")
                time.sleep(5)
                # file_list = []
    
        # Check there are files to parse
        if len(file_list) > 0:
            
            # download the files
            for file in file_list:
                # print(file)
                # Make 5 attempts at download before quit
                for attempt in range(5):
                    try:
                        download = trading.historic.download_file(file_path = file, store_directory = download_folder)
                        break
                    # print(download)
                    except:
                        continue
                    
                    # All 5 attempts failed
                    pdb.set_trace()
            
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
                    max_workers = 1, # DO NOT USE MORE CORES FOR PROCESS. THE ODDS GO OFF MAYBE BECAUSE OF STREAM?
                    collapse_output_to_df = True,
                    odds_cutoff_time = betfair_data_config['odds_cutoff_time'],
                    average_last_n = betfair_data_config['last_n_bets_to_average'],
                    trading = trading)
            
            
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
