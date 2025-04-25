# This function takes a Betfair stream and saves the data to a data frame

def betfair_stream_to_df(stream, historic = True, inplay = False, include_prop_bets = False):
    
    if historic:
        # create generator
        gen = stream.get_generator()
    
    
    # Create an empty odds object to be populated with the odds info
    betfair_odds_list = []
    
    # Create a file for each marketbook
    for market_books in gen():
        for market_book in market_books:
                        
            # Get a list of the runners names and put them into a dictionary
            runners_dict = {}
            for runner in market_book.market_definition.runners: # Runner names are contained in the marketbook
                runners_dict = runners_dict | {runner.selection_id: runner.name}
            
            # Now loop through the runners and get the odds data
            for runner in market_book.runners:
                
                # pdb.set_trace()
                
                # Add a dictionary to the odds list for this runner
                betfair_odds_list.append({
                    'GMT_datetime': market_book.publish_time, # is technically in UTC but that is the same as GMT in 24H clock
                    'market_id': market_book.market_id,
                    'bout': market_book.market_definition.event_name,
                    'status': market_book.status,
                    'inplay': market_book.inplay,
                    'runner': runners_dict[runner.selection_id],
                    'last_price_traded': runner.last_price_traded
                })
    
    # Get the date of the event
    output_df = pd.DataFrame(betfair_odds_list)
    
    # Get rid of NA bets
    output_df = output_df[~output_df['last_price_traded'].isna()]
    
    # pdb.set_trace()
    
    # If no prop bets should be included, filter them out by looking for the fighters only in the runner column
    if not include_prop_bets:
        
        try:
            # Exclude common strings from prop bets
            exclusions = output_df['runner'].apply(lambda x: 1 if 'under' in x.lower() or 'over' in x.lower() or 'points' in x.lower() or '/' in x or ':' in x or ' round ' in x.lower() or ' by ' in x.lower() or 'draw ' in x.lower() else 0)
            output_df = output_df[exclusions == 0]
            
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
