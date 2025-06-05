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
