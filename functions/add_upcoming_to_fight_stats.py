# This function takes the fight stats and upcoming fight details data and adds empty upcoming fight stats onto the fight stats data frame
# for the upcoming fights

def add_upcoming_to_fight_stats(fight_stats, upcoming_fight_details_df, events, upcoming_events):
        
    # Add upcoming events
    all_events = pd.concat([upcoming_events, events], axis = 0)

    # Add the date to the upcoming fights
    upcoming_fight_details = upcoming_fight_details_df.merge(all_events[['event', 'date']], on = 'event', how = 'left')

    # Create empty stats rows for each fight
    upcoming_fight_details[['fighter1', 'fighter2']] = upcoming_fight_details['bout'].str.split('vs\\.', expand = True)
    upcoming_fight_stats = pd.concat([
        upcoming_fight_details[['event', 'bout', 'fighter1']].rename(columns = {'fighter1': 'fighter'}),
        upcoming_fight_details[['event', 'bout', 'fighter2']].rename(columns = {'fighter2': 'fighter'})
        ])
    
    return pd.concat([upcoming_fight_stats, fight_stats])
    
    