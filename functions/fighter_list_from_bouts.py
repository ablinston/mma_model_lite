# This function takes a pd series of bouts (without cleaning so WITH spaces) and transforms it into a unique
# list of fighters fighting in those bouts

def fighter_list_from_bouts(bout_series):
    
    # bout_series = upcoming_fight_details_df['BOUT']
    
    df = pd.DataFrame()
    
    # First split the bouts
    df[['f1', 'f2']] = bout_series.str.split(' vs\\. ', expand = True)
    
    return list(set(df['f1'].to_list() + df['f2'].to_list()))
