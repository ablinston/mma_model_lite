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
            