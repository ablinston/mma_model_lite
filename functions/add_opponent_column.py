
def add_opponent_column(df):
    """
    Adds an 'opponent' column to the DataFrame based on the 'fighter' and 'bout' columns.
    Assumes 'bout' is in 'fighter vs. opponent' or 'opponent vs. fighter' format.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing 'fighter' and 'bout' columns.

    Returns:
    pd.DataFrame: The DataFrame with an added 'opponent' column.
    """
    # Split 'bout' into two columns, 'fighter_1' and 'fighter_2'
    fighter_pairs = df['bout'].str.split('vs\.', expand=True)
    df['fighter_1'] = fighter_pairs[0]
    df['fighter_2'] = fighter_pairs[1]

    # Use np.where to assign 'opponent' based on whether 'fighter' matches 'fighter_1' or 'fighter_2'
    df['opponent'] = np.where(df['fighter'] == df['fighter_1'], df['fighter_2'], df['fighter_1'])

    # Drop the temporary columns used for the split
    df.drop(columns=['fighter_1', 'fighter_2'], inplace=True)
    
    return df

# Example usage
# new_odds_df = add_opponent_column(new_odds_df)
# print(new_odds_df[['fighter', 'bout', 'opponent']])
