# This removes duplicates where a fighter appears in a data frame on the same date more than once, as either fighter1 or fighter2

def remove_duplicate_fighter_pairs(df):
    """
    Removes duplicate rows from the DataFrame where each fighter appears only once per date,
    considering both 'fighter1' and 'fighter2' columns.

    Parameters:
    - data (dict): Dictionary containing 'date', 'fighter1', and 'fighter2' columns as lists.

    Returns:
    - pd.DataFrame: Cleaned DataFrame with duplicates removed.
    """
    # pdb.set_trace()
    
    # Initialize a set to track unique fighter-date pairs
    seen_fighters = set()
    rows_to_keep = []

    # Iterate over each row and check if fighters have appeared on that date
    for idx, row in df.iterrows():
        date = row['date']
        fighter1 = row['fighter1']
        fighter2 = row['fighter2']

        # Create unique keys for each fighter with the date
        key1 = (date, fighter1)
        key2 = (date, fighter2)

        # Check if both fighters are new for this date
        if key1 not in seen_fighters and key2 not in seen_fighters:
            rows_to_keep.append(idx)  # Keep this row
            seen_fighters.add(key1)
            seen_fighters.add(key2)

    # Create a cleaned DataFrame with only the rows we decided to keep
    df_cleaned = df.loc[rows_to_keep].reset_index(drop=True)

    return df_cleaned
