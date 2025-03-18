
def undo_clean_inf(df, threshold=99999):
    """
    Replaces all values >= threshold with np.inf and all values <= -threshold with -np.inf.
    Operates directly on numeric columns for maximum speed.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        threshold (int, optional): The threshold value. Defaults to 99999.

    Returns:
        pd.DataFrame: A modified DataFrame with replacements.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    # Ignore dates
    numeric_cols = [col for col in numeric_cols if ('date' not in col and 'dob' not in col) or 'diff' in col]
    
    arr = df[numeric_cols].values  # Convert to NumPy array for speed

    # Vectorized replacement
    arr[arr >= threshold] = np.inf
    arr[arr <= -threshold] = -np.inf

    df[numeric_cols] = arr  # Assign back to DataFrame
    return df
