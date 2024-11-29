def longest_streak(series, target_value):
    """
    Calculates the longest streak of consecutive occurrences of the target value in a pandas Series,
    ignoring `0` but including streaks interrupted by other non-zero values.

    Parameters:
    - series (pd.Series): The input Series.
    - target_value (float64): The value (e.g., 1 or -1) for which to find the longest streak.

    Returns:
    - int: The length of the longest streak for the target value.
    """
    
    
    # pdb.set_trace()
    # Replace `0` with `None` to break streaks
    filtered = series[series != 0]

    # Create a mask for consecutive `target_value`
    mask = filtered == target_value

    # Group by changes in the mask and count streak lengths
    streak_lengths = mask.groupby((~mask).cumsum()).sum()

    # Return the longest streak length
    return streak_lengths.max() or 0
