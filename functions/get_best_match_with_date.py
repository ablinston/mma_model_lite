def get_best_match_with_date(
    row,
    df2,
    date_col='date',
    text_col='name',
    threshold=85,
    max_days=2
):
    """
    Finds the best fuzzy match from df2 for a given row in df1,
    based on:
      - fuzzy matching on a text column (e.g., 'name')
      - a date column being within ±N days (default 2 days)

    Parameters:
    ----------
    row : pd.Series
        A single row from df1 (used with apply).
    df2 : pd.DataFrame
        The DataFrame to match against.
    date_col : str
        Name of the date column in both dataframes.
    text_col : str
        Name of the text column to fuzzy match on.
    threshold : int
        Minimum fuzzy match score (0–100) to accept a match.
    max_days : int
        Maximum number of days allowed between dates (±N days).

    Returns:
    -------
    str or None
        A str with the fighter name from the stats df.
    """

    # pdb.set_trace()

    # Define the time window for date tolerance
    date_min = row[date_col] - pd.Timedelta(days=max_days)
    date_max = row[date_col] + pd.Timedelta(days=max_days)

    # Filter df2 to only include rows within the date window
    candidates = df2[(df2[date_col] >= date_min) & (df2[date_col] <= date_max)]

    # If no date-match candidates, return None early
    if candidates.empty:
        return None

    # Use RapidFuzz to find the best fuzzy text match among the candidates
    match = process.extractOne(
        query=row[text_col],
        choices=candidates[text_col].tolist(),
        scorer=fuzz.token_sort_ratio  # use token_sort_ratio for good name matching
    )

    # If a good enough match is found
    if match and match[1] >= threshold:
        # pdb.set_trace()
        # Find the actual row in candidates that matched
        matched_row = candidates[candidates[text_col] == match[0]]
        return matched_row[text_col].iloc[0]

    return None  # No suitable match found
