# This function takes a pandas series and works out what the minimum possible sum is from all starting points and all points, summing in sequence. This is used to calculate max drawdown

def minimum_sum_series(series):
    """
    Calculate the minimum sum for every starting point and subarray length in a pandas Series.
    
    Args:
        series (pd.Series): Input series containing -1, 0, and 1.
    
    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to a starting index,
                      and each column corresponds to the subarray length, containing
                      the minimum sums.
    """
    n = len(series)
    arr = series.to_numpy()
    
    # Compute cumulative sum
    cumsum = np.zeros(n + 1)
    cumsum[1:] = np.cumsum(arr)
    
    # Create a 2D array of sums for all subarray lengths
    # Sums are calculated using broadcasting of the cumulative sums
    subarray_sums = cumsum[np.newaxis, :] - cumsum[:, np.newaxis]
    
    # Keep only valid sums (upper triangular part)
    valid_sums = np.triu(subarray_sums, k=1)  # k=1 excludes the diagonal
    
    # Remove the cumulative sum offset for valid lengths
    min_sums = np.full((n, n), np.inf)
    for length in range(1, n + 1):
        min_sums[:n - length + 1, length - 1] = valid_sums[np.arange(n - length + 1), np.arange(length, n + 1)]
    
    return np.min(min_sums)

