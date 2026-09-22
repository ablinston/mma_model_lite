def check_identical_excl_string(list1, list2, string1, string2, check_both = False):
    """
    This function checks whether two lists are identical once strings are removed from each

    arguments:

    """
    list1 = [element.replace(string1, '') for element in list1]
    list2 = [element.replace(string2, '') for element in list2]
    
    if check_both:
        list1 = [element.replace(string2, '') for element in list1]
        list2 = [element.replace(string1, '') for element in list2]

    return list1 == list2

def clear_directory(path):
    # Use scandir for faster performance with many small files
    with os.scandir(path) as entries:
        for entry in entries:
            try:
                if entry.is_file() or entry.is_symlink():
                    os.unlink(entry.path)
                elif entry.is_dir():
                    shutil.rmtree(entry.path)
            except Exception as e:
                print(f"Failed to delete {entry.path}: {e}")

def load_parquet(file):
    """Helper function to load a single Parquet file into a DataFrame."""
    return fp.ParquetFile(file).to_pandas()

def concat_parquets_fastparquet(folder_path):
    """
    Reads all parquet files in a folder using fastparquet and concatenates them into a single DataFrame.
    
    Args:
        folder_path (str): The folder containing parquet files.
        
    Returns:
        pd.DataFrame: The concatenated DataFrame.
    """
    # Get all parquet file paths in the folder
    parquet_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.parquet')]
    if not parquet_files:
        raise ValueError(f"No parquet files found in folder: {folder_path}")

    # Parallelize reading of Parquet files
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_parquet, file) for file in parquet_files]
        
        # Collect DataFrames as they are read
        dataframes = [future.result() for future in as_completed(futures)]

    # Concatenate all DataFrames into one
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Loaded {len(parquet_files)} files into a single DataFrame.")
    
    return combined_df

def dataframes_equal_ignore_types(df1, df2, sort_rows=True):
    # Check columns match (ignoring order)
    if set(df1.columns) != set(df2.columns):
        return False

    # Align columns in the same order
    df1 = df1[sorted(df1.columns)]
    df2 = df2[sorted(df2.columns)]

    # Convert all values to string (or float for numeric matching)
    df1_converted = df1.astype(str)
    df2_converted = df2.astype(str)

    # Optionally sort rows to ignore row order
    if sort_rows:
        df1_converted = df1_converted.sort_values(by=list(df1_converted.columns)).reset_index(drop=True)
        df2_converted = df2_converted.sort_values(by=list(df2_converted.columns)).reset_index(drop=True)

    return df1_converted.equals(df2_converted)

# This function converts elements of a dictionary to numeric where possible
def dict_to_numeric(dict):
    
    return pd.to_numeric(pd.Series(dict), errors='coerce').fillna(pd.Series(dict)).to_dict()

# This function takes a column and fills NAs with 0 except where it is the first entry for a certain group

def fill_na_exclude_first_entry(df, group_by, column_to_fill):
        # Create a mask to identify the first entry in each group
    df['is_first'] = df.groupby(group_by).cumcount() == 0
    
    # Fill NAs with 0, except for the first entry in each group
    df[column_to_fill] = df.apply(
        lambda x: 0 if pd.isna(x[column_to_fill]) and not x['is_first'] else x[column_to_fill], axis=1
    )
    
    # Drop the auxiliary 'is_first' column
    df = df.drop('is_first', axis=1)
    
    return df[column_to_fill]

# This function gives a true or false if one string is fuzzily contained in another

def fuzzy_contains(substring, text, threshold=80):
    return fuzz.partial_ratio(substring.lower(), text.lower()) >= threshold

# This function uses rapid fuzz to get a fuzzy match

def get_best_match(x, choices, threshold=70):
    match = process.extractOne(x, choices, scorer=fuzz.token_sort_ratio)
    if match and match[1] >= threshold:
        return match[0]
    else:
        return None

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

# This function gets the correct suffix for a day of the month
def get_day_suffix(day):
    # Determine the suffix for the day
    if 10 <= day % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    return suffix

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

# This function moves a column in a pandas data frame
def move_column(df, column_name, new_position):
    col = df.pop(column_name)  # Remove the column
    df.insert(new_position, column_name, col)  # Insert it at the new position
    return df

def move_columns(df, cols_to_move=[], ref_col='', place=''):
    """
    reoder columns in df
    move a list of columns before or after a reference column
    taken from https://towardsdatascience.com/reordering-pandas-dataframe-columns-thumbs-down-on-standard-solutions-1ff0bc2941d5

    arguments:
    df (df): a dataframe
    cols_to_move (list): list of columns to move
    ref_col (str): reference column on where to move list of columns
    place (str): where to place list of columns, enter 'before' or 'after'

    """
    cols = df.columns.tolist()
    if place == 'after':
        seg1 = cols[:list(cols).index(ref_col) + 1]
        seg2 = cols_to_move
    if place == 'before':
        seg1 = cols[:list(cols).index(ref_col)]
        seg2 = cols_to_move + [ref_col]
    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols if i not in seg1 + seg2]
    return df[seg1 + seg2 + seg3]

def parallel_run_function(
        func,
        input_list: list,
        max_workers: int,
        collapse_output_to_df: bool = False,
        **kwargs):

    """
    This function takes a function and applies it to a list with parallel runs
    
    """

    # Set up a list to contain all the results
    output_list = []
    completed_count = 0
    
    # Run the function in parallel
    with ThreadPoolExecutor(max_workers = max_workers) as executor:
        future_to_url = {executor.submit(func, item, **kwargs): item for item in input_list}

        # Collect results as they complete
        for future in as_completed(future_to_url):
            try:
                result = future.result()
                # Add the result to the list
                output_list.append(result)
                
            except Exception as e:
                print(f"Error for item {future}: {e}")

            # Increment the completed count
            completed_count += 1
            if completed_count % 100 == 0:
                print('Completed ' + str(completed_count) + ' of ' + str(len(input_list)))
                
    # This ensures that all threads are done
    executor.shutdown(wait=True)
    
    if collapse_output_to_df:
        return pd.concat(output_list, axis = 0)
    else:
        return output_list

def safe_divide_np(numerator, denominator, zero_zero_fill = 1):
    # Perform element-wise division using np.divide
    result = np.divide(numerator, denominator, where=denominator != 0)
        
    # Handle the case where denominator is zero
    result[denominator == 0] = np.inf
    
    # Handle the case where both numerator and denominator are zero
    result[(numerator == 0) & (denominator == 0)] = zero_zero_fill
    
    return result

# # Example usage
# numerator = np.array([[10, 0, 30, 0, 50], [10, 0, 30, 0, 50]])
# denominator = np.array([[2, 0, 5, 0, 0], [10, 0, 30, 0, 50]])

# result = safe_divide_np(numerator, denominator)
# print(result)

# Function to handle different cases for the mode
def safe_mode(x):
    modes = x.mode()
    if modes.empty:
        return pd.NaT  # Return NaT if no mode found
    else:
        return modes.iloc[0]  # Take the first mode if it exists

def save_group_to_parquet(year, group, output_folder, compression):
    """Helper function to save a specific year group to a parquet file."""
    file_path = os.path.join(output_folder, f"{year}.parquet")
    fp.write(file_path, group, compression=compression)
    print(f"Saved {file_path}")

def split_by_year_and_save_fastparquet(df, date_column, output_folder, compression='ZSTD'):
    """
    Splits the DataFrame into parts by year and saves each part as a parquet file using fastparquet.
    
    Args:
        df (pd.DataFrame): The DataFrame to split.
        date_column (str): The name of the column containing date values.
        output_folder (str): The folder where parquet files will be saved.
        compression (str): Compression type for parquet files (default is ZSTD).
        
    Raises:
        ValueError: If the date_column is not in the DataFrame or isn't in datetime format.
    """
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in the DataFrame.")
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        raise ValueError(f"Column '{date_column}' must be in datetime format.")
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract unique years
    years = df[date_column].dt.year.unique()

    # Create ThreadPoolExecutor for concurrent file writing
    with ThreadPoolExecutor() as executor:
        futures = []
        # For each year, filter the DataFrame and save it
        for year in years:
            group = df[df[date_column].dt.year == year]
            # Submit the task to the executor and store the future
            futures.append(executor.submit(save_group_to_parquet, year, group, output_folder, compression))

        # Wait for all threads to complete
        for future in as_completed(futures):
            future.result()  # This will raise any exceptions encountered during execution

def split_of_columns(df, col_list):
    """
    this function will take a data frame and a list of columns and split the text by " of "
    then pass them into two new columns suffixed with _landed and _attempt

    arguments:
    df: pandas data frame
    col_list: list of columns

    returns:
    df
    """
    
    # Loop through each column
    for col in col_list:
        try:
            df[[col + '_landed', col + '_attempt']] = df[col].str.split(" of ", expand = True).apply(pd.to_numeric)
        except:
            print(f'{col} column encountered error')
        # df[col + '_accuracy'] = df[col + '_landed'] / df[col + '_attempt']

    return df.drop(columns = col_list)

def time_to_minutes_decimal(series):
    """
    Takes a string time with format `mm:ss` and converts it to numeric minutes decimal

    arguments:
    series (str): series or pandas column

    returns:
    series
    """
   # pdb.set_trace()
    df = pd.DataFrame(columns = ['minutes', 'seconds'])
    df[['minutes', 'seconds']] = series.str.replace('-', '').str.split(":", expand = True)
    df = to_numeric(df, ['minutes', 'seconds'])

    return df['minutes'] + df['seconds'] / 60

def to_numeric(df, cols):
    """
    converts a set of columns in pandas to numeric by replacing all non numeric with '' and extracting numbers

    arguments:
    df (df): a pandas data frame
    cols: a list of columns

    """
    for col in cols:
        try:
            df[col] = df[col].fillna(np.nan)
        except:
            pass
        try:
            df[col] = pd.to_numeric(df[col].apply(lambda x: sub("[^0-9]", "", str(x))), errors = 'coerce')
        except:
            pass
        try:
            df[col] = pd.to_numeric(df[col].apply(lambda x: sub("[^0-9]", "", x)), errors = 'coerce')
        except:
            pass

        df[col] = df[col].astype(float)
    return df
