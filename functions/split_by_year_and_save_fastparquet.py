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
