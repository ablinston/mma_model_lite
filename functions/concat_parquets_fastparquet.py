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