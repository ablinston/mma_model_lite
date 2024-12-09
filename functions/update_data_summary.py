
def update_data_summary(results_csv_path, data_frames, summary_date, save_output = False):
    """
    Updates the results dataframe with summary statistics of the provided dataframes.
    Each date will be a column with the summary for each dataframe.

    Parameters:
    - results_df (pd.DataFrame): The dataframe where the summary will be added.
    - data_frames (dict): A dictionary of dataframe names as keys and the dataframes as values.
    - summary_dates (list): A list of dates for each summary to be logged. Example: ['2024-12-09', '2024-12-10']

    Returns:
    - pd.DataFrame: The updated results dataframe with new summary columns.
    """
    
    pdb.set_trace()
    
    results_df = pd.read_csv(results_csv_path, index_col = 0)
    
    # Create an empty list to store the summary data for each dataframe
    summary_data = []
    
    for df_name, df in data_frames.items():
        # Get the number of rows and columns
        num_rows, num_cols = df.shape
        
        # Count the number of NaNs in each column
        na_count = df.isna().sum()
        
        # Create a dictionary for the summary of this dataframe
        summary_row = {
            'date': summary_date,
            'dataframe': df_name,
            'rows': num_rows,
            'columns': num_cols,
            **na_count.to_dict()  # Add NaN counts as columns
        }
        
        # Append the summary row to the summary_data list
        summary_data.append(summary_row)
    
    # Convert the summary data to a DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Ensure the dates all correct
    summary_df['date'] = summary_date
    
    # Append the new summary to the results_df
    updated_results_df = pd.concat([results_df, summary_df], ignore_index=True)

    # Save to file before outputting
    if save_output:
        updated_results_df.to_csv(results_csv_path, index = True)

    return updated_results_df
