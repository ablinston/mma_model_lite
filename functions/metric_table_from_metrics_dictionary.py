# Converts the metrics saved in dictionaries into a pandas dataframe

def metric_table_from_metrics_dictionary(data_dict, col):
    
    # Extract the relevant data for the given 'col'
    if col not in data_dict:
        raise ValueError(f"Column '{col}' not found in the dictionary.")
    
    # Extract and convert the data to DataFrame
    groups = data_dict[col]
    df = pd.DataFrame(groups).transpose()  # Transpose so groups become rows
    
    # Reset index to have groups as rows
    df = df.reset_index().rename(columns={'index': 'group'})
    
    return df