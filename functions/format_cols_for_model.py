# This function sets consistent column formats for modelling

def format_cols_for_model(data):
   
    # Convert all dates to integers
    date_cols = [col for col in data.columns if data.dtypes[col] == 'datetime64[ns]']
    data[date_cols] = data[date_cols].astype('int64')
    
    return data
