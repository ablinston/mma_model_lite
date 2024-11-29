# This function cleans a text column. It's used for the matching columns to ensure they're all done in a consistent way
def clean_text_columns(df):
    
    # Loop through the columns. This is faster than .apply with lambda
    for col in df.columns:
        df.loc[:, col] = df[col].str.replace(' ', '')
        
    return df
