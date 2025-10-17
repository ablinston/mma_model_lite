# This function cleans a text column. It's used for the matching columns to ensure they're all done in a consistent way
def clean_text_columns(df):
    
    # Loop through the columns. This is faster than .apply with lambda
    for col in df.columns:
        df.loc[:, col] = df[col].str.replace(' ', '')
        # Remove capitals and special characters
        df.loc[:, col] = df[col].str.lower()
        df.loc[:, col] = df[col].str.replace('-', '')
        df.loc[:, col] = df[col].str.replace("'", "")

        # If looking at bouts, need to put in a placeholder for vs. before getting rid of full stops
        if col in ['bout', 'BOUT']:
            df.loc[:, col] = df[col].str.replace('vs.', 'vs##')
        df.loc[:, col] = df[col].str.replace('.', '')
        if col in ['bout', 'BOUT']:
            df.loc[:, col] = df[col].str.replace('vs##', 'vs.')
        df.loc[:, col] = df[col].str.replace('saint', 'st')
        
        # Clean any names that are equivalent
        if col in ['fighter', 'FIGHTER']:
            df.loc[:, col] = df[col].str.replace('montserendon', 'montserratrendon')
            df.loc[:, col] = df[col].str.replace('michaelaswell', 'michaelaswelljr')
        
    return df
