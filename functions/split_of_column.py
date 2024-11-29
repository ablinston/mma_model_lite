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
