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
