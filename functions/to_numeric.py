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

