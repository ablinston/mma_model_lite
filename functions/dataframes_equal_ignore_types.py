def dataframes_equal_ignore_types(df1, df2, sort_rows=True):
    # Check columns match (ignoring order)
    if set(df1.columns) != set(df2.columns):
        return False

    # Align columns in the same order
    df1 = df1[sorted(df1.columns)]
    df2 = df2[sorted(df2.columns)]

    # Convert all values to string (or float for numeric matching)
    df1_converted = df1.astype(str)
    df2_converted = df2.astype(str)

    # Optionally sort rows to ignore row order
    if sort_rows:
        df1_converted = df1_converted.sort_values(by=list(df1_converted.columns)).reset_index(drop=True)
        df2_converted = df2_converted.sort_values(by=list(df2_converted.columns)).reset_index(drop=True)

    return df1_converted.equals(df2_converted)