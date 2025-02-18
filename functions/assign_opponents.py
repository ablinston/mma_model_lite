def assign_opponents(df):
    """
    Assigns opponent to each fighter in the DataFrame.
    """
    df = df.copy().reset_index(drop = True)
    df["opponent"] = df["fighter"].shift(-1)
    df.loc[df.index % 2 == 1, "opponent"] = df["fighter"].shift(1)
    return df