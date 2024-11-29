def move_columns(df, cols_to_move=[], ref_col='', place=''):
    """
    reoder columns in df
    move a list of columns before or after a reference column
    taken from https://towardsdatascience.com/reordering-pandas-dataframe-columns-thumbs-down-on-standard-solutions-1ff0bc2941d5

    arguments:
    df (df): a dataframe
    cols_to_move (list): list of columns to move
    ref_col (str): reference column on where to move list of columns
    place (str): where to place list of columns, enter 'before' or 'after'

    """
    cols = df.columns.tolist()
    if place == 'after':
        seg1 = cols[:list(cols).index(ref_col) + 1]
        seg2 = cols_to_move
    if place == 'before':
        seg1 = cols[:list(cols).index(ref_col)]
        seg2 = cols_to_move + [ref_col]
    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols if i not in seg1 + seg2]
    return df[seg1 + seg2 + seg3]
