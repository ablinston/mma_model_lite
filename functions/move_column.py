# This function moves a column in a pandas data frame
def move_column(df, column_name, new_position):
    col = df.pop(column_name)  # Remove the column
    df.insert(new_position, column_name, col)  # Insert it at the new position
    return df