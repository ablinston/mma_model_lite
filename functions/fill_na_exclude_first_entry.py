# This function takes a column and fills NAs with 0 except where it is the first entry for a certain group

def fill_na_exclude_first_entry(df, group_by, column_to_fill):
        # Create a mask to identify the first entry in each group
    df['is_first'] = df.groupby(group_by).cumcount() == 0
    
    # Fill NAs with 0, except for the first entry in each group
    df[column_to_fill] = df.apply(
        lambda x: 0 if pd.isna(x[column_to_fill]) and not x['is_first'] else x[column_to_fill], axis=1
    )
    
    # Drop the auxiliary 'is_first' column
    df = df.drop('is_first', axis=1)
    
    return df[column_to_fill]
