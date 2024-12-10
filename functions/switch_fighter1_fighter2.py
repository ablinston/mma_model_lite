# This function switches who is fighter1 and fighter2 on the merged final dataset

def switch_fighter1_fighter2(df_original, filter_boolean):
    """
    # This functions takes a dataframe and a filter of same length and switches fighter1 and fighter2 in the rows labelled 'True'
    
    
    """

    # Define the data to be switched
    df = df_original[filter_boolean]

    # First rename fighter1 columns
    df.columns = df.columns.str.replace('fighter1', 'fighterX')

    # Now rename fighter2
    df.columns = df.columns.str.replace('fighter2', 'fighter1')

    # Now create the final name for fighter1
    df.columns = df.columns.str.replace('fighterX', 'fighter2')

    # Now reverse and diff columns
    diff_columns = [col for col in df.columns if 'diff' in col and 'diff_ratio' not in col]
    df[diff_columns] = -df[diff_columns]

    diff_ratio_columns = [col for col in df.columns if 'diff_ratio' in col]
    df[diff_ratio_columns] = 1 / df[diff_ratio_columns]

    ## Now change individual columns

    # Reverse the matchup
    fighter1, fighter2 = df['matchup'].str.split('-vs-')
    df['matchup'] = fighter2 + "-vs-" fighter1

    # Change the outcomes
    df['outcome'] = np.where(df['outcome'] == 'W/L', 'L/W', npwhere(df['outcome'] == 'L/W', 'W/L', df['outcome']))

    # Now combine this data with the rows not switched
    final_df = pd.concat([df, df_original[~filter_boolean]])

    # Check we have the correct number of rows
    stop_if_not(len(final_df) == len(df_original), 'Dataset length check')

    return final_df