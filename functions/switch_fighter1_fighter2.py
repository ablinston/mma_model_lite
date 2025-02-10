# This function switches who is fighter1 and fighter2 on the merged final dataset

def switch_fighter1_fighter2(df_original = None, filter_boolean = None, X = None, y = None, for_model = True):
    """
    # This functions takes a dataframe and a filter of same length and switches fighter1 and fighter2 in the rows labelled 'True'
    
    It can either take a data frame as a whole, or take data in the form of X y, i.e. predictors split into a data frame X and y is the observed outcomes
    
    """
    pdb.set_trace()

    # Make the filter an array from a list to avoid index issues
    filter_boolean = np.array(list(filter_boolean))

    # Define the data to be switched
    if X is None:
        df = df_original.copy()
        df = df[filter_boolean].reset_index(drop = True).copy()
    else:
        df = X.copy()
        df = df[filter_boolean].reset_index(drop = True).copy()
        y_new = y.copy()
        y_new = y_new[filter_boolean].reset_index(drop = True).copy()

    final_cols = df.columns.tolist()
        
    # Undo any cleaning that's been done on the data
    df = undo_clean_inf(df)

    # Swap column names
    df.columns = df.columns.str.replace(r'fighter1|fighter2', lambda x: swap_dict.get(x.group(0), x.group(0)), regex=True)

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

    # Reverse the matchup. Have only done this when the matchup column is as a whole, dummies wont be properly adjusted! But they never seem to be high in importance so assumed not included in final models
    if 'matchup' in df.columns:
        fighter_types = df['matchup'].str.split('-vs-', expand = True)
        df['matchup'] = fighter_types[1] + "-vs-" + fighter_types[0]

    # Change the outcomes
    if 'outcome' in df.columns:
        df['outcome'] = np.where(df['outcome'] == 'W/L', 'L/W', np.where(df['outcome'] == 'L/W', 'W/L', df['outcome']))
        
    # Now combine this data with the rows not switched
    if X is None:
        final_df = pd.concat([df, df_original.copy().reset_index(drop = True)[~filter_boolean]])
    else:
        # # Create the filtered original data for those records not adjusted
        # X_filtered = X.copy()[~filter_boolean]
        
        # # Now re-align the columns of the new data
        # df = df.reindex(columns=X_filtered.columns)
        
        # final_df = pd.DataFrame(index=range(len(df) + len(X_filtered)), columns=df.columns)
        # final_df.loc[: len(df) - 1] = df.values
        # final_df.loc[len(df) :] = X_filtered.values
        
        # # Set column types to match the original df
        # final_df = final_df.astype(df.dtypes.to_dict())

        final_df = pd.concat([df.reset_index(drop = True).copy(), X.reset_index(drop = True).copy()[~filter_boolean]], ignore_index=True)
    
    # Handle cases where we have X, y input instead of a single data frame
    if y is not None: # Check this works with W/L and isn't 0, 1, 2
        y_new = pd.Series(np.where(y_new == '2', '0', np.where(y_new == '0', '2', y_new)))
        
        # Now combine this data with the rows not switched
        y_new = pd.concat([y_new, y[~filter_boolean]])
        
        # Convert the test output to an integer to be consistent with old y
        y_new = y_new.astype('int32')
    
    # Recalculate diff columns because Infs etc might have been cleaned before. Only keep the columns needed
    diff_df = add_fighter_differentials(final_df)
    
    # Keep the diff in dates from the original data and use the diffs from the rest of the data
    diff_df = pd.concat([
        diff_df[[col for col in diff_df.columns if 'date' not in col and 'dob' not in col]].reset_index(drop = True), 
        final_df[[col for col in final_cols if ('date' in col or 'dob' in col) or col not in diff_df.columns]].reset_index(drop = True)
        ], axis = 1)
    
    final_df = diff_df.loc[:, final_cols].copy()
    
    # Clean columns of infs if necessary
    if for_model:
        final_df = clean_inf_and_large(final_df)
        final_df = format_cols_for_model(final_df)
        
    # Make sure the data types of the columns are as before
    if X is None:
        final_df = final_df.astype(df_original.dtypes.to_dict())
    else:
        final_df = final_df.astype(X.dtypes.to_dict())
    
    # Check we have the correct number of rows and columns
    if X is None:
        stop_if_not(final_df.shape == df_original.shape, 'Dataset length check')
    else:
        stop_if_not(final_df.shape == X.shape, 'Dataset length check')

    # # Ensure the old index is preserved
    # if X is None:
    #     final_df.index = df_original.index
    # else:
    #     final_df.index = X.index
    #     y_new.index = y.index
    

    if y is None:
        return final_df.copy()
    else:
        return final_df.copy(), y_new.copy()
