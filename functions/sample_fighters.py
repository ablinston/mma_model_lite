# This function takes a data frame and selects a random sample of rows, but it does this by randomly selecting fighters then
# selecting all their fights for the data.

# rows_with_bagging is how many rows the final data should have after bagging 

def sample_fighters(df_complete = None, X = None, y = None, n_rows = 100, rows_with_bagging = None):
    
    # pdb.set_trace()
    
    if df_complete is None:
        df = X.copy()
        df['y'] = y
    else:
        df = df_complete.copy()
    
    # Create a series of all the fighters
    fighters = pd.concat([df['notformodel_fighter1'], df['notformodel_fighter2']])

    # Count how many fights they have and shuffle
    fighters_counts = pd.DataFrame(fighters.value_counts().sample(frac = 1)).reset_index()
    
    # Now do cumulative sums until I get the required rows
    fighters_counts['cumsums'] = fighters_counts['count'].cumsum()
    
    # Create a flag for which fighters will be in the selected data. Multiply by 1.1 because there will be overlap in these fighters fighting each other
    fighters_counts['select_flag'] = np.where(fighters_counts['cumsums'] < (n_rows * 1.2), 1, 0)

    # Get the selected fighters
    selected_fighters = fighters_counts[fighters_counts['select_flag'] == 1]['index'].tolist()
    
    final_df = df[df['notformodel_fighter1'].isin(selected_fighters) | df['notformodel_fighter2'].isin(selected_fighters)]

    # Do a check to see if n rows is reasonable
    if abs(len(final_df) - n_rows) > (n_rows * 0.05):
        print(f"WARNING: sample_fighters() function returned a row number > 5% different from target. Sample is {(len(final_df) / n_rows) - 1} of data.")

    # If doing bagging, expand the sample
    if rows_with_bagging is not None:
        final_df = resample(final_df, replace = True, n_samples = rows_with_bagging)

    # Now return the selected data
    if df_complete is None:
        # Return an X and y
        return final_df.drop(columns = 'y'), final_df['y']
    else:
        return final_df
