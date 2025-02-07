# This function takes a data frame and calculates the differentials

def add_fighter_differentials(df):

    # pdb.set_trace()
    
    merged_data = df.copy()

    merged_data.reset_index(drop = True, inplace = True)

    # First, remove any existing diff columns
    existing_diff_cols = [col for col in df.columns if 'diff' in col or 'offense_minus' in col]
    
    if len(existing_diff_cols) > 0:
        merged_data = merged_data[[col for col in df.columns if col not in existing_diff_cols]]

    # Check there are no duplicates first
    duplicate_columns = merged_data.columns[merged_data.columns.duplicated()]
    stop_if_not(len(duplicate_columns) == 0)

    
    # Add direct differentials between fighter offence and defence
    fighter1_accuracy_off_cols = [col for col in merged_data.columns if ('accuracy_percent' in col or 'ctrl_minutes_per_minute' in col) and 'diff' not in col and 'fighter1' in col and 'against' not in col]
    fighter2_accuracy_off_cols = [col for col in merged_data.columns if ('accuracy_percent' in col or 'ctrl_minutes_per_minute' in col) and 'diff' not in col and 'fighter2' in col and 'against' not in col]
    fighter1_accuracy_def_cols = [col for col in merged_data.columns if ('accuracy_percent' in col or 'ctrl_minutes_against_per_minute' in col) and 'diff' not in col and 'fighter1' in col and 'against' in col]
    fighter2_accuracy_def_cols = [col for col in merged_data.columns if ('accuracy_percent' in col or 'ctrl_minutes_against_per_minute' in col) and 'diff' not in col and 'fighter2' in col and 'against' in col]
    
    # Check the columns line up
    stop_if_not(check_identical_excl_string(fighter1_accuracy_off_cols, [col.replace('_against', '') for col in fighter2_accuracy_def_cols], 'fighter1', 'fighter2'))
    stop_if_not(check_identical_excl_string(fighter2_accuracy_off_cols, [col.replace('_against', '') for col in fighter1_accuracy_def_cols], 'fighter2', 'fighter1'))
    
    # Calculate the differentials. Fill NAs with 0 for the purposes of this calc
    fighter1_off_fighter2_def = pd.DataFrame(merged_data[fighter1_accuracy_off_cols].copy().fillna(0).to_numpy() - (1 - merged_data[fighter2_accuracy_def_cols].copy().fillna(0).to_numpy()),
                                             columns = [col.replace('fighter1', 'fighter1_offense_minus_fighter2_defense') for col in fighter1_accuracy_off_cols])
    fighter1_def_fighter2_off = pd.DataFrame(merged_data[fighter2_accuracy_off_cols].copy().fillna(0).to_numpy() - (1 -  merged_data[fighter1_accuracy_def_cols].copy().fillna(0).to_numpy()),
                                             columns = [col.replace('fighter2', 'fighter2_offense_minus_fighter1_defense') for col in fighter2_accuracy_off_cols])
    
    # Combined with the data
    merged_data = pd.concat([merged_data, fighter1_off_fighter2_def, fighter1_def_fighter2_off], axis = 1)

    #######################
    # Add in some combination of differentials for an overall fight
    # Get the fighter columns
    fighter1_cols = [col for col in merged_data.columns if col.startswith('fighter1_')]
    fighter2_cols = [col for col in merged_data.columns if col.startswith('fighter2_')]

    # Remove string columns
    # Identify string columns
    string_columns = merged_data.select_dtypes(include=['object']).columns
    fighter1_cols = [col for col in fighter1_cols if col not in string_columns]
    fighter2_cols = [col for col in fighter2_cols if col not in string_columns]

    fighter1_cols.sort()
    fighter2_cols.sort()

    # Separate out the dates
    date_columns = merged_data.select_dtypes(include=['datetime64']).columns

    fighter1_cols_no_dates = [col for col in fighter1_cols if col not in date_columns]
    fighter2_cols_no_dates = [col for col in fighter2_cols if col not in date_columns]
    fighter1_cols_dates = [col for col in fighter1_cols if col in date_columns]
    fighter2_cols_dates = [col for col in fighter2_cols if col in date_columns]

    # Check that the columns correspond to one another
    if check_identical_excl_string(fighter1_cols, fighter2_cols, 'fighter1_', 'fighter2_', check_both = True):
        print("Column correspondence check PASSED")
    else:
        print("Column correspondence check FAILED")
        raise SystemExit

    stop_if_not(check_identical_excl_string(fighter1_cols_no_dates, fighter2_cols_no_dates, 'fighter1_', 'fighter2_', check_both = True))

    # Create differential columns
    diff_columns = pd.DataFrame(merged_data[fighter1_cols_no_dates].to_numpy() - merged_data[fighter2_cols_no_dates].to_numpy(),
                                columns = [col.replace('fighter1_', 'diff_').replace('fighter2_', '') for col in fighter1_cols_no_dates])

    # Create differential columns but as ratios rather than differences
    diff_ratio_columns = pd.DataFrame(safe_divide_np(merged_data[fighter1_cols_no_dates].to_numpy(), merged_data[fighter2_cols_no_dates].to_numpy()),
                                columns = [col.replace('fighter1_', 'diff_ratio_').replace('fighter2_', '') for col in fighter1_cols_no_dates])

    # Perform element-wise division and replace NaN (from division by zero) with np.inf    
    diff_date_columns = pd.DataFrame((merged_data[fighter1_cols_dates].to_numpy(dtype='datetime64') - merged_data[fighter2_cols_dates].to_numpy(dtype='datetime64')) / np.timedelta64(1, 'D'),
                                columns = [col.replace('fighter1_', 'diff_').replace('fighter2_', '') for col in fighter1_cols_dates])

    # Remove differential columns that might not be correct or not needed
    diff_columns = diff_columns[[col for col in diff_columns.columns if col not in ['diff_weight', 'diff_open_odds', 'diff_best_closing_odds']]]
    try:
        diff_ratio_columns.drop(columns = ['diff_ratio_weight'], inplace = True)
    except:
        pass

    # Combine the columns onto the master dataset
    merged_data = pd.concat([merged_data, diff_columns, diff_ratio_columns, diff_date_columns], axis = 1)
    del([diff_columns, diff_date_columns, diff_ratio_columns])


    # Check rows is consistent
    if (len(merged_data) == len(df)):
        print("Row number check PASSED")
    else:
        print("Row number check FAILED")
        raise SystemExit
    
    return merged_data
