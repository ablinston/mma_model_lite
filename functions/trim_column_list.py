# This function takes a list of columns, from the random forest, and tries to remove duplicates or very similar columns.
# The list should be ordered in terms of priority to keep the columns, with name 0 being the highest priority.

def trim_column_list(column_list):
    
    # pdb.set_trace()
    
    # First create a new version and replace any bayes and adjust names
    trunc_column_series = pd.Series(column_list).str.replace(r'_ratio|_bayes', '', regex = True).replace(r'_adjsum|_adjmean', '_adj', regex = True).replace(r'last_3_fights|last_2_fights|last_1_fight', 'last_x_fight', regex = True)
    
    # Now return the original list but without any duplicates across those strings
    
    return  pd.Series(column_list)[(~trunc_column_series.duplicated()).tolist()].tolist()
