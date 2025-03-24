# This function adds formatted names to a data frame based on a cleaned fighter column

def add_formatted_fighter_names(df, on = ['fighter'], formatted_col_names = ['Fighter'], config = config):
    
    # pdb.set_trace()
    
    # Get formatted names data
    formatted_names = fp.ParquetFile('./data/' + config['fighter_names_file_name']).to_pandas()
    
    new_df = df.copy()
    
    for i in range(0,len(on)):
        # Merge on the names
        new_df = new_df.merge(formatted_names, left_on = on[i], right_on = 'fighter', how = 'left')
        
        # Rename the fighter column
        new_df[formatted_col_names[i]] = new_df['Fighter'].copy()
        new_df.drop(columns = ['Fighter'], inplace = True)
        # new_df.rename(columns = {'fighter': formatted_col_name[i]}, inplace = True)
    
    # Check rows correct
    stop_if_not(len(new_df) == len(df))
    
    return new_df