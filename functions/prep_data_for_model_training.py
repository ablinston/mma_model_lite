# This function takes a dataset and prepares the data for model training by splitting it into train and test

# It returns 7 objects that can be output into env objects like
# X_train, X_test, y_train, y_test, y_test_bin, no_of_classes, oot = prep_data_for_model_training(data, config)

def prep_data_for_model_training(df, config, split_or_other_flag = False, test_size = 0.1):
    # pdb.set_trace()
    data = df.copy()
    
    # Replace infinite values with large values
    data.replace([np.inf], 99999, inplace = True)
    data.replace([-np.inf], -99999, inplace = True)
    
    # Filter the OOT data and training data to be ignored
    oot = data[data['date'] >= config['oot_start_date']]
    data = data[(data['date'] >= config['min_training_date']) & (data['date'] < config['oot_start_date'])]
    
    # Create a flag for exclusion of split decision and other in training data if required
    if (split_or_other_flag):
        data['split_dec_or_other'] = np.where(data['method'].isin(['Decision-Split', 'Other']), 1, 0)

    # Remove column that has results, or that contain information we don't want included.
    data.drop(columns = ['method', 'fighter1', 'fighter2', 'index', 'level_0', 'date'], inplace = True, errors = 'ignore')
    
    # Set target
    target = 'outcome'
    
    # Recode the target. Don't need to distinguish between D and NC
    data[target] = np.where((data[target] == "D/D") | (data[target] == "NC/NC"), "OTH", data[target])
    
    # Convert all dates to integers
    date_cols = [col for col in data.columns if data.dtypes[col] == 'datetime64[ns]']
    data[date_cols] = data[date_cols].astype('int64')
    
    # Convert object columns into series' of dummies
    obj_cols = [col for col in data.columns if data.dtypes[col] == 'object']
    obj_cols.remove(target)
    data = pd.get_dummies(data, columns = obj_cols)
    
    # Convert the target into a labelled column ready for multiclass modelling
    data[target] = clean_target_column(data[target])
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns = [target]),
        data[target],
        test_size = test_size,
        random_state = config['seed_all']
    )
    
    # Convert the test output to a binarised data frame. Can help with AUC calculations
    y_test_bin = label_binarize(y_test, classes = np.unique(y_test))
    
    # # Exclude split decisions and other if required
    # if exclude_split_other_from_train:
    #     X_train = X_train[X_train['split_dec_or_other'] == 0]
    #     X_train.drop(columns = ['split_dec_or_other'], inplace = True)
    
    # Check data types
    data.dtypes.value_counts()
    
    # Set number of classes for training
    no_of_classes = len(set(y_train))
   
  
    # Return the objects needed as a list
    return [X_train, X_test, y_train, y_test, y_test_bin, no_of_classes, oot]
