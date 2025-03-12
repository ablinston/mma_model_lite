# This function takes a dataset and prepares the data for model training by splitting it into train and test

# It returns 7 objects that can be output into env objects like
# X_train, X_test, y_train, y_test, y_test_bin, no_of_classes, oot = prep_data_for_model_training(data)

def prep_data_for_model_training(df, split_or_other_flag = False, upcoming_fights = False):
    
    # pdb.set_trace()
        
    data = clean_inf_and_large(df.copy())
    
    # If we're prepping upcoming fights, take whole sample
    if upcoming_fights:
        data['train_test_oot_flag'] = 'train'
    
    # Filter the OOT data and training data to be ignored
    oot = data[data['train_test_oot_flag'] == 'oot']
    data = data[data['train_test_oot_flag'] != 'oot']
        
    # Create a flag for exclusion of split decision and other in training data if required
    if (split_or_other_flag):
        data['split_dec_or_other'] = np.where(data['method'].isin(['Decision-Split', 'Other']), 1, 0)

    # Remove column that has results, or that contain information we don't want included.
    data.drop(columns = ['method', 'index', 'level_0', 'date'], inplace = True, errors = 'ignore')
    data.rename(columns = {'fighter1': 'notformodel_fighter1', 'fighter2': 'notformodel_fighter2', 'train_test_oot_flag': 'notformodel_train_test_oot_flag'}, inplace = True)
    
    # Set target
    target = 'outcome'
    
    if not upcoming_fights:
        # Recode the target. Don't need to distinguish between D and NC
        data[target] = np.where((data[target] == "D/D") | (data[target] == "NC/NC"), "OTH", data[target])
    
    data = format_cols_for_model(data)
    
    # Convert object columns into series' of dummies
    obj_cols = [col for col in data.columns if data.dtypes[col] == 'object' and 'notformodel' not in col]
    try:
        obj_cols.remove(target)
    except:
        pass
    data = pd.get_dummies(data, columns = obj_cols)
    
    if not upcoming_fights:
        # Convert the target into a labelled column ready for multiclass modelling
        data[target] = clean_target_column(data[target])
            
    X_train = data[data['notformodel_train_test_oot_flag'] == 'train'].drop(columns = [target])
    y_train = data[data['notformodel_train_test_oot_flag'] == 'train'][target]
    X_test = data[data['notformodel_train_test_oot_flag'] == 'test'].drop(columns = [target])
    y_test = data[data['notformodel_train_test_oot_flag'] == 'test'][target]
    
    if not upcoming_fights and len(X_test) > 0:
        # Convert the test output to a binarised data frame. Can help with AUC calculations
        y_test_bin = label_binarize(y_test, classes = np.unique(y_test))
    
        # # Exclude split decisions and other if required
        # if exclude_split_other_from_train:
        #     X_train = X_train[X_train['split_dec_or_other'] == 0]
        #     X_train.drop(columns = ['split_dec_or_other'], inplace = True)
        
        # Check data types
        data.dtypes.value_counts()
    else:
        y_test_bin = None
    
    # Set number of classes for training
    no_of_classes = len(set(y_train))
    
    
    # Return the objects needed as a list
    return [X_train, X_test, y_train, y_test, y_test_bin, no_of_classes, oot]
