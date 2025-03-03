# This function takes a dataset and prepares the data for model training by splitting it into train and test

# It returns 7 objects that can be output into env objects like
# X_train, X_test, y_train, y_test, y_test_bin, no_of_classes, oot = prep_data_for_model_training(data, config)

def prep_data_for_model_training(df, config, split_or_other_flag = False, test_size = 0.1, upcoming_fights = False):
    
    # pdb.set_trace()o  
    
    data = clean_inf_and_large(df.copy())
    
    # Filter the OOT data and training data to be ignored
    oot = data[data['date'] >= config['oot_start_date']]
    data = data[(data['date'] >= config['min_training_date']) & (data['date'] < config['oot_start_date'])]    
        
    # First get a list of all fighters to select the test data
    random.seed(config['seed_all'])
    test_fighters = random.sample(
        list(set(data['fighter1'].tolist() + data['fighter2'].tolist())),
        int(test_size * len(data) * 0.1) # multiply by this to get roughly the correct number of rows as test_size wishes
    )
        
    # Create a flag for exclusion of split decision and other in training data if required
    if (split_or_other_flag):
        data['split_dec_or_other'] = np.where(data['method'].isin(['Decision-Split', 'Other']), 1, 0)

    # Remove column that has results, or that contain information we don't want included.
    data.drop(columns = ['method', 'index', 'level_0', 'date'], inplace = True, errors = 'ignore')
    data.rename(columns = {'fighter1': 'notformodel_fighter1', 'fighter2': 'notformodel_fighter2'}, inplace = True)
    
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
            
    if test_size > 0:
        # Split into train and test
        train = data[(~data['notformodel_fighter1'].isin(test_fighters)) & (~data['notformodel_fighter2'].isin(test_fighters))]
        test = data[(data['notformodel_fighter1'].isin(test_fighters)) | (data['notformodel_fighter2'].isin(test_fighters))]
        
        X_train = train.drop(columns = target)
        y_train = train[target]
        
        X_test = test.drop(columns = target)
        y_test = test[target]    
        
        del([train, test])
        
    else:
        X_train = data.drop(columns = [target])
        y_train = data[target]
        X_test = None
        y_test = None
    
    if not upcoming_fights and test_size > 1e-98:
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
   
    # ####################
    # ##AB: FOR SOME REASON, THE MODELS DON'T TRAIN OR PREDICT CORRECTLY WHEN THESE HAVEN'T GONE THROUGH THIS FUNCTION. IT PRODUCES THE SAME RESULT BUT FOR SOME REASON XGB DOESN'T GIVE MODELLING OUTPUTS CORRECTLY, SO LEAVING THIS IN AS A WORKAROUND
    
    # X_train, y_train = switch_fighter1_fighter2(X = X_train, y = y_train, filter_boolean = np.random.choice([True], size = len(X_train)))
    # X_train, y_train = switch_fighter1_fighter2(X = X_train, y = y_train, filter_boolean = np.random.choice([True], size = len(X_train)))
    
    # gc.collect()
    
    # if X_test is not None:
    #     X_test, y_test = switch_fighter1_fighter2(X = X_test, y = y_test, filter_boolean = np.random.choice([True], size = len(X_test)))
    #     X_test, y_test = switch_fighter1_fighter2(X = X_test, y = y_test, filter_boolean = np.random.choice([True], size = len(X_test)))
  
    # ####################
    
    
    # Return the objects needed as a list
    return [X_train, X_test, y_train, y_test, y_test_bin, no_of_classes, oot]
