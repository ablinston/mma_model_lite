# This function takes a data frame and flags the data with train/test/oot for a given config

def flag_train_test_oot(df, config):
    
    # pdb.set_trace()
    data = df.copy()
    
    # Flag the oot sample and set the rest to train
    data['train_test_oot_flag'] = np.where(
        (data['date'] >= config['min_training_date']) & (data['date'] < config['oot_start_date']),
        'train',
        np.where(data['date'] >= config['oot_start_date'], 'oot', np.nan)
    )
    
    # Get a separate data frame with only non-oot fighters
    data_without_oot = data[data['train_test_oot_flag'] == 'train'].copy()
        
    # Get a list of all fighters to select the test data
    random.seed(config['seed_all'])
    test_fighters = random.sample(
        list(set(data_without_oot['fighter1'].tolist() + data_without_oot['fighter2'].tolist())),
        int(config['test_proportion'] * len(data_without_oot) * 0.15) # multiply by this to get roughly the correct number of rows as test_size wishes
    )
    
    # Set the test fighters to test
    data['train_test_oot_flag'] = np.where(
        (data['train_test_oot_flag'] == 'train') & (data['fighter1'].isin(test_fighters) | data['fighter2'].isin(test_fighters)),
        'test',
        data['train_test_oot_flag']
    )
    
    # Check the values
    print(data['train_test_oot_flag'].value_counts(normalize = True))
        
    # Return the data
    return data
