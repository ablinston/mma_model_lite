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
    random.seed(config['seed_all'] + 45)
    
    ## Generate a random number for each fighter in a way that keeps them consistent when new ones are added
    # First generate one for every fight, then drop duplicates to take the first random number allocated for each fighter. Need two sets so that fighter2 always starts at the same point too
    
    rand_set_1 = np.random.rand(len(data_without_oot))
    
    random.seed(config['seed_all'] + 97502)
    rand_set_2 = np.random.rand(len(data_without_oot))
    
    rands = pd.DataFrame({'fighter': data_without_oot['fighter1'].tolist() + data_without_oot['fighter2'].tolist(),
                         'rand': np.concatenate((rand_set_1, rand_set_2))}).drop_duplicates(subset = ['fighter'])
    
    # Set test fighters as those below a certain threshold, multiplied by a factor to get roughly the right proportion of fights
    test_fighters = rands[rands['rand'] < config['test_proportion'] * 0.35]['fighter'].to_list()
    
    # test_fighters = random.sample(
    #     list(set(data_without_oot['fighter1'].tolist() + data_without_oot['fighter2'].tolist())),
    #     int(config['test_proportion'] * len(data_without_oot) * 0.15) # multiply by this to get roughly the correct number of rows as test_size wishes
    # )
    
    # Set the test fighters to test
    data['train_test_oot_flag'] = np.where(
        (data['train_test_oot_flag'] == 'train') & (data['fighter1'].isin(test_fighters) | data['fighter2'].isin(test_fighters)),
        'test',
        data['train_test_oot_flag']
    )
    
    # Check the values
    print(data['train_test_oot_flag'].value_counts(normalize = True))
    
    # Check the backtesting metrics of the chosen sample
    train_df = data[(data['train_test_oot_flag'] == 'train') & ~(data['outcome'].isna()) & ~(data['fighter1_backtesting_odds'].isna()) & ~(data['fighter2_backtesting_odds'].isna())]
    
    test_df = data[(data['train_test_oot_flag'] == 'test') & ~(data['outcome'].isna()) & ~(data['fighter1_backtesting_odds'].isna()) & ~(data['fighter2_backtesting_odds'].isna())]
    
    backtesting_metrics_train = calculate_metrics_odds(train_df, clean_target_column(train_df['outcome']))
    backtesting_metrics_test = calculate_metrics_odds(test_df, clean_target_column(test_df['outcome']))
    
    print(f"Train odds logloss: {backtesting_metrics_train['X Odds Logloss']}")
    print(f"Test odds logloss: {backtesting_metrics_test['X Odds Logloss']}")
    
    del [train_df, test_df]
    
    # Return the data
    return data
