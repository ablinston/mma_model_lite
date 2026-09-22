# Load all the packages and functions needed
exec(open('global.py').read())

X_val = fp.ParquetFile('./tests/data/test_switch_fighter1_fighter2_input_X.parquet').to_pandas()
y_val = fp.ParquetFile('./tests/data/test_switch_fighter1_fighter2_input_y.parquet').to_pandas().iloc[:, 0]


def test_switch_fighter1_fighter2():
    # Create X with a non-sequential index
    X = pd.DataFrame({
        'fight': [1, 2, 3, 4, 5],
        'fighter1_feature': [1, 2, 3, 4, 5],
        'fighter2_feature': [10, 20, 30, 40, 50],
        'fighter1_stat_accuracy_percent': [11.0, 22.0, 33.0, 44.0, 55.0],
        'fighter1_stat_accuracy_percent_against': [0.1, 0.2, 0.3, 0.4, 0.5],
        'fighter2_stat_accuracy_percent': [100.0, 200.0, 300.0, 400.0, 500.0],
        'fighter2_stat_accuracy_percent_against': [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        'outcome': ['W/L', 'L/W', 'D/D', 'L/W', 'W/L'],
        'train_fighter1_odds': [2.1, 2.2, 2.3, 2.4, 2.5],
        'train_fighter2_odds': [3.1, 3.2, 3.3, 3.4, 3.5]
    }, index=[20, 10, 30, 40, 50])
    
    X = add_fighter_differentials(X)
    
    
    # Create y with a different index order
    y = pd.Series(['2', '0', '1', '0', '2'], index=[20, 10, 30, 40, 50])
    
    # Create a filter boolean mask
    filter_boolean = pd.Series([True, False, True, False, False])
    
    # Call the function
    X_new, y_new = switch_fighter1_fighter2(X=X, y=y, filter_boolean=filter_boolean)
    
    # Set the correct X_new and y_new
    X_new_correct = pd.DataFrame({
        'fight': [1, 3, 2, 4, 5],
        'fighter1_feature': [10, 30, 2, 4, 5],
        'fighter2_feature': [1, 3, 20, 40, 50],
        'fighter1_stat_accuracy_percent': [100.0, 300.0, 22.0, 44.0, 55.0],
        'fighter1_stat_accuracy_percent_against': [1000.0, 3000.0, 0.2, 0.4, 0.5],
        'fighter2_stat_accuracy_percent': [11.0, 33.0, 200.0, 400.0, 500.0],
        'fighter2_stat_accuracy_percent_against': [0.1, 0.3, 2000.0, 4000.0, 5000.0],
        'outcome': ['L/W', 'D/D', 'L/W', 'L/W', 'W/L'],
        'train_fighter1_odds': [3.1, 3.3, 2.2, 2.4, 2.5],
        'train_fighter2_odds': [2.1, 2.3, 3.2, 3.4, 3.5]
    }, index=[20, 30, 10, 40, 50])
    
    X_new_correct = add_fighter_differentials(X_new_correct)
    
    y_new_correct = pd.Series([0, 1, 0, 0, 2], index=[20, 30, 10, 40, 50])
    
    # Ensure X and y have the same index
    # assert X_new.index.equals(y_new.index), "Indices of X and y should match"
    
    # Ensure no unintended modifications occurred
    assert X_new_correct[X_new.columns].reset_index(drop = True).equals(X_new.reset_index(drop = True)) # X_new_correct[X_new.columns].compare(X_new)
    assert all(y_new_correct == y_new)
    
    # Now perform two switches and check if the outputs are the same
    X_val1, y_val1 = switch_fighter1_fighter2(X = X_val, y = y_val, filter_boolean = np.random.choice([True], size = len(X_val)))
    X_val1, y_val1 = switch_fighter1_fighter2(X = X_val1, y = y_val1, filter_boolean = np.random.choice([True], size = len(X_val1)))

    assert X_val.round(8).equals(X_val1.round(8))
    
