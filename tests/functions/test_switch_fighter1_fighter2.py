# Load all the packages and functions needed
exec(open('global.py').read())

def test_switch_fighter1_fighter2():
    # Create X with a non-sequential index
    X = pd.DataFrame({
        'fight': [1, 2, 3, 4, 5],
        'fighter1_feature': [1, 2, 3, 4, 5],
        'fighter2_feature': [10, 20, 30, 40, 50],
        'fighter1_offense': [11.0, 22.0, 33.0, 44.0, 55.0],
        'fighter1_defense': [0.1, 0.2, 0.3, 0.4, 0.5],
        'fighter2_offense': [100.0, 200.0, 300.0, 400.0, 500.0],
        'fighter2_defense': [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        'outcome': ['W/L', 'L/W', 'D/D', 'L/W', 'W/L']
    }, index=[10, 20, 30, 40, 50])
    
    X['fighter1_offense_minus_fighter2_defense'] = X['fighter1_offense'] - X['fighter2_defense']
    X['fighter2_offense_minus_fighter1_defense'] = X['fighter2_offense'] - X['fighter1_defense']
    
    # Create y with a different index order
    y = pd.Series(['2', '0', '1', '0', '2'], index=[50, 30, 10, 40, 20])
    
    # Create a filter boolean mask
    filter_boolean = pd.Series([True, False, True, False, False])
    
    # Call the function
    X_new, y_new = switch_fighter1_fighter2(X=X, y=y, filter_boolean=filter_boolean)
    
    # Set the correct X_new and y_new
    X_new_correct = pd.DataFrame({
        'fight': [1, 3, 2, 4, 5],
        'fighter1_feature': [10, 30, 2, 4, 5],
        'fighter2_feature': [1, 3, 20, 40, 50],
        'fighter1_offense': [100.0, 300.0, 22.0, 44.0, 55.0],
        'fighter1_defense': [1000.0, 3000.0, 0.2, 0.4, 0.5],
        'fighter2_offense': [11.0, 33.0, 200.0, 400.0, 500.0],
        'fighter2_defense': [0.1, 0.3, 2000.0, 4000.0, 5000.0],
        'outcome': ['L/W', 'D/D', 'L/W', 'L/W', 'W/L']
    })
    
    X_new_correct['fighter1_offense_minus_fighter2_defense'] = X_new_correct['fighter1_offense'] - X_new_correct['fighter2_defense']
    X_new_correct['fighter2_offense_minus_fighter1_defense'] = X_new_correct['fighter2_offense'] - X_new_correct['fighter1_defense']
    
    y_new_correct = pd.Series(['0', '1', '0', '0', '2'])
    
    # Ensure X and y have the same index
    assert X_new.index.equals(y_new.index), "Indices of X and y should match"
    
    # Ensure no unintended modifications occurred
    assert X_new_correct[X_new.columns].equals(X_new)
    assert all(y_new_correct == y_new)
