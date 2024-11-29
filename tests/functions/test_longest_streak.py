# Load all the packages and functions needed
exec(open('global.py').read())

def test_longest_streak():
    # Example series
    s = pd.Series([0, 1, 1, -1, 0, -1, -1, -1, 0, 1, 1, 1, -1])
    
    # Test longest streak of 1
    assert longest_streak(s, 1) == 3, "Longest streak of 1 should be 3"
    
    # Test longest streak of -1
    assert longest_streak(s, -1) == 4, "Longest streak of -1 should be 4"
    
