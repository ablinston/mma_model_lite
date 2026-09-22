# Load all the packages and functions needed
exec(open('global.py').read())

# Get an example series
test_series = pd.Series([1, -1, 0, 1, -1, -1, -1, 1, -1, 1, -1, -1])

# This test looks at a specific URL for an historic event and checks that the results match what is expected
def test_minimum_sum_series():
    assert minimum_sum_series(test_series) == -4
    
