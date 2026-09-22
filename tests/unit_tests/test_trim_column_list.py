# Load all the packages and functions needed
exec(open('global.py').read())

# Test cases for the trim_column_list function
def test_trim_column_list():
    # Test 1: Regular case with some duplicates
    column_list = ['example_adjsum', 'value_adjmean', 'ratio_bayes', 'value_adjsum', 'test', 'example_adjsum']
    expected_result = ['example_adjsum', 'value_adjmean', 'ratio_bayes', 'test']
    assert trim_column_list(column_list) == expected_result
    
    # Test 2: Case with no duplicates
    column_list = ['example_adjsum', 'value_adjmean', 'ratio_bayes', 'test']
    expected_result = ['example_adjsum', 'value_adjmean', 'ratio_bayes', 'test']
    assert trim_column_list(column_list) == expected_result

    # Test 3: Case with only duplicates (after replacement)
    column_list = ['example_adjsum', 'example_adjsum', 'test_bayes', 'test_bayes']
    expected_result = ['example_adjsum', 'test_bayes']
    assert trim_column_list(column_list) == expected_result
    
    # Test 4: Case with no special strings to replace
    column_list = ['example', 'value', 'ratio', 'test']
    expected_result = ['example', 'value', 'ratio', 'test']
    assert trim_column_list(column_list) == expected_result

    # Test 5: Case with an empty list
    column_list = []
    expected_result = []
    assert trim_column_list(column_list) == expected_result

    # Test 6: Case with one item in the list (no duplicates possible)
    column_list = ['example_adjsum']
    expected_result = ['example_adjsum']
    assert trim_column_list(column_list) == expected_result

    # Test 7: Case with only the replaceable substrings
    column_list = ['_adjsum', '_adjmean', '_ratio', '_bayes']
    expected_result = ['_adjsum', '_ratio']
    assert trim_column_list(column_list) == expected_result
    
