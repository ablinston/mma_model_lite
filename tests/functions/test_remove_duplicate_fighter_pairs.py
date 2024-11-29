# Load all the packages and functions needed
exec(open('global.py').read())

# This test looks whether the function effectively removes the duplicate rows
def test_remove_duplicate_fighter_pairs():
# Test data
    data = {
        'date': ['2024-11-01', '2024-11-01', '2024-11-01', '2024-11-02', '2024-11-03'],
        'fighter1': ['Fighter A', 'Fighter C', 'Fighter A', 'Fighter C', 'Fighter A'],
        'fighter2': ['Fighter B', 'Fighter B', 'Fighter C', 'Fighter A', 'Fighter D']
    }

    # Expected result
    expected_data = {
        'date': ['2024-11-01', '2024-11-02', '2024-11-03'],
        'fighter1': ['Fighter A', 'Fighter C', 'Fighter A'],
        'fighter2': ['Fighter B', 'Fighter A', 'Fighter D']
    }
    expected_df = pd.DataFrame(expected_data)
    expected_df['date'] = pd.to_datetime(expected_df['date'])

    # Run the function
    result_df = remove_fighter_duplicates(data)

    # Assert that the result matches the expected DataFrame
    pd.testing.assert_frame_equal(result_df, expected_df)
    
