# Load all the packages and functions needed
import pandas as pd

from mma_functions.dataset import assign_opponents

def test_assign_opponents():
    data = {"fighter": ["A", "B", "C", "D", "E", "F"]}
    df = pd.DataFrame(data)
    expected = {"fighter": ["A", "B", "C", "D", "E", "F"],
                "opponent": ["B", "A", "D", "C", "F", "E"]}
    expected_df = pd.DataFrame(expected)
    result_df = assign_opponents(df)
    pd.testing.assert_frame_equal(result_df, expected_df)
