# Load all the packages and functions needed
import fastparquet as fp
import numpy as np
import pandas as pd

from mma_functions.features import calculate_knocked_out_stats


correct = fp.ParquetFile("./tests/data/test_calculate_knocked_out_stats_correct.parquet").to_pandas()

def test_calculate_knocked_out_stats():
    # Example data (10+ rows)
    data = {
        "event": ["E1","E1","E2","E2","E3","E3","E4","E4","E5","E5","E6","E6"],
        "bout": [1,1,2,2,3,3,4,4,5,5,6,6],
        "fighter": ["A","B","B","A","A","B","A","B","C","A","C","A"],
        "date": pd.to_datetime([
            "2025-01-01","2025-01-01","2025-02-01","2025-02-01","2025-03-01","2025-03-01",
            "2025-04-01","2025-04-01","2025-05-01","2025-05-01","2025-06-01","2025-06-15"
        ]),
        "result": ["W","L","W","L","L","L","W","L","L","W","L","L"],
        "method": ["DEC","DEC","KO/TKO","KO/TKO","KO/TKO","KO/TKO","DEC","DEC","KO/TKO","KO/TKO","KO/TKO","KO/TKO"],
        "round": [1,1,2,2,np.nan,3,2,2,1,1,3,3]
    }
    
    df = pd.DataFrame(data)
    
    # Run function
    result = calculate_knocked_out_stats(df)
    
    # Basic assertions
    assert len(result) == len(df)#, "Number of rows should stay the same"
    
    # Check columns exist
    expected_cols = [
        "event","bout","fighter","date_last_knocked_out","hist_times_knocked_out",
        "hist_times_knocked_out_round_1","hist_times_knocked_out_round_2",
        "hist_times_knocked_out_round_3","hist_times_knocked_out_round_4","hist_times_knocked_out_round_5"
    ]
    for col in expected_cols:
        assert col in result.columns#, f"Missing column {col}"
    
    # Check no NA values in hist columns
    hist_cols = expected_cols[4:]
    assert result[hist_cols].isna().sum().sum() == 0#, "Hist columns should have no NaNs"
    
    # Check date_last_knocked_out filled
    assert pd.notna(result["date_last_knocked_out"]).all()#, "date_last_knocked_out should have no NaNs"

    # Optional: check one fighter's hist_times_knocked_out makes sense
    fighter_a = result[result["fighter"]=="A"]
    assert (fighter_a["hist_times_knocked_out"] >= 0).all()# "hist_times_knocked_out should be non-negative"

    # Check the data is the same as expected
    pd.testing.assert_frame_equal(
        correct.sort_values("fighter").reset_index(drop=True),
        result.sort_values("fighter").reset_index(drop=True),
        check_exact=False,
        check_dtype=False)

# fp.write('./tests/data/test_calculate_knocked_out_stats_correct.parquet', result, compression = 'ZSTD')
