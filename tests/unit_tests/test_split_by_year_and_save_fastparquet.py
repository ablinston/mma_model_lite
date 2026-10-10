# Load all the packages and functions needed
import os
import pandas as pd

from mma_functions.utils import concat_parquets_fastparquet, split_by_year_and_save_fastparquet

import shutil

# Define some sample data
sample_data = pd.DataFrame({
    "date": pd.date_range(start="2010-01-01", end="2012-12-31", freq="ME"),
    "value": range(36)
})

def test_concat_parquets_fastparquet():
    
    # Make the temp dir
    temp_dir = "./tests/temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    """Test that the concat_parquets_fastparquet function correctly combines parquet files."""
    # Prepare test data by splitting and saving it
    split_by_year_and_save_fastparquet(sample_data.copy(), date_column="date", output_folder=temp_dir, compression="ZSTD")
    
    # Call the function
    combined_df = concat_parquets_fastparquet(temp_dir)
    
    # Ensure the combined DataFrame matches the original data
    pd.testing.assert_frame_equal(
        combined_df.sort_values(by="date").reset_index(drop=True),
        sample_data.sort_values(by="date").reset_index(drop=True),
        check_dtype=False
    )
    
    shutil.rmtree(temp_dir)
