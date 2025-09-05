import pandas as pd
import polars as pl

def merge_df_both_fighters(left_df: pd.DataFrame, df_list: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge fighter-related dataframes onto a main dataframe using Polars for efficiency.

    This function:
    1. Converts the input Pandas DataFrames to Polars DataFrames.
    2. Concatenates all fighter1 merges into one big table.
    3. Concatenates all fighter2 merges into one big table.
    4. Performs only two joins in Polars (one for fighter1, one for fighter2).
    5. Converts the result back into a Pandas DataFrame.

    Parameters
    ----------
    left_df : pd.DataFrame
        The base dataframe containing columns:
        ['event', 'bout', 'fighter1', 'fighter2'].
        This is the "main" dataframe we will merge onto.

    df_list : list[pd.DataFrame]
        A list of dataframes to merge. Each dataframe must contain:
        ['event', 'bout', 'fighter'] plus any additional columns.

    Returns
    -------
    pd.DataFrame
        A new Pandas DataFrame with the merged fighter1 and fighter2 data.

    Notes
    -----
    - Uses Polars under the hood for faster, more memory-efficient joins.
    - Reduces memory pressure compared to Pandas by doing only 2 joins total,
      instead of 2 per dataframe.
    - Ideal for low-memory environments like Raspberry Pi.
    """

    # Convert left_df to Polars
    left_pl = pl.from_pandas(left_df)

    for df in df_list:
        if len(df) == 0:
            continue

        df_pl = pl.from_pandas(df)

        # Fighter 1 prefixed version
        f1 = df_pl.rename({c: f"fighter1_{c}" for c in df_pl.columns}).rename({
            "fighter1_event": "event", "fighter1_bout": "bout", "fighter1_fighter":"fighter1"})

        # Fighter 2 prefixed version
        f2 = df_pl.rename({c: f"fighter2_{c}" for c in df_pl.columns}).rename({
            "fighter2_event": "event", "fighter2_bout": "bout", "fighter2_fighter":"fighter2"})
        
        # Merge fighter1 data
        left_pl = left_pl.join(
                f1,
                on=["event", "bout", "fighter1"],
                how="left",
            )

        # Merge fighter2 data
        left_pl = left_pl.join(
                f2,
                on=["event", "bout", "fighter2"],
                how="left",
            )
        
        del [f1, f2]
        
    # Convert back to Pandas
    return left_pl.to_pandas()
