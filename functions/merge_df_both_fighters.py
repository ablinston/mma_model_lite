import pandas as pd
import polars as pl

def merge_df_both_fighters_polars(left_df: pd.DataFrame, df_list: list[pd.DataFrame]) -> pd.DataFrame:
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

    fighter1_dfs = []
    fighter2_dfs = []

    for df in df_list:
        if len(df) == 0:
            continue

        df_pl = pl.from_pandas(df)

        # Fighter 1 prefixed version
        f1 = df_pl.rename({c: f"fighter1_{c}" for c in df_pl.columns})
        fighter1_dfs.append(f1)

        # Fighter 2 prefixed version
        f2 = df_pl.rename({c: f"fighter2_{c}" for c in df_pl.columns})
        fighter2_dfs.append(f2)

    # Concatenate all fighter1/2 dfs (if any)
    fighter1_all = pl.concat(fighter1_dfs) if fighter1_dfs else None
    fighter2_all = pl.concat(fighter2_dfs) if fighter2_dfs else None

    # Merge fighter1 data
    if fighter1_all is not None:
        left_pl = left_pl.join(
            fighter1_all,
            left_on=["event", "bout", "fighter1"],
            right_on=["fighter1_event", "fighter1_bout", "fighter1_fighter"],
            how="left",
        ).drop(["fighter1_event", "fighter1_bout", "fighter1_fighter"])

    # Merge fighter2 data
    if fighter2_all is not None:
        left_pl = left_pl.join(
            fighter2_all,
            left_on=["event", "bout", "fighter2"],
            right_on=["fighter2_event", "fighter2_bout", "fighter2_fighter"],
            how="left",
        ).drop(["fighter2_event", "fighter2_bout", "fighter2_fighter"])

    # Convert back to Pandas
    return left_pl.to_pandas()
