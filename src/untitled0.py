import pandas as pd
import numpy as np

def add_fighter_differentials(df):
    merged_data = df.drop(columns=[col for col in df.columns if 'diff' in col or 'offense_minus' in col], errors='ignore')

    if merged_data.columns.duplicated().any():
        raise ValueError("Duplicate columns found in the input data.")

    # Identify offensive and defensive columns
    f1_off_cols = [col for col in merged_data.columns if 'fighter1' in col and ('accuracy_percent' in col or 'ctrl_minutes_per_minute' in col) and 'against' not in col]
    f2_off_cols = [col for col in merged_data.columns if 'fighter2' in col and ('accuracy_percent' in col or 'ctrl_minutes_per_minute' in col) and 'against' not in col]
    f1_def_cols = [col.replace('_against', '') for col in merged_data.columns if 'fighter1' in col and 'against' in col]
    f2_def_cols = [col.replace('_against', '') for col in merged_data.columns if 'fighter2' in col and 'against' in col]

    if f1_off_cols != f2_def_cols or f2_off_cols != f1_def_cols:
        raise ValueError("Column mismatch in offensive and defensive metrics.")

    # Compute differentials using NumPy for speed
    f1_off_def_diff = np.nan_to_num(merged_data[f1_off_cols].to_numpy()) - (1 - np.nan_to_num(merged_data[f2_def_cols].to_numpy()))
    f2_off_def_diff = np.nan_to_num(merged_data[f2_off_cols].to_numpy()) - (1 - np.nan_to_num(merged_data[f1_def_cols].to_numpy()))

    # Assign new column names
    f1_off_def_cols = [col.replace('fighter1', 'fighter1_offense_minus_fighter2_defense') for col in f1_off_cols]
    f2_off_def_cols = [col.replace('fighter2', 'fighter2_offense_minus_fighter1_defense') for col in f2_off_cols]

    # Add computed differentials
    merged_data = merged_data.join(pd.DataFrame(f1_off_def_diff, columns=f1_off_def_cols, index=df.index))
    merged_data = merged_data.join(pd.DataFrame(f2_off_def_diff, columns=f2_off_def_cols, index=df.index))

    # Extract numerical fighter columns
    f1_cols = [col for col in merged_data.columns if col.startswith('fighter1_')]
    f2_cols = [col for col in merged_data.columns if col.startswith('fighter2_')]
    
    # Filter out non-numeric columns efficiently
    numeric_mask = merged_data.dtypes != 'object'
    f1_cols = [col for col in f1_cols if numeric_mask[col]]
    f2_cols = [col for col in f2_cols if numeric_mask[col]]

    if sorted(f1_cols) != sorted([col.replace('fighter2_', 'fighter1_') for col in f2_cols]):
        raise ValueError("Column mismatch between fighter1 and fighter2 attributes.")

    # Compute raw differentials and ratios
    f1_vals = merged_data[f1_cols].to_numpy()
    f2_vals = merged_data[f2_cols].to_numpy()

    diff_vals = f1_vals - f2_vals
    diff_ratio_vals = np.divide(f1_vals, f2_vals, out=np.full_like(f1_vals, np.nan), where=f2_vals != 0)

    diff_cols = [col.replace('fighter1_', 'diff_').replace('fighter2_', '') for col in f1_cols]
    diff_ratio_cols = [col.replace('fighter1_', 'diff_ratio_').replace('fighter2_', '') for col in f1_cols]

    merged_data = merged_data.join(pd.DataFrame(diff_vals, columns=diff_cols, index=df.index))
    merged_data = merged_data.join(pd.DataFrame(diff_ratio_vals, columns=diff_ratio_cols, index=df.index))

    # Remove unwanted differential columns in one step
    merged_data.drop(columns=['diff_weight', 'diff_open_odds', 'diff_best_closing_odds', 'diff_mean_closing_odds', 'diff_ratio_weight'], errors='ignore', inplace=True)

    if len(merged_data) != len(df):
        raise ValueError("Row count mismatch after processing.")

    return merged_data
