# This function takes a data frame and calculates the differentials

def add_fighter_differentials(df):

    # pdb.set_trace()
    
    # Convert pandas DataFrame to Polars
    df_pl = pl.from_pandas(df)

    # Remove existing diff columns
    diff_cols = [c for c in df_pl.columns if 'diff' in c or 'offense_minus' in c]
    if diff_cols:
        df_pl = df_pl.select([c for c in df_pl.columns if c not in diff_cols])
    
    # Get sorted columns
    sorted_columns = sorted(df_pl.columns)    
       
    # Add direct differentials between fighter offence and defence
    fighter1_accuracy_off_cols = [col for col in sorted_columns if ('accuracy_percent' in col or 'ctrl_minutes_per_minute' in col) and 'diff' not in col and 'fighter1' in col and 'against' not in col]
    fighter2_accuracy_off_cols = [col for col in sorted_columns if ('accuracy_percent' in col or 'ctrl_minutes_per_minute' in col) and 'diff' not in col and 'fighter2' in col and 'against' not in col]
    fighter1_accuracy_def_cols = [col for col in sorted_columns if ('accuracy_percent' in col or 'ctrl_minutes_against_per_minute' in col) and 'diff' not in col and 'fighter1' in col and 'against' in col]
    fighter2_accuracy_def_cols = [col for col in sorted_columns if ('accuracy_percent' in col or 'ctrl_minutes_against_per_minute' in col) and 'diff' not in col and 'fighter2' in col and 'against' in col]
    
    # Check the columns line up
    stop_if_not(check_identical_excl_string(fighter1_accuracy_off_cols, [col.replace('_against', '') for col in fighter2_accuracy_def_cols], 'fighter1', 'fighter2'))
    stop_if_not(check_identical_excl_string(fighter2_accuracy_off_cols, [col.replace('_against', '') for col in fighter1_accuracy_def_cols], 'fighter2', 'fighter1'))
    
    # Calculate offense-minus-defense differentials
    for f1_off, f2_def in zip(fighter1_accuracy_off_cols, fighter2_accuracy_def_cols):
        new_col = f1_off.replace('fighter1', 'fighter1_offense_minus_fighter2_defense')
        df_pl = df_pl.with_columns(
            ((pl.col(f1_off).fill_null(0)) - (1 - pl.col(f2_def).fill_null(0))).alias(new_col)
        )

    for f2_off, f1_def in zip(fighter2_accuracy_off_cols, fighter1_accuracy_def_cols):
        new_col = f2_off.replace('fighter2', 'fighter2_offense_minus_fighter1_defense')
        df_pl = df_pl.with_columns(
            ((pl.col(f2_off).fill_null(0)) - (1 - pl.col(f1_def).fill_null(0))).alias(new_col)
        )

    #######################
    # Add in some combination of differentials for an overall fight
    # Get the fighter columns
    fighter1_cols = [c for c in df_pl.columns if c.startswith('fighter1_') and 'notformodel' not in c and 'backtesting' not in c]
    fighter2_cols = [c for c in df_pl.columns if c.startswith('fighter2_') and 'notformodel' not in c and 'backtesting' not in c]

    # Remove string columns
    # Identify string columns
    string_cols = [c for c, dt in df_pl.schema.items() if dt in (pl.Utf8, pl.Categorical)]
    fighter1_cols = [c for c in fighter1_cols if c not in string_cols]
    fighter2_cols = [c for c in fighter2_cols if c not in string_cols]

    fighter1_cols.sort()
    fighter2_cols.sort()

    # Separate out the dates
    date_columns = [c for c, dt in df_pl.schema.items() if dt == pl.Datetime]

    fighter1_cols_no_dates = [col for col in fighter1_cols if col not in date_columns]
    fighter2_cols_no_dates = [col for col in fighter2_cols if col not in date_columns]
    fighter1_cols_dates = [col for col in fighter1_cols if col in date_columns]
    fighter2_cols_dates = [col for col in fighter2_cols if col in date_columns]

    # Check that the columns correspond to one another
    if check_identical_excl_string(fighter1_cols, fighter2_cols, 'fighter1_', 'fighter2_', check_both = True):
        print("Column correspondence check PASSED")
    else:
        print("Column correspondence check FAILED")
        raise SystemExit

    stop_if_not(check_identical_excl_string(fighter1_cols_no_dates, fighter2_cols_no_dates, 'fighter1_', 'fighter2_', check_both = True))
    pdb.set_trace()
    # Differences
    for f1, f2 in zip(fighter1_cols_no_dates, fighter2_cols_no_dates):
        col_name = f1.replace("fighter1_", "diff_").replace("fighter2_", "")
        if df_pl.schema[f1] == pl.Boolean or df_pl.schema[f2] == pl.Boolean:
            df_pl = df_pl.with_columns(
                (pl.col(f1).cast(pl.Int64) - pl.col(f2).cast(pl.Int64)).alias(col_name)
            )
        else:
            df_pl = df_pl.with_columns((pl.col(f1) - pl.col(f2)).alias(col_name))

    # Ratios
    for f1, f2 in zip(fighter1_cols_no_dates, fighter2_cols_no_dates):
        col_name = f1.replace("fighter1_", "diff_ratio_").replace("fighter2_", "")
        if df_pl.schema[f1] == pl.Boolean or df_pl.schema[f2] == pl.Boolean:
            df_pl = df_pl.with_columns(
                (
                    pl.when(pl.col(f2) != 0)
                    .then(pl.col(f1).cast(pl.Float64) / pl.col(f2).cast(pl.Float64))
                    .otherwise(np.inf)
                ).alias(col_name)
            )
        else:
            df_pl = df_pl.with_columns(
                (
                    pl.when(pl.col(f2) != 0)
                    .then(pl.col(f1) / pl.col(f2))
                    .otherwise(np.inf)
                ).alias(col_name)
            )

    # Date differences in days
    for f1, f2 in zip(fighter1_cols_dates, fighter2_cols_dates):
        col_name = f1.replace('fighter1_', 'diff_').replace('fighter2_', '')
        df_pl = df_pl.with_columns(((pl.col(f1).cast(pl.Int64) - pl.col(f2).cast(pl.Int64)) / 86400000000000).alias(col_name))
        # 1 day = 86400*10^9 ns

    # Remove unwanted columns (same as original)
    for c in ['diff_weight', 'diff_open_odds', 'diff_best_closing_odds', 'diff_mean_closing_odds', 'diff_ratio_weight']:
        if c in df_pl.columns:
            df_pl = df_pl.drop(c)

    # Check rows is consistent
    if (len(df_pl) == len(df)):
        print("Row number check PASSED")
    else:
        print("Row number check FAILED")
        raise SystemExit
    
    return df_pl.to_pandas()
