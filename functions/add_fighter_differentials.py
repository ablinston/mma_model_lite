# This function takes a data frame and calculates the differentials

def add_fighter_differentials(df, zero_over_zero_fill = 1):

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
    
    first_exprs = []

    # Fighter1 offense minus Fighter2 defense
    first_exprs.extend([
        (pl.col(f1).fill_null(0) - (1 - pl.col(f2).fill_null(0)))
        .alias(f1.replace("fighter1", "fighter1_offense_minus_fighter2_defense"))
        for f1, f2 in zip(fighter1_accuracy_off_cols, fighter2_accuracy_def_cols)
    ])

    # Fighter2 offense minus Fighter1 defense
    first_exprs.extend([
        (pl.col(f2).fill_null(0) - (1 - pl.col(f1).fill_null(0)))
        .alias(f2.replace("fighter2", "fighter2_offense_minus_fighter1_defense"))
        for f2, f1 in zip(fighter2_accuracy_off_cols, fighter1_accuracy_def_cols)
    ])

    df_pl = df_pl.with_columns(first_exprs)


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

    # Check that the columns correspond to one another
    if check_identical_excl_string(fighter1_cols, fighter2_cols, 'fighter1_', 'fighter2_', check_both = True):
        print("Column correspondence check PASSED")
    else:
        print("Column correspondence check FAILED")
        raise SystemExit
 
    # Split off datetime columns
    date_cols = [c for c, dt in df_pl.schema.items() if dt == pl.Datetime]
    f1_no_dates = [c for c in fighter1_cols if c not in date_cols]
    f2_no_dates = [c for c in fighter2_cols if c not in date_cols]
    f1_dates = [c for c in fighter1_cols if c in date_cols]
    f2_dates = [c for c in fighter2_cols if c in date_cols]
    
    stop_if_not(check_identical_excl_string(f1_no_dates, f2_no_dates, 'fighter1_', 'fighter2_', check_both = True))

    # --- Differences ---
    # Build a set of all boolean columns
    bool_cols = {c for c, dt in df_pl.schema.items() if dt == pl.Boolean}
    
    # f1_exprs and f2_exprs for arithmetic
    f1_exprs = [pl.col(c).cast(pl.Float64) if c in bool_cols else pl.col(c) for c in f1_no_dates]
    f2_exprs = [pl.col(c).cast(pl.Float64) if c in bool_cols else pl.col(c) for c in f2_no_dates]

    df_pl = df_pl.with_columns([
        (f1 - f2).alias(c.replace("fighter1_", "diff_").replace("fighter2_", ""))
        for f1, f2, c in zip(f1_exprs, f2_exprs, f1_no_dates)
    ])

    # --- Ratios ---
    df_pl = df_pl.with_columns([
        (
            pl.when((f1 == 0) & (f2 == 0))           # both numerator & denominator zero
              .then(zero_over_zero_fill)
            .when(f2 == 0)                           # denominator zero
              .then(np.inf)
            .otherwise(f1 / f2)                      # normal division
        ).alias(c.replace("fighter1_", "diff_ratio_").replace("fighter2_", ""))
        for f1, f2, c in zip(f1_exprs, f2_exprs, f1_no_dates)
    ])

    # --- Date differences in days ---
    df_pl = df_pl.with_columns([
        ((pl.col(f1).cast(pl.Int64) - pl.col(f2).cast(pl.Int64)) / 86_400_000_000_000)
        .alias(f1.replace("fighter1_", "diff_").replace("fighter2_", ""))
        for f1, f2 in zip(f1_dates, f2_dates)
    ])

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
