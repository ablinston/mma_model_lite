from ablinston_ml import stop_if_not
import numpy as np
import pandas as pd
import pdb
import polars as pl

from mma_functions import cleaning, dataset, utils
from mma_functions.config import config

import requests

# This function takes a data frame with stats columns, another dataframe with every fighter's historic stats and calculates adjusted
# statistics on the stats data. By default, this function adjusts with the opponents Bayes smoothed stats

def add_adjusted_stats(
        stats_df, # The cleaned stats df with a bit of processing
        hist_df, # The aggregated historic stats from the aggregate_last_X_fights() function
        columns_with=["per_minute_bayes"], # A string to select a subset of columns to cal stats for with this string
        exclude_columns_with=[]): # A string to exclude a subset of columns to cal stats for with this string
    
    """
    Take a set of predictions and actuals and calculates a series of metrics
    
    Args:
        y_pred_prob (array): an array of probabilities for the predictions
        y_test_df: a series of actuals to test against
        classes (array): the classes of the predicted variable
        exclude_columns_with (list): a list of column strings to be excluded
        
    Returns:
        A dictionary of metrics
    
    """
    # pdb.set_trace()

    stop_if_not(type(columns_with) == type(["list"]), "columns_with type check")

    # We want to merge the historic stats of opponents onto the data
    # FIrst get opponent column
    new_stats_df = dataset.add_opponent_column(stats_df.copy())
    
    # Add per minute stats on
    if "per_minute" in columns_with:
        new_stats_df = calculate_per_minute_stats(new_stats_df, excluded_strings=["percent", "fight_length", "fight_minute", "opponent", "bayes", "per_minute"] + exclude_columns_with)
    
    # Find the columns to adjust
    columns_to_adjust = []
    adjustment_cols = []
    for item in columns_with:
        columns_to_adjust += [col for col in new_stats_df.columns if item in col and "against" not in col]
        # Now find corresponding historic columns for the columns to adjust
        adjustment_cols += ["hist_" + col.replace("_per_minute", "_against_per_minute").replace("_accuracy_percent", "_accuracy_percent_against") for col in columns_to_adjust]
        # Now add on the opposing stats for defense
        adjustment_cols += ["hist_" + col for col in columns_to_adjust]
        # Now add on the against stats
        columns_to_adjust += [col for col in new_stats_df.columns if item in col and "against" in col]
        
    # Remove any excluded strings
    for item in exclude_columns_with:
        columns_to_adjust = [col for col in columns_to_adjust if item not in col]
        adjustment_cols = [col for col in adjustment_cols if item not in col]
        
    # Check column correspondence
    stop_if_not((
        pd.Series(columns_to_adjust).str.replace("_against", "") == pd.Series(
            adjustment_cols).str.replace("_against", "")).sum() == 0,
        "adjusted stats column correspondence check")
    
    # Now merge on each opponents historic stats
    new_stats_df = new_stats_df.merge(hist_df, left_on=["event", "bout", "opponent"], right_on=["event", "bout", "fighter"], how="left")
    
    # Check all the columns are in the data
    missing_cols =  [col for col in adjustment_cols if col not in new_stats_df.columns]
    stop_if_not(len(missing_cols) == 0, "Columns in hist data check")
    
    # Calculate the adjusted stats
    new_stats = pd.DataFrame(
        new_stats_df[columns_to_adjust].to_numpy() - new_stats_df[adjustment_cols].to_numpy()
        , columns=[col + "_adj" for col in columns_to_adjust])
    
    # Final data
    final_stats_df = pd.concat([stats_df.reset_index(drop=True), new_stats], axis=1)
    
    # Check length
    stop_if_not(len(final_stats_df) == len(stats_df), "Dataset length")
    
    return final_stats_df

# This function takes a data frame and calculates the differentials

def add_fighter_differentials(df, zero_over_zero_fill=1):

    # pdb.set_trace()
    
    # Convert pandas DataFrame to Polars
    df_pl = pl.from_pandas(df)

    # Remove existing diff columns
    diff_cols = [c for c in df_pl.columns if "diff" in c or "offense_minus" in c]
    if diff_cols:
        df_pl = df_pl.select([c for c in df_pl.columns if c not in diff_cols])
    
    # Get sorted columns
    sorted_columns = sorted(df_pl.columns)    
       
    # Add direct differentials between fighter offence and defence
    fighter1_accuracy_off_cols = [col for col in sorted_columns if ("accuracy_percent" in col or "ctrl_minutes_per_minute" in col) and "diff" not in col and "fighter1" in col and "against" not in col]
    fighter2_accuracy_off_cols = [col for col in sorted_columns if ("accuracy_percent" in col or "ctrl_minutes_per_minute" in col) and "diff" not in col and "fighter2" in col and "against" not in col]
    fighter1_accuracy_def_cols = [col for col in sorted_columns if ("accuracy_percent" in col or "ctrl_minutes_against_per_minute" in col) and "diff" not in col and "fighter1" in col and "against" in col]
    fighter2_accuracy_def_cols = [col for col in sorted_columns if ("accuracy_percent" in col or "ctrl_minutes_against_per_minute" in col) and "diff" not in col and "fighter2" in col and "against" in col]
    
    # Check the columns line up
    stop_if_not(utils.check_identical_excl_string(fighter1_accuracy_off_cols, [col.replace("_against", "") for col in fighter2_accuracy_def_cols], "fighter1", "fighter2"))
    stop_if_not(utils.check_identical_excl_string(fighter2_accuracy_off_cols, [col.replace("_against", "") for col in fighter1_accuracy_def_cols], "fighter2", "fighter1"))
    
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
    fighter1_cols = [c for c in df_pl.columns if c.startswith("fighter1_") and "notformodel" not in c and "backtesting" not in c]
    fighter2_cols = [c for c in df_pl.columns if c.startswith("fighter2_") and "notformodel" not in c and "backtesting" not in c]

    # Remove string columns
    # Identify string columns
    string_cols = [c for c, dt in df_pl.schema.items() if dt in (pl.Utf8, pl.Categorical)]
    fighter1_cols = [c for c in fighter1_cols if c not in string_cols]
    fighter2_cols = [c for c in fighter2_cols if c not in string_cols]

    fighter1_cols.sort()
    fighter2_cols.sort()

    # Check that the columns correspond to one another
    if utils.check_identical_excl_string(fighter1_cols, fighter2_cols, "fighter1_", "fighter2_", check_both=True):
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
    
    stop_if_not(utils.check_identical_excl_string(f1_no_dates, f2_no_dates, "fighter1_", "fighter2_", check_both=True))

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
    for c in ["diff_weight", "diff_open_odds", "diff_best_closing_odds", "diff_mean_closing_odds", "diff_ratio_weight"]:
        if c in df_pl.columns:
            df_pl = df_pl.drop(c)

    # Check rows is consistent
    if (len(df_pl) == len(df)):
        print("Row number check PASSED")
    else:
        print("Row number check FAILED")
        raise SystemExit
    
    return df_pl.to_pandas()

# This function adds the stats for submissions landed to the stats data using the results data

def add_submissions_landed(fight_stats, fight_results):
        
    # pdb.set_trace()
    
    # Get submission success
    fighter1_submissions = fight_results[["event", "bout", "fighter1", "outcome", "method", "round"]].rename(columns={"fighter1": "fighter"})
    fighter2_submissions = fight_results[["event", "bout", "fighter2", "outcome", "method", "round"]].rename(columns={"fighter2": "fighter"})

    # Create a columne which shows the round of the submission
    fighter1_submissions["submission"] = np.where(
        (fighter1_submissions["outcome"] == "W/L") & (fighter1_submissions["method"] == "Submission"), 
        fighter1_submissions["round"], 
        0
    )
    
    fighter2_submissions["submission"] = np.where(
        (fighter2_submissions["outcome"] == "L/W") & (fighter2_submissions["method"] == "Submission"), 
        fighter2_submissions["round"], 
        0
    )
    
    # Concat fighter1 and fighter2
    submissions = pd.concat([
        fighter1_submissions,
        fighter2_submissions
        ],
        axis=0)
    
    # Merge onto the stats data
    new_stats = fight_stats.merge(
        submissions[["event", "bout", "fighter", "submission"]],
        on=["event", "bout", "fighter"], 
        how="left")
    
    # Check there are no NAs after 2010. Settled for 10 because some tiny misalignments are too cumbersome to fix
    stop_if_not(new_stats[new_stats["date"] > "2016-01-01"]["submission"].isna().sum() < 10)
        
    # Create round columns
    for round in range(1, 6):
        new_stats["round" + str(round) + "_sub_landed"] = np.where(new_stats["submission"] == round, 1, 0)
        
        # CHeck that landed isn't > attempts and if it is, replace it. (This only happens in a handful of cases, so not a systemic issue)
        new_stats["round" + str(round) + "_sub_attempt"] = np.where(
            new_stats["round" + str(round) + "_sub_landed"] > new_stats["round" + str(round) + "_sub_attempt"],
            new_stats["round" + str(round) + "_sub_landed"],
            new_stats["round" + str(round) + "_sub_attempt"]
        )
    
    new_stats["allrounds_sub_landed"] = new_stats[[col for col in new_stats.columns if "sub_landed" in col]].sum(axis=1)
    
    # CHeck that landed isn't > attempts and if it is, replace it
    new_stats["allrounds_sub_attempt"] = np.where(
        new_stats["allrounds_sub_landed"] > new_stats["allrounds_sub_attempt"],
        new_stats["allrounds_sub_landed"],
        new_stats["allrounds_sub_attempt"]
    )
    del([fighter1_submissions, fighter2_submissions])
    
    # Cleanup
    new_stats.drop(columns=["submission"], inplace=True)
    
    # Final checks
    stop_if_not(new_stats["allrounds_sub_landed"].max() == 1, "Max submissions landed check")
    stop_if_not(len(new_stats) == len(fight_stats), "Data length subs landed")
    
    return new_stats
    
    

def aggregate_last_X_fight_data(df, id_cols, group_by, prefix, X, per_minute_bayes, per_minute_calc=False, bayes_priors_df=None, bayes_alpha_plus_beta_rebase=None):
    """
    Convert fight data into cumulative fight history for the last X entries (excluding the current entry).
    THis version of the functions uses polars for the underlying calcs

    arguments:
    df: a data frame with one row for each fighter
    id_cols: the columns not to be aggregated
    group_by: a list of columns to be aggregated by
    prefix: a str to be added to the start of stat columns
    X: the number of historic fights to include
    per_minute_calc (bool): whether to add the per minute stats to the data or not
    bayes_priors_df: a data frame containing the bayes priods from the basian_priors() function. Only required if doing per minute calcs
    
    returns:S
    data frame showing the sum for the last 3 entries for each column excluding the current date.
    """
    
    # pdb.set_trace()
    
    # Get columns for grouping and stats
    id_columns = list(set(id_cols + group_by))
    stat_columns = [col for col in df.columns if col not in id_columns]
    
    # Convert to polars
    # Sort DataFrame by group_by columns and date for ordered cumulative calculation
    pl_sorted = pl.from_pandas(df).sort(by=group_by + ["date"])
    
    # Set the join columns
    join_cols = [col for col in id_cols if "date" not in col]
    
    # Check that the join cols give unique rows
    stop_if_not(len(pl_sorted) == len(pl_sorted.unique(subset=join_cols)), "Check that join cols identified in aggregate_last_X_fight_data inputs have a unique row for each occurance.")
    
    # pdb.set_trace()
    
    # Group by the group_by columns and calculate rolling sums for the last 3 entries
    rolling_data = pl_sorted.with_columns([
        pl.col(c).shift(1).rolling_sum(X, min_samples=1).over(group_by)
        for c in stat_columns
    ])
    
    
        
    # Rename the adj columns to make it clear what they are
    rolling_data = rolling_data.rename({col: col[:-4] + "_adjsum" if col.endswith("_adj") else col for col in rolling_data.columns})
    
    # Calculate rolling means for the adj statistics
    rolling_mean_data = pl_sorted.with_columns([
        pl.col(c).shift(1).rolling_mean(X, min_samples=1).over(group_by)
        for c in [col for col in stat_columns if "_adj" in col]
    ]).select(id_cols + [col for col in stat_columns if "_adj" in col])
    
    # Rename the adj columns to make it clear what they are
    rolling_mean_data = rolling_mean_data.rename({col: col[:-4] + "_adjmean" if col.endswith("_adj") else col for col in rolling_mean_data.columns})
    
    # Combine the dataset
    rolling_data = rolling_data.join(rolling_mean_data.drop("date"), on=join_cols, how="left")
            
    # If fights column is in, replace NAs with 0
    if "fights" in stat_columns:
        rolling_data = rolling_data.with_columns([pl.col("fights").fill_null(0)])
    
    # Rename columns with prefix and concatenate with the id_columns
    rolling_data = rolling_data.rename({x: prefix + x for x in rolling_data.columns if x not in id_columns})
    df_rolling = pl_sorted[join_cols].join(rolling_data, on=join_cols, how="left").to_pandas()

    # Calculate per minute stats if needed
    if per_minute_calc:
        df_rolling = calculate_per_minute_stats(df_rolling)
        if bayes_priors_df is not None and per_minute_bayes:
            df_rolling = calculate_per_minute_bayesian_stats(df_rolling, bayes_priors_df, bayes_alpha_plus_beta_rebase=bayes_alpha_plus_beta_rebase)
    
    return df_rolling

def aggregate_last_X_fight_data_pd(df, ignore_cols, group_by, prefix, X, per_minute_calc=False, bayes_priors_df=None, bayes_alpha_plus_beta_rebase=None):
    """
    Convert fight data into cumulative fight history for the last 3 entries (excluding the current entry).

    arguments:
    df: a data frame with one row for each fighter
    ignore_cols: the columns not to be aggregated
    group_by: a list of columns to be aggregated by
    prefix: a str to be added to the start of stat columns
    X: the number of historic fights to include
    per_minute_calc (bool): whether to add the per minute stats to the data or not
    bayes_priors_df: a data frame containing the bayes priods from the basian_priors() function. Only required if doing per minute calcs
    
    returns:S
    data frame showing the sum for the last 3 entries for each column excluding the current date.
    """
    
    pdb.set_trace()
    
    # Get columns for grouping and stats
    id_columns = list(set(ignore_cols + group_by))
    stat_columns = [col for col in df.columns if col not in id_columns]
    
    # Sort DataFrame by group_by columns and date for ordered cumulative calculation
    df_sorted = df.sort_values(by=group_by + ["date"]).reset_index(drop=True)
    
    # pdb.set_trace()
    
    # Group by the group_by columns and calculate rolling sums for the last 3 entries
    rolling_data = (
        df_sorted.groupby(group_by)
        .apply(lambda x: x[stat_columns].shift(1).rolling(window=X, min_periods=1).sum(), include_groups=False)
        .reset_index(level=group_by, drop=True) # Keep the original index
    ) # Not the fastest way ordinarily, but with partial windows is the best you can do
    
    # Rename the adj columns to make it clear what they are
    rolling_data.rename(columns={col: col[:-4] + "_adjsum" if col.endswith("_adj") else col for col in df.columns}, inplace=True)
    
    rolling_mean_data = (
        df_sorted.groupby(group_by)
        .apply(lambda x: x[[stat for stat in stat_columns if "_adj" in stat]].shift(1).rolling(window=X, min_periods=1).mean(), include_groups=False)
        .reset_index(level=group_by, drop=True) # Keep the original index
    )
    
    # Rename the adj columns to make it clear what they are
    rolling_mean_data.rename(columns={col: col[:-4] + "_adjmean" if col.endswith("_adj") else col for col in df.columns}, inplace=True)
    
    # Combine the dataset
    rolling_data = pd.concat([rolling_data, rolling_mean_data], axis=1)
    
    # # Convert your pandas DataFrame to a Dask DataFrame
    # ddf = dd.from_pandas(df_sorted, npartitions=4)  # Adjust the number of partitions as needed

    # # Ensure the data is sorted by the group key to allow for proper rolling
    # ddf = ddf.sort_values(by=group_by + ['date'])

    # # Perform the shift and rolling sum
    # def apply_rolling(group):
    #     group[stat_columns] = group[stat_columns].shift(1)
    #     return group[stat_columns].rolling(window=X, min_periods=1).sum()

    # # Apply the rolling operation with a custom meta to avoid the warning
    # rolling_data = (
    #     ddf.groupby(group_by, group_keys=False)
    #     .apply(apply_rolling, meta={col: 'f8' for col in stat_columns})  # Specify the meta here
    #     .compute()  # Convert back to pandas
    # )
        
    # If fights column is in, replace NAs with 0
    if "fights" in stat_columns:
        rolling_data["fights"] = rolling_data["fights"].fillna(0)
    
    # Rename columns with prefix and concatenate with the id_columns
    rolling_data = rolling_data.rename(columns=lambda x: prefix + x)
    df_rolling = pd.concat([df_sorted[id_columns], rolling_data], axis=1).reset_index(drop=True)

    # Calculate per minute stats if needed
    if per_minute_calc:
        df_rolling = calculate_per_minute_stats(df_rolling)
        if bayes_priors_df is not None:
            df_rolling = calculate_per_minute_bayesian_stats(df_rolling, bayes_priors_df, bayes_alpha_plus_beta_rebase=bayes_alpha_plus_beta_rebase)
    
    return df_rolling

# This function takes a list of column names and expands it to contain all the columns that will be needed to calculate differentials too

def all_columns_for_differentials(col_list):
    
    # pdb.set_trace()
    
    # Get the underlying for diff cols
    diff_cols = pd.Series([col for col in col_list if "diff_" in col and "diff_ratio_" not in col and "offense" not in col])
    new_fighter1_cols = diff_cols.str.replace("diff", "fighter1").str.replace("_against", "")
    new_fighter2_cols = diff_cols.str.replace("diff", "fighter2").str.replace("_against", "")
    
    # Create against versions
    new_fighter1_cols_against = new_fighter1_cols.str.replace("accuracy_percent", "accuracy_percent_against").str.replace("per_minute", "against_per_minute")
    new_fighter2_cols_against = new_fighter2_cols.str.replace("accuracy_percent", "accuracy_percent_against").str.replace("per_minute", "against_per_minute")
    
    # Get the underlying diff ratio columns needed
    diff_ratio_cols = pd.Series([col for col in col_list if "diff_ratio_" in col and "offense" not in col])
    new_fighter1_ratio_cols = diff_ratio_cols.str.replace("diff_ratio", "fighter1").str.replace("_against", "")
    new_fighter2_ratio_cols = diff_ratio_cols.str.replace("diff_ratio", "fighter2").str.replace("_against", "")
    
    # Create against versions
    new_fighter1_ratio_cols_against = new_fighter1_ratio_cols.str.replace("accuracy_percent", "accuracy_percent_against").str.replace("per_minute", "against_per_minute")
    new_fighter2_ratio_cols_against = new_fighter2_ratio_cols.str.replace("accuracy_percent", "accuracy_percent_against").str.replace("per_minute", "against_per_minute")
    
    # Combine these with the original list
    new_col_series = pd.concat([
        pd.Series([col for col in col_list if "diff" not in col]),
        new_fighter1_cols,
        new_fighter2_cols,
        new_fighter1_ratio_cols,
        new_fighter2_ratio_cols
        ]).drop_duplicates()
    
    # Now look for offence minus defence cols. There are two types. One with offense_minus_defense_ string, and one with fighter1_offense_minus_fighter
    # We'll need the against version and the non against version. In these lists, against is either after accuracy_percent or before _per_minute
        
    ### offense_minus_defense_
    accuracy_percent_cols = pd.Series([col for col in new_col_series.tolist() if "offense_minus_defense_" in col])
    
    # First get the base name of the columns
    raw_off_cols = accuracy_percent_cols.str.replace("offense_minus_defense_", "")
    
    # Because there's a mix of fighter 1 and 2, get a list for both
    raw_fighter1_cols = raw_off_cols.str.replace("fighter2", "fighter1")
    raw_fighter2_cols = raw_off_cols.str.replace("fighter1", "fighter2")
    
    # And add against cols
    raw_fighter1_cols_against = raw_fighter1_cols.str.replace("accuracy_percent", "accuracy_percent_against").str.replace("per_minute", "against_per_minute")
    raw_fighter2_cols_against = raw_fighter2_cols.str.replace("accuracy_percent", "accuracy_percent_against").str.replace("per_minute", "against_per_minute")
    
    
    ### fighter1_offense_minus_fighter
    split_offense = pd.Series([col for col in new_col_series.tolist() if "offense_minus_fighter" in col])
    
    # Get the underlying columns
    raw_split_offense = split_offense.str.replace("offense_minus_fighter1_defense_", "").str.replace("offense_minus_fighter2_defense_", "")
    
    # Get just the individual fighter cols
    fighter1_split_cols = raw_split_offense.str.replace("fighter2", "fighter1")
    fighter2_split_cols = raw_split_offense.str.replace("fighter1", "fighter2")
    
    # Now get the against cols
    fighter1_split_cols_against = fighter1_split_cols.str.replace("accuracy_percent", "accuracy_percent_against").str.replace("per_minute", "against_per_minute")
    fighter2_split_cols_against = fighter2_split_cols.str.replace("accuracy_percent", "accuracy_percent_against").str.replace("per_minute", "against_per_minute")

    #### All against cols
    # Need to create against versions of all columns for differentials
    all_non_diff = pd.Series([col for col in col_list if "diff" not in col and "offense" not in col])
    
    # Create non against versions
    all_fighter1_non_against = all_non_diff.str.replace("fighter2", "fighter1").str.replace("_against", "")
    all_fighter2_non_against = all_non_diff.str.replace("fighter1", "fighter2").str.replace("_against", "")
    
    # Now create against
    all_fighter1_against = all_fighter1_non_against.str.replace("accuracy_percent", "accuracy_percent_against").str.replace("per_minute", "against_per_minute")
    all_fighter2_against = all_fighter2_non_against.str.replace("accuracy_percent", "accuracy_percent_against").str.replace("per_minute", "against_per_minute")

    # Now combine all the additional columns and new ones
    full_col_list = pd.concat([
        pd.Series(col_list),
        new_fighter1_cols,
        new_fighter2_cols,
        new_fighter1_cols_against,
        new_fighter2_cols_against,
        new_fighter1_ratio_cols,
        new_fighter2_ratio_cols,
        new_fighter1_ratio_cols_against,
        new_fighter2_ratio_cols_against,
        raw_fighter1_cols,
        raw_fighter2_cols,
        raw_fighter1_cols_against,
        raw_fighter2_cols_against,
        fighter1_split_cols,
        fighter2_split_cols,
        fighter1_split_cols_against,
        fighter2_split_cols_against,
        all_fighter1_non_against,
        all_fighter2_non_against,
        all_fighter1_against,
        all_fighter2_against
        ], ignore_index=True).drop_duplicates().tolist()
    
    return full_col_list

def bayesian_priors(fight_stats_df, id_cols, group_by, fight_span, ignore_last_n_fights=15):
    
    """
    Takes a historical dataset and calculates the bayesian priors for each stat that will be needed to calculate
    Bayesian smoothed
    
    Args:
        hist_stats_df (df): a dataframe with historic stats
        id_cols (list): the cols used to identify a fight
        group_by (list): the columns we want to group by for the Bayesian priors
        fight_span (integer): the number of historic fights to use for the averages. 100 is about 18 months in more recent years
        
    Returns:
        A dictionary of metrics
    
    """
    # id_cols = ['event', 'bout', 'fighter']
    # fight_stats_df = fight_stats.copy().merge(fight_results[['event', 'bout', 'weightclass']], on = ['event', 'bout'], how = 'left')
    # group_by = ['weightclass']
    # fight_span = 100
    
    stats_df = fight_stats_df.copy()
    
    # pdb.set_trace()

    # Sort the data by weightclass and date, and add in sort by id_cols to ensure consistency in the ordering between different runs
    stats_df.sort_values(group_by + ["date"] + id_cols, inplace=True)

    # Get a list of all the numeric columns
    numeric_columns = stats_df.select_dtypes(include=["number", "float"]).columns.tolist()
    
    # Get the aggregate figures for each fight. Excluded the last 15 just to ensure no data from the same date is used. 2 * fightspan because we have a row per fighter
    if len(group_by) > 0:
        total_stats = stats_df.groupby(group_by, group_keys=False).apply(lambda x: x[numeric_columns].rolling(window=2 * fight_span).sum().shift(ignore_last_n_fights * 2 + 1), include_groups=False) # Shift 1 to not include current observation, then 2 * n fighters
    else:
        total_stats = stats_df[numeric_columns].rolling(window=2 * fight_span).sum().shift(ignore_last_n_fights * 2 + 1) # Shift 1 to not include current observation, then 2 * n fighters
   
    # Rename the columns
    total_stats.columns = ["bayes_total_" + col for col in total_stats.columns]
    
    # Merge onto the original data
    final_data = stats_df[id_cols + group_by].merge(total_stats, left_index=True, right_index=True, how="left")
 
    # Check we have the correct number of rows
    stop_if_not(len(final_data) == len(fight_stats_df))
    
    # Drop the group column and also only keep one record per bout. We want both fighters to use the same priors when calculating the bayes numbers
    final_data = final_data.drop_duplicates(subset=[col for col in id_cols if "fighter" not in col]).drop(group_by + ["fighter"], axis=1)
        
    # Return the data frame    
    return final_data

# This function takes the fight stats and fight stats history and calculates percentages (with bayes if needed)
# and adjusted percentages (with bayes if needed)

# It returns the original data frame with the new percentage columns added

def calculate_adjusted_percentages(
        fight_stats, 
        fight_stats_hist, # This must contain both against and normal versions of stats
        fight_stats_bayes_priors,
        data_processing_config):

    # pdb.set_trace()
    
    # Calculate percentages for each fight
    fight_stats_percentages = calculate_percentages(
        fight_stats,
        fight_stats_bayes_priors, 
        bayes_alpha_plus_beta_rebase=data_processing_config["bayes_alpha_plus_beta_rebase_percent"]
    )
    
    # Calculate percentages for each fighters history up to that point
    fight_stats_hist_percentages = calculate_percentages(
        fight_stats_hist,
        fight_stats_bayes_priors, 
        bayes_alpha_plus_beta_rebase=data_processing_config["bayes_alpha_plus_beta_rebase_percent"]
    )
    
    # Add on adjusted percentages if required
    if data_processing_config["add_adj_percentage_columns"]:
        
        fight_stats_percentages = add_adjusted_stats(
            stats_df=fight_stats_percentages, 
            hist_df=fight_stats_hist_percentages,
            columns_with=["accuracy_percent"],
            exclude_columns_with=["bayes"]
            )
    
    # Add on adjusted bayes percentages if required
    if data_processing_config["add_adj_percentage_bayes_columns"]:
        
        fight_stats_percentages = add_adjusted_stats(
            stats_df=fight_stats_percentages, 
            hist_df=fight_stats_hist_percentages,
            columns_with=["bayes_accuracy_percent"],
            exclude_columns_with=["_adj"]
            )
        
    # Check all the row numbers are correct
    stop_if_not(
        len(fight_stats) == len(fight_stats_percentages),
        "Percentages data check")
        
    del(fight_stats_hist_percentages)
    
    return fight_stats_percentages

def calculate_hist_opp_stance_stats(fight_stats, fighters, fight_stats_bayes_priors, data_processing_config):
    
    # pdb.set_trace()
    
    # Get the fight stats of each fighter against only specific types of fighters
    stances = fighters["stance"].str.replace(" ", "").unique().tolist()    
    
    # Merge on opponent stance
    fight_stats = fight_stats.merge(
        fighters[["fighter", "stance"]],
        left_on="opponent",
        right_on="fighter",
        how="left",
        suffixes=("", "_opp")
    )
    
    # Cleanup columns
    fight_stats = fight_stats.rename(columns={"stance": "opponent_stance"}).drop(columns=["fighter_opp"])

    # Get the historic records of each fighter for each opponent stance
    fight_stats_hist_opponent_stance = aggregate_last_X_fight_data(
        fight_stats,
        id_cols=["date", "event", "bout", "fighter", "opponent", "opponent_stance"],
        group_by=["fighter", "opponent_stance"],
        prefix="hist_opp_stance_",
        X=100,
        per_minute_bayes=data_processing_config["add_bayes_per_minute_columns"],
        per_minute_calc=True, 
        bayes_priors_df=fight_stats_bayes_priors,
        bayes_alpha_plus_beta_rebase=data_processing_config["bayes_alpha_plus_beta_rebase_per_minute"]
    ).drop(columns=["date", "opponent", "opponent_stance"])

    # Check the row numbers are correct
    stop_if_not(len(fight_stats) == len(fight_stats_hist_opponent_stance), "Opponent data length check")
    
    return(fight_stats_hist_opponent_stance)

def calculate_knocked_out_stats(fighter_results):
    
    ###########################################################################################
    # Calculate the date was last knocked out
    dates_knocked_out = fighter_results.copy()
    dates_knocked_out["knocked_out"] = np.where((dates_knocked_out["method"] == "KO/TKO") & (dates_knocked_out["result"] == "L"), 1, 0)
    dates_knocked_out["date_knocked_out"] = np.where(dates_knocked_out["knocked_out"] == 1, dates_knocked_out["date"], np.datetime64("NaT"))

    dates_knocked_out.sort_values(by=["fighter", "date"], inplace=True)

    # Calculate date last knocked out
    dates_knocked_out["date_last_knocked_out"] = dates_knocked_out.groupby(["fighter"])["date_knocked_out"].shift(1)
    dates_knocked_out["date_last_knocked_out"] = dates_knocked_out.groupby(["fighter"])["date_last_knocked_out"].ffill()

    # Calculate the number times knocked out previously
    dates_knocked_out["hist_times_knocked_out"] = dates_knocked_out.groupby(["fighter"])["knocked_out"].cumsum() - dates_knocked_out["knocked_out"]

    # Fill in missing rounds with round 3
    dates_knocked_out["round"] = dates_knocked_out["round"].fillna(3)

    # Calculate how many times each round they've been knocked out
    for round in [1, 2, 3, 4, 5]:
        
        # Create a flag for whether the person was knocked out in that fight
        dates_knocked_out["round_knockout"] = np.where(
            (dates_knocked_out["round"] == round) &
            (dates_knocked_out["method"] == "KO/TKO") & 
            (dates_knocked_out["result"] == "L"),
            1, 0)
        
        # Aggregate this for fighters
        dates_knocked_out["hist_times_knocked_out_round_" + str(round)] = dates_knocked_out.groupby("fighter")["round_knockout"].cumsum() - dates_knocked_out["round_knockout"]

    dates_knocked_out = dates_knocked_out[["event",
                                           "bout",
                                           "fighter", 
                                           "date_last_knocked_out", 
                                           "hist_times_knocked_out",
                                           "hist_times_knocked_out_round_1",
                                           "hist_times_knocked_out_round_2",
                                           "hist_times_knocked_out_round_3",
                                           "hist_times_knocked_out_round_4",
                                           "hist_times_knocked_out_round_5"]]

    # Fill NAs with 0
    dates_knocked_out.loc[:, ["hist_times_knocked_out", "hist_times_knocked_out_round_1", "hist_times_knocked_out_round_2", "hist_times_knocked_out_round_3", "hist_times_knocked_out_round_4", "hist_times_knocked_out_round_5"]] = dates_knocked_out[["hist_times_knocked_out", "hist_times_knocked_out_round_1", "hist_times_knocked_out_round_2", "hist_times_knocked_out_round_3", "hist_times_knocked_out_round_4", "hist_times_knocked_out_round_5"]].fillna(0) 

    # Fill NAs in dates with arbitrary early date
    dates_knocked_out["date_last_knocked_out"] = pd.to_datetime(dates_knocked_out["date_last_knocked_out"].fillna("1900-01-01"))

    # Check number of rows hasn't changed
    stop_if_not(len(dates_knocked_out) == len(fighter_results))
    
    return dates_knocked_out

def calculate_knocked_out_stats_old(fighter_results):
   pdb.set_trace()
    ###########################################################################################
   # Calculate the date was last knocked out
   dates_knocked_out = fighter_results.copy()
   dates_knocked_out["knocked_out"] = np.where((dates_knocked_out["method"] == "KO/TKO") & (dates_knocked_out["result"] == "L"), 1, 0)
   dates_knocked_out["date_knocked_out"] = np.where(dates_knocked_out["knocked_out"] == 1, dates_knocked_out["date"], np.datetime64("NaT"))

   dates_knocked_out.sort_values(by=["fighter", "date"], inplace=True)

   # Calculate date last knocked out
   dates_knocked_out["date_last_knocked_out"] = dates_knocked_out.groupby(["fighter"])["date_knocked_out"].shift(1)
   dates_knocked_out["date_last_knocked_out"] = dates_knocked_out.groupby(["fighter"])["date_last_knocked_out"].ffill()

   # Calculate the number times knocked out previously
   dates_knocked_out["hist_times_knocked_out"] = dates_knocked_out.groupby(["fighter"])["knocked_out"].cumsum() - dates_knocked_out["knocked_out"]

   row_check = len(dates_knocked_out)

   # Calculate how many times each round they've been knocked out
   ## IT LOOKS LIKE THE HISTORIC STATS ARE CORRECT, BUT THEY ARE ONLY APPEARING WHEN A FIGHTER GETS KOd IN ROUND 1 IN THE FIGHT WE WANT TO PREDICT
   for round in [1, 2, 3, 4, 5]:
       round_knocked_out = dates_knocked_out[(dates_knocked_out["round"] == round) & dates_knocked_out["knocked_out"] == 1].copy()
       round_knocked_out["hist_times_knocked_out_round_" + str(round)] = round_knocked_out.groupby(["fighter"])["knocked_out"].cumsum() - 1
       dates_knocked_out = dates_knocked_out.merge(round_knocked_out[["event", "bout", "fighter", "hist_times_knocked_out_round_" + str(round)]], 
                                                   on=["event", "bout", "fighter"], 
                                                   how="left")
       del(round_knocked_out)

   dates_knocked_out = dates_knocked_out[["event",
                                          "bout",
                                          "fighter", 
                                          "date_last_knocked_out", 
                                          "hist_times_knocked_out",
                                          "hist_times_knocked_out_round_1",
                                          "hist_times_knocked_out_round_2",
                                          "hist_times_knocked_out_round_3",
                                          "hist_times_knocked_out_round_4",
                                          "hist_times_knocked_out_round_5"]]

   # Fill NAs with 0
   dates_knocked_out.loc[:, ["hist_times_knocked_out", "hist_times_knocked_out_round_1", "hist_times_knocked_out_round_2", "hist_times_knocked_out_round_3", "hist_times_knocked_out_round_4", "hist_times_knocked_out_round_5"]] = dates_knocked_out[["hist_times_knocked_out", "hist_times_knocked_out_round_1", "hist_times_knocked_out_round_2", "hist_times_knocked_out_round_3", "hist_times_knocked_out_round_4", "hist_times_knocked_out_round_5"]].fillna(0) 

   # Fill NAs in dates with arbitrary early date
   dates_knocked_out["date_last_knocked_out"] = pd.to_datetime(dates_knocked_out["date_last_knocked_out"].fillna("1900-01-01"))
    
   return dates_knocked_out

# This function takes fight stats and calculates per minute versions on the dataset then returns the full data with added per minute columns

# Can set some excluded strings to not include columns containing those phrases

def calculate_per_minute_bayesian_stats(fight_stats_hist, bayes_priors_df, bayes_alpha_plus_beta_rebase, excluded_strings=["percent", "fight_length", "fight_minute", "ratio", "per_minute", "baye"]):

    # Merge the bayesian prios onto the data
    stats_df = fight_stats_hist.merge(bayes_priors_df, on=["event", "bout"], how="left")
    
    stop_if_not(len(stats_df) == len(fight_stats_hist))
    
    # pdb.set_trace()
    
    new_stats_df = fight_stats_hist.copy()
    
    # Go through each round's stats individually
    rounds = ["round1", "round2", "round3", "round4", "round5", "allrounds"]
    
    for round in rounds:
        # Get all round columns
        round_columns = [col for col in stats_df.columns if (round in col)]
        
        if len(round_columns) == 0:
            continue
        
        # Get the prefix of the columns
        prefix = str(round_columns[0]).split(str(round))[0]
        
        # Now exclude any columns we don't need per minute stats for
        for string in excluded_strings + ["bayes"]:
            round_columns = [col for col in round_columns if string not in col]
    
        # Check all the Bayes prior columns are in the data
        if prefix == "":
            stop_if_not(all(item.replace("_against", "") in stats_df.columns for item in ["bayes_total_" + col for col in round_columns]), "Bayes columns check in data")
        else:
            stop_if_not(all(item.replace("_against", "") in stats_df.columns for item in [col.replace(prefix, "bayes_total_") for col in round_columns]), "Bayes columns check in data")
        
        # Work out the associated bayes columns for numerator and denominator
        if prefix == "":
            bayes_numerator_cols = ["bayes_total_" + item.replace("_against", "") for item in round_columns]
        else:
            bayes_numerator_cols = [item.replace("_against", "").replace(prefix, "bayes_total_") for item in round_columns]
        
        # Now calculate the alpha values based on the rebased number of minutes
        bayes_alphas = stats_df[bayes_numerator_cols] * bayes_alpha_plus_beta_rebase / stats_df["bayes_total_" + round + "_fight_minutes"].to_numpy()[:, np.newaxis]
        
        # Calculate the per minute stats and merged onto data
        new_stats_df = pd.concat([
            new_stats_df,
            pd.DataFrame((stats_df[round_columns] + bayes_alphas.to_numpy()).div(stats_df[prefix + round + "_fight_minutes"].to_numpy() + bayes_alpha_plus_beta_rebase, axis=0).to_numpy(),
                         columns=[(col + "_per_minute_bayes") for col in round_columns]),          
        ], axis=1)
    
    # Do a check
    if len(new_stats_df) == len(fight_stats_hist):
        return new_stats_df
    else:
        print("Error: per minute stats couldn't be calculated")
        return None

# This function takes fight stats and calculates per minute versions on the dataset then returns the full data with added per minute columns

# Can set some excluded strings to not include columns containing those phrases

def calculate_per_minute_stats(fight_stats_hist, excluded_strings=["percent", "fight_length", "fight_minute", "ratio", "baye", "per_minute"]):

    stats_df = fight_stats_hist.copy().reset_index(drop=True)
    
    # pdb.set_trace()
    
    # Go through each round's stats individually
    rounds = ["round1", "round2", "round3", "round4", "round5", "allrounds"]
    
    for round in rounds:
        # Get all round columns
        round_columns = [col for col in stats_df.columns if (round in col)]
        
        if len(round_columns) == 0:
            continue
        
        # Get the prefix of the columns
        prefix = str(round_columns[0]).split(str(round))[0]
        
        # Now exclude any columns we don't need per minute stats for
        for string in excluded_strings:
            round_columns = [col for col in round_columns if string not in col]
    
        # Calculate the per minute stats and merged onto data
        stats_df = pd.concat([
            stats_df,
            pd.DataFrame(stats_df[round_columns].div(stats_df[prefix + round +"_fight_minutes"].to_numpy(), axis=0).to_numpy(),
                         columns=[(col + "_per_minute") for col in round_columns]),          
        ], axis=1)
    
    # Do a check
    if len(stats_df) == len(fight_stats_hist):
        return stats_df
    else:
        print("Error: per minute stats couldn't be calculated")
        return None

# This function takes the merged data or fight stats data and calculates percentages

def calculate_percentages(merged_data_df, fight_stats_bayes_priors, bayes_alpha_plus_beta_rebase):

    # pdb.set_trace()
    merged_data = merged_data_df.copy()
    
    # Calculate percentages for those columns with attempts and lands
    percent_columns = [col for col in merged_data.columns if "_attempt" in col and "per_minute" not in col]
    landed_columns = [col for col in merged_data.columns if "_landed" in col and "per_minute" not in col]
    percent_columns.sort()
    landed_columns.sort()
    
    # Check we have all the corresponding columns
    if ([string.replace("_attempt", "") for string in percent_columns] == [string.replace("_landed", "") for string in landed_columns]):
        print("Percent column correspodence check PASSED")
    else:
        print("Percent column correspodence check FAILED")
        raise SystemExit
    
    # Now calculate the percentages
    new_percent_cols = [col.replace("_attempt", "_accuracy_percent") for col in percent_columns]
    merged_data[landed_columns].index = merged_data[percent_columns].index
    
    # Shapes where strange so had to use numpy
    new_data = pd.DataFrame(utils.safe_divide_np(merged_data[landed_columns].to_numpy(), merged_data[percent_columns].to_numpy(), zero_zero_fill=np.nan), columns=new_percent_cols)
    
    # Remove percentages for sample sizes below minimum thresholds, excluding takedowns
    percent_columns_no_td = [col for col in percent_columns if "_td_" not in col]
    new_percent_cols_no_td = [col for col in new_percent_cols if "_td_" not in col]
    new_data.reset_index(drop=True, inplace=True)
    new_data[new_percent_cols_no_td] = np.where(merged_data[percent_columns_no_td] < config["min_attempts"], np.nan, new_data[new_percent_cols_no_td])
    
    # Combine with data
    merged_data = pd.concat([merged_data, new_data], axis=1)

    ##########################################
    # Now calculate the bayes percentages
    
    
    if fight_stats_bayes_priors is not None:
        
        # Add on the bayes variables
        merged_data_bayes = merged_data.merge(fight_stats_bayes_priors, on=["event", "bout"], how="left")
        
        # Find what numerators and denominators are needed
        bayes_numerators = ["bayes_total_" + (col.replace("fighter1_", "").replace("fighter2_", "").replace("hist_", "").replace("opp_type_", "").replace("opp_stance_", "").replace("_against", "").replace("last_1_fights_", "").replace("last_2_fights_", "").replace("last_3_fights_", "")) for col in landed_columns]
        
        bayes_denominators = ["bayes_total_" + (col.replace("fighter1_", "").replace("fighter2_", "").replace("hist_", "").replace("opp_type_", "").replace("opp_stance_", "").replace("_against", "").replace("last_1_fights_", "").replace("last_2_fights_", "").replace("last_3_fights_", "")) for col in percent_columns]
        
        # Check these are in the bayes data
        stop_if_not(all(item in merged_data_bayes.columns for item in bayes_numerators + bayes_denominators), "Bayes priors check in data")
        
        # Calculate the bayes percentage accuracies
        new_bayes_percent_cols = [col.replace("_attempt", "_bayes_accuracy_percent") for col in percent_columns]
        
        # Calculate the bayes alphas rebased to the required number
        bayes_alphas = merged_data_bayes[bayes_numerators] * bayes_alpha_plus_beta_rebase / merged_data_bayes[bayes_denominators].to_numpy()
        
        # Calculate the bayes percentages
        new_data = pd.DataFrame(utils.safe_divide_np(
            merged_data_bayes[landed_columns].fillna(0).to_numpy() + bayes_alphas.to_numpy(), 
            merged_data_bayes[percent_columns].fillna(0).to_numpy() + bayes_alpha_plus_beta_rebase, 
            zero_zero_fill=np.nan
            ), 
            columns=new_bayes_percent_cols
        )
        
        # Check there are non > 0
        stop_if_not(new_data[new_data>1].sum().sum() == 0, "Bayes percentages less than 1")
        stop_if_not(new_data[new_data<0].sum().sum() == 0, "Bayes percentages greater than 0")
        
        # Merge the data
        merged_data = pd.concat([merged_data, new_data], axis=1)
    
    # Check we have the correct number of rows
    stop_if_not(len(merged_data_df) == len(merged_data), "Dataset length")
    
    # Percent check
    stop_if_not(merged_data[new_percent_cols].max().max() <= 1.0, "Percentage calculations 100 max")
    stop_if_not(merged_data[new_percent_cols].min().min() >= 0.0, "Percentage calculations 100 max")
    
    del([new_data, percent_columns, landed_columns, new_percent_cols])
    
    return merged_data

# This function takes the fighter results data (obtained after converting the fight results to be one row per fighter, not per fight) and derives various columns for win/loss stats and adds them to the data

def calculate_win_loss_stats(fighter_results, finishes=["KO/TKO", "Submission"], last_x_fights=10):

    # pdb.set_trace()    

    ###########################################################################################
    # Calculate number of previous wins/losses
    win_loss_stats = fighter_results.copy()
    win_loss_stats["date_win"] = np.where(win_loss_stats["result"] == "W", win_loss_stats["date"], np.datetime64("NaT"))
    win_loss_stats["date_loss"] = np.where(win_loss_stats["result"] == "L", win_loss_stats["date"], np.datetime64("NaT"))

    win_loss_stats.sort_values(by=["fighter", "date"], inplace=True)

    # Calculate date of last win and loss
    win_loss_stats["date_last_win"] = win_loss_stats.groupby(["fighter"])["date_win"].shift(1)
    win_loss_stats["date_last_loss"] = win_loss_stats.groupby(["fighter"])["date_loss"].shift(1)
    # Fill in the places where there is NAs because of losses and wins between each other
    win_loss_stats["date_last_win"] = win_loss_stats.groupby(["fighter"])["date_last_win"].ffill()
    win_loss_stats["date_last_loss"] = win_loss_stats.groupby(["fighter"])["date_last_loss"].ffill()

    # Count number of historic wins and losses
    win_loss_stats["win"] = np.where(win_loss_stats["result"] == "W", 1, 0)
    win_loss_stats["loss"] = np.where(win_loss_stats["result"] == "L", 1, 0)
    win_loss_stats["hist_wins"] = win_loss_stats.groupby(["fighter"])["win"].cumsum() - win_loss_stats["win"]
    win_loss_stats["hist_losses"] = win_loss_stats.groupby(["fighter"])["loss"].cumsum() - win_loss_stats["loss"]
    win_loss_stats["win_finish"] = np.where((win_loss_stats["result"] == "W") & (win_loss_stats["method"].isin(finishes)), 1, 0)
    win_loss_stats["loss_finish"] = np.where((win_loss_stats["result"] == "L") & (win_loss_stats["method"].isin(finishes)), 1, 0)
    
    # Begin a count of finishes to be populated as we go through the methods
    win_loss_stats["hist_wins_finishes"] = 0
    win_loss_stats["hist_losses_finishes"] = 0
    
    # Get a unique list of methods
    unique_method_list = win_loss_stats["method"].dropna().unique()
    
    # Now calculate the wins and losses by method
    for method in unique_method_list:
        # Create a binary flag to say whether a win or loss was by a method
        win_loss_stats["W_" + method] = (win_loss_stats["method"] == method).astype(int) * win_loss_stats["win"]
        win_loss_stats["L_" + method] = (win_loss_stats["method"] == method).astype(int) * win_loss_stats["loss"]
        
        # Now count the historical wins and losses by that method
        win_loss_stats["hist_wins_" + method] = win_loss_stats.groupby(["fighter"])["W_" + method].cumsum() - win_loss_stats["W_" + method]
        win_loss_stats["hist_losses_" + method] = win_loss_stats.groupby(["fighter"])["L_" + method].cumsum() - win_loss_stats["L_" + method]
        
        # If this method is a finish, then add to the finish count
        if method in finishes:
            win_loss_stats["hist_wins_finishes"] += win_loss_stats["hist_wins_" + method]
            win_loss_stats["hist_losses_finishes"] += win_loss_stats["hist_losses_" + method]

    # Now calculate the win rate for the last X fights. Do in polars for speed
    win_loss_stats_pl = pl.from_pandas(win_loss_stats)
    for x in range(1, (last_x_fights + 1)):
        win_loss_stats_pl = win_loss_stats_pl.with_columns([
            pl.col("win").shift(1).rolling_mean(x, min_samples=x).over("fighter").alias(f"hist_win_percent_{x}_fights"),
            pl.col("win_finish").shift(1).rolling_mean(x, min_samples=x).over("fighter").alias(f"hist_win_finish_percent_{x}_fights"),
            pl.col("loss_finish").shift(1).rolling_mean(x, min_samples=x).over("fighter").alias(f"hist_loss_finish_percent_{x}_fights")
            ])
    
    # for x in range(1, (last_x_fights + 1)):
    #     win_loss_stats[f'hist_win_percent_{x}_fights'] = (
    #         win_loss_stats.groupby('fighter')['win']
    #         .transform(lambda s: s.shift().rolling(x, min_periods=x).mean())
    #     )
        
    #     win_loss_stats[f'hist_win_finish_percent_{x}_fights'] = (
    #         win_loss_stats.groupby('fighter')['win_finish']
    #         .transform(lambda s: s.shift().rolling(x, min_periods=x).mean())
    #     )
        
    #     win_loss_stats[f'hist_loss_finish_percent_{x}_fights'] = (
    #         win_loss_stats.groupby('fighter')['loss_finish']
    #         .transform(lambda s: s.shift().rolling(x, min_periods=x).mean())
    #     )

    win_loss_stats = win_loss_stats_pl.to_pandas()
    del(win_loss_stats_pl)

    # Add a flag to say whether a fighter has moved a weightclass since their last 2 fights (adding it into the win loss data because it has it in the format required already)
    win_loss_stats["weightclass_m1"] = win_loss_stats.groupby("fighter")["weightclass"].shift(1)
    win_loss_stats["weightclass_m2"] = win_loss_stats.groupby("fighter")["weightclass"].shift(2)

    win_loss_stats["moved_weightclass"] = np.where(
        (~win_loss_stats["weightclass_m2"].isna()) & 
        (~win_loss_stats["weightclass_m2"].isna()) &
        (win_loss_stats["weightclass_m1"] == win_loss_stats["weightclass_m2"]) & 
        (win_loss_stats["weightclass_m2"] != win_loss_stats["weightclass"]),
        1,
        0
    )

    ############################
    # Add per fight stats
    numerator_strings = ["hist_wins", "hist_losses"]

    # Calculate denominator
    win_loss_stats["hist_fights"] = win_loss_stats.groupby("fighter").cumcount()
    
    # Get the column names
    cols = []
    for i in range(0, len(numerator_strings)):
        cols = cols + [col for col in win_loss_stats.columns if numerator_strings[i] in col]
    
    new_cols = {col: col + "_per_fight" for col in cols}

    win_loss_stats = win_loss_stats.assign(
        **{new_col: win_loss_stats[old_col] / win_loss_stats["hist_fights"] 
           for old_col, new_col in new_cols.items()}
    )

    # Deal with any infinities
    win_loss_stats.loc[:, [col + "_per_fight" for col in cols]] = win_loss_stats[[col + "_per_fight" for col in cols]].replace([np.inf, -np.inf], np.nan)

    ############################
    # Clean up columns
    win_loss_stats.drop(columns=["hist_fights", "weightclass", "weightclass_m1", "weightclass_m2", "method", "round", "time format", "date", "rounds", "result", "win", "loss", "win_finish", "loss_finish", "date_win", "date_loss"] + ("W_" + unique_method_list).tolist() + ("L_" + unique_method_list).tolist(), inplace=True)

    # Merge back onto the fighter results data
    output_data = fighter_results.merge(win_loss_stats, on=["event", "bout", "fighter"], how="left")
    
    # Do a check
    if len(output_data) == len(fighter_results):
        return output_data.copy()
    else:
        print("Error: win loss stats couldn't be merged")
        return None

# This function uses the fight stats history to define the type of fighter someone is
# It uses the historic fight stats for each fighter, and the results data at fighter level (not bout)

# It returns a pd Series of fighter types

def define_fighter_type(fight_stats_hist, fighter_results, config):
    
    # pdb.set_trace()
    
    merged_data = fight_stats_hist.merge(fighter_results, on=["event", "bout", "fighter"], how="left")
    
    # Use the thresholds in the config. Loop through the fighter types
    for fighter_type in list(config["fighter_type_thresholds"].keys()):
        # Check if its empty
        if config["fighter_type_thresholds"][fighter_type] == None:
            merged_data["fighter_type"] = fighter_type
        # If not empty, look for the conditions
        else:
            conditions = {key:value for key, value in config["fighter_type_thresholds"][fighter_type].items()}

            # Find which rows meet the conditions
            meets_conditions = (merged_data[list(conditions)] > pd.Series(conditions)).all(axis=1)
            
            # Then set those fighters to the fighter type
            merged_data.loc[meets_conditions, "fighter_type"] = fighter_type
    
    # Check the results is the correct length
    if len(merged_data["fighter_type"]) == len(fight_stats_hist):
        return merged_data["fighter_type"]
    else:
        print("Error: fighter type couldn't be derived")
        return None

# This function gets the elevation of a location

def get_elevation_series(locations, api_key):
    """
    Fetch the elevation above sea level for a pandas Series of locations using Google Maps Elevation API.

    Parameters:
        locations (pd.Series): A pandas Series containing location strings.
        api_key (str): Your Google Maps API key.

    Returns:
        pd.Series: A pandas Series containing elevations in meters or None if unavailable.
    """
    elevation_cache = {}  # Cache to store elevations for unique locations

    def fetch_elevation(location):
        if location in elevation_cache:
            return elevation_cache[location]  # Return cached elevation

        try:
            # Geocode the location to get coordinates
            geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json"
            geocode_params = {"address": location, "key": api_key}
            geocode_response = requests.get(geocode_url, params=geocode_params, timeout=10).json()

            if geocode_response["status"] != "OK":
                elevation_cache[location] = None
                return None  # Location not found

            # Get latitude and longitude
            lat = geocode_response["results"][0]["geometry"]["location"]["lat"]
            lon = geocode_response["results"][0]["geometry"]["location"]["lng"]

            # Fetch elevation using Google Maps Elevation API
            elevation_url = "https://maps.googleapis.com/maps/api/elevation/json"
            elevation_params = {"locations": f"{lat},{lon}", "key": api_key}
            elevation_response = requests.get(elevation_url, params=elevation_params, timeout=10).json()

            if elevation_response["status"] == "OK" and "results" in elevation_response:
                elevation = elevation_response["results"][0]["elevation"]
                elevation_cache[location] = elevation  # Cache the result
                return elevation

            elevation_cache[location] = None
            return None  # Elevation data not available

        except Exception as e:
            elevation_cache[location] = None
            return None  # Handle errors gracefully

    return locations.apply(fetch_elevation)

def longest_streak(series, target_value):
    """
    Calculates the longest streak of consecutive occurrences of the target value in a pandas Series,
    ignoring `0` but including streaks interrupted by other non-zero values.

    Parameters:
    - series (pd.Series): The input Series.
    - target_value (float64): The value (e.g., 1 or -1) for which to find the longest streak.

    Returns:
    - int: The length of the longest streak for the target value.
    """
    
    
    # pdb.set_trace()
    # Replace `0` with `None` to break streaks
    filtered = series[series != 0]

    # Create a mask for consecutive `target_value`
    mask = filtered == target_value

    # Group by changes in the mask and count streak lengths
    streak_lengths = mask.groupby((~mask).cumsum()).sum()

    # Return the longest streak length
    return streak_lengths.max() or 0

# This function switches who is fighter1 and fighter2 on the merged final dataset

def switch_fighter1_fighter2(df_original=None, filter_boolean=None, X=None, y=None, for_model=True):
    """
    # This functions takes a dataframe and a filter of same length and switches fighter1 and fighter2 in the rows labelled 'True'
    
    It can either take a data frame as a whole, or take data in the form of X y, i.e. predictors split into a data frame X and y is the observed outcomes
    
    """
    # pdb.set_trace()
    
    # First, check the type of y. It will be forced to int
    if y is not None:
        if y.dtype != "int32":
            print("Warning: y vector passed into switch_fighter1_fighter2() is not an int32 type. Output will be forced to int32")

    # Make the filter an array from a list to avoid index issues
    filter_boolean = np.array(list(filter_boolean))

    # Define the data to be switched
    if X is None:
        df = df_original.copy()
        df = df[filter_boolean].copy()
    else:
        df = X.copy()
        df = df[filter_boolean].copy()
        y_new = y.copy()
        y_new = y_new[filter_boolean].copy()

    final_cols = df.columns.tolist()
        
    # Undo any cleaning that's been done on the data
    df = cleaning.undo_clean_inf(df)

    # Swap column names
    swap_dict = {"fighter1": "fighter2", "fighter2": "fighter1"}
    df.columns = df.columns.str.replace(r"fighter1|fighter2", lambda x: swap_dict.get(x.group(0), x.group(0)), regex=True)

    # # First rename fighter1 columns
    # df.columns = df.columns.str.replace('fighter1', 'fighterX')

    # # Now rename fighter2
    # df.columns = df.columns.str.replace('fighter2', 'fighter1')

    # # Now create the final name for fighter1
    # df.columns = df.columns.str.replace('fighterX', 'fighter2')

    # Now reverse and diff columns
    diff_columns = [col for col in df.columns if "diff" in col and "diff_ratio" not in col]
    df[diff_columns] = -df[diff_columns]

    diff_ratio_columns = [col for col in df.columns if "diff_ratio" in col]
    df[diff_ratio_columns] = 1 / df[diff_ratio_columns]

    ## Now change individual columns

    # Reverse the matchup. Have only done this when the matchup column is as a whole, dummies wont be properly adjusted! But they never seem to be high in importance so assumed not included in final models
    if "matchup" in df.columns:
        fighter_types = df["matchup"].str.split("-vs-", expand=True)
        df["matchup"] = fighter_types[1] + "-vs-" + fighter_types[0]

    # Change the outcomes
    if "outcome" in df.columns:
        df["outcome"] = np.where(df["outcome"] == "W/L", "L/W", np.where(df["outcome"] == "L/W", "W/L", df["outcome"]))
        
    # Now combine this data with the rows not switched
    if X is None:
        final_df = pd.concat([df, df_original.copy().reset_index(drop=True)[~filter_boolean]])
    else:
        # # Create the filtered original data for those records not adjusted
        # X_filtered = X.copy()[~filter_boolean]
        
        # # Now re-align the columns of the new data
        # df = df.reindex(columns=X_filtered.columns)
        
        # final_df = pd.DataFrame(index=range(len(df) + len(X_filtered)), columns=df.columns)
        # final_df.loc[: len(df) - 1] = df.values
        # final_df.loc[len(df) :] = X_filtered.values
        
        # # Set column types to match the original df
        # final_df = final_df.astype(df.dtypes.to_dict())

        final_df = pd.concat([df.copy(), X.copy()[~filter_boolean]])
    
    # Handle cases where we have X, y input instead of a single data frame
    if y is not None: # Check this works with W/L and isn't 0, 1, 2
        y_new = y_new.astype("int32")
        y_df = pd.DataFrame(y_new)
        y_df.iloc[:, 0] = np.where(y_df.iloc[:, 0] == 2, 0, np.where(y_df.iloc[:, 0] == 0, 2, y_df.iloc[:, 0]))
        
        # Now combine this data with the rows not switched
        y_new = pd.concat([y_df.iloc[:, 0], y[~filter_boolean]])
        
    # pdb.set_trace()
    # Recalculate diff columns because Infs etc might have been cleaned before. Only keep the columns needed
    save_index = final_df.index.copy()
    diff_df = add_fighter_differentials(final_df)
    
    # Keep the diff in dates from the original data and use the diffs from the rest of the data
    diff_df = pd.concat([
        diff_df[[col for col in diff_df.columns if "date" not in col and "dob" not in col]].reset_index(drop=True), 
        final_df[[col for col in final_cols if ("date" in col or "dob" in col) or col not in diff_df.columns]].reset_index(drop=True)
        ], axis=1)
    
    # Set index back to what it was
    diff_df.index = save_index
    
    final_df = diff_df.loc[:, final_cols].copy()
    
    # Clean columns of infs if necessary
    if for_model:
        final_df = cleaning.clean_inf_and_large(final_df)
        final_df = dataset.format_cols_for_model(final_df)
        
    # Make sure the data types of the columns are as before
    if X is None:
        final_df = final_df.astype(df_original.dtypes.to_dict())
    else:
        final_df = final_df.astype(X.dtypes.to_dict())
        y_new = y_new.astype("int32")
    
    # Check we have the correct number of rows and columns
    if X is None:
        stop_if_not(final_df.shape == df_original.shape, "Dataset length check")
    else:
        stop_if_not(final_df.shape == X.shape, "Dataset length check")

    # # Ensure the old index is preserved
    # ## CAN'T GET IT TO WORK
    # if X is None:
    #     final_df = final_df.loc[df_original.index]
    # else:
    #     final_df2 = final_df.loc[X.index]
    #     y_new2 = y_new.loc[y.index.values]
    

    if y is None:
        return final_df.copy()
    else:
        return final_df.copy(), y_new.copy()
