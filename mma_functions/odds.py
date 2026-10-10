from ablinston_ml import stop_if_not
from datetime import datetime, timedelta
import fastparquet as fp
import joblib
import numpy as np
import pandas as pd

from mma_functions import dataset, features, utils
from mma_functions.config import config, model_hyper_params, odds_scrape_config

# This function takes the odds data and finds the event and bout information from the event details and fight stats data

# It returns the odds data with the correct event and bout matched along with a correct date

def add_event_and_bout_to_odds_data(odds_df, event_details_df, stats_df, date_check="2010-01-01"):
    
    # # debug
    # odds_df = parsed_odds_df.copy()
    # event_details_df = updated_event_details_df.copy()
    # stats_df = parsed_fight_stats_df.copy()
    # pdb.set_trace()
    
    # First create event and bout dummies if not there
    if "event" not in odds_df.columns:
        odds_df["event"] = np.nan
    if "bout" not in odds_df.columns:
        odds_df["bout"] = np.nan
        
    # Convert the event details date to a datetime
    event_dates = event_details_df.copy()[["event", "date"]]
    if pd.api.types.is_datetime64_any_dtype(event_dates["date"]):
        event_dates.loc[:, "event_date"] = event_dates["date"]
    else:
        event_dates.loc[:, "event_date"] = event_dates["date"].apply(lambda x: datetime.strptime(x, "%B %d, %Y"))
    
    # Merge the dates onto to stats data
    new_stats_df = stats_df.merge(event_dates[["event", "event_date"]], how="left", on="event")
        
    # Clean the bouts and fighter columns
    new_stats_df[["bout", "fighter"]] = utils.clean_text_columns(new_stats_df[["bout", "fighter"]])
    odds_df.loc[:,["fighter", "opponent"]] = utils.clean_text_columns(odds_df[["fighter", "opponent"]])
    
    # Drop columns not needed and duplicates. Drop NA because they will affect the match
    new_stats_df = new_stats_df[["event", "bout", "fighter", "event_date"]].drop_duplicates()
    new_stats_df = new_stats_df.dropna()
    
    # Create an opponent column
    dataset.add_opponent_column(new_stats_df)
    
    ######################
    # Going in reverse priority, try to replace the event and bout from matches in the data
    
    # Create alternative dates on the odds data 
    new_odds_df = odds_df.copy()
    
    # Create a custom bout
    if "bout" not in new_odds_df.columns:
        new_odds_df["bout"] = np.nan

    ######################
    # Do a merge based on opponent. This is a last resort where the date of some bouts is missing and so can't match
    
    # Create a data frame with opponents details.
    opponents_df = new_stats_df.copy()
    opponents_df = opponents_df[["event", "bout", "event_date", "fighter", "opponent"]]
    
    # Remove all rows that have duplicates i.e. where fighters have fought more than once
    opponents_df = opponents_df.drop_duplicates(subset=["fighter", "opponent"], keep=False)

    
    # # Replace events and bouts with those from the opponents data #  Removed this because it caused issues with duplicates
    # new_odds_df = replace_event_and_bout_odds(odds_df = new_odds_df, stats_df = opponents_df, join_on = ['fighter', 'opponent'], day_shift = 0, odds_event_priority = False)
    
    # Now, working from 2 days down to one, fill the missing events
    new_odds_df = replace_event_and_bout_odds(odds_df=new_odds_df, stats_df=new_stats_df, join_on=["fighter", "date", "opponent"], day_shift=2, odds_event_priority=False)
    new_odds_df = replace_event_and_bout_odds(odds_df=new_odds_df, stats_df=new_stats_df, join_on=["fighter", "date", "opponent"], day_shift=-2, odds_event_priority=False)
    new_odds_df = replace_event_and_bout_odds(odds_df=new_odds_df, stats_df=new_stats_df, join_on=["fighter", "date", "opponent"], day_shift=1, odds_event_priority=False)
    new_odds_df = replace_event_and_bout_odds(odds_df=new_odds_df, stats_df=new_stats_df, join_on=["fighter", "date", "opponent"], day_shift=-1, odds_event_priority=False)
    new_odds_df = replace_event_and_bout_odds(odds_df=new_odds_df, stats_df=new_stats_df, join_on=["fighter", "date", "opponent"], day_shift=0, odds_event_priority=False)
        
    # Check the new data is a consistent length with the old data
    if len(new_odds_df) != len(odds_df):    
        print("ERROR: MATCHING EVENTS AND BOUTS TO ODDS CREATED TOO MANY RECORDS")
        return None
        
    # Do a check to see where we have not found UFC events
    ufc_bouts_missing_odds = new_stats_df[new_stats_df["event_date"] > date_check]["bout"][~new_stats_df["bout"].isin(new_odds_df["bout"].tolist())]
    ufc_events_missing_odds = new_stats_df[new_stats_df["event_date"] > date_check]["event"][~new_stats_df["event"].isin(new_odds_df["event"].tolist())]
    
    # Give a warning about any that haven't been matched to an event and bout
    if len(ufc_events_missing_odds) > 0:
        print(f"The following events after {date_check} are missing odds data: {set(ufc_events_missing_odds)}")
    
    # Remove columns no longer needed    
    new_odds_df = new_odds_df[odds_df.columns]
        
    return new_odds_df

# This function adds some random noise to odds columns using a normal distribution. It does this with a multiplicative factor

def add_noise_to_odds(df, norm_mean=1, norm_sd=0.1, include_backtesting_odds=False):
        
    # pdb.set_trace()
    
    # Get a list of the odds columns required. Don't include backtesting odds
    if include_backtesting_odds:
        odds_cols = [col for col in df.columns if "odds" in col]
    else:
        odds_cols = [col for col in df.columns if "odds" in col and "backtesting" not in col]
    
    # Create data with just the odds
    odds_df = df[["event", "bout"] + odds_cols].copy()
    
    # Add randomness to the odds. Loop through so we can check each column
    if include_backtesting_odds:
        odds_to_shift = ["open_odds", "mean_closing_odds", "best_closing_odds", "backtesting_odds"]
    else:
        ["open_odds", "mean_closing_odds", "best_closing_odds"]
        
    for col in odds_to_shift:
        if "fighter1_" + col in odds_cols:
            rands = np.random.normal(norm_mean, norm_sd, len(odds_df))
            odds_df["fighter1_" + col] = (odds_df["fighter1_" + col] - 1) * rands + 1
            odds_df["fighter2_" + col] = (odds_df["fighter2_" + col] - 1) / rands + 1
        
    # Calculate all odds cols
    if "fighter1_mean_closing_odds" in odds_cols:
        odds_df["fighter1_odds_change"] = odds_df["fighter1_mean_closing_odds"] - odds_df["fighter1_open_odds"]
        odds_df["fighter2_odds_change"] = odds_df["fighter2_mean_closing_odds"] - odds_df["fighter2_open_odds"]
        
        # Add on a flag to say whether the odds have changed by more than 0.1
        odds_df["fighter1_odds_change_gt_0"] = np.where(odds_df["fighter1_odds_change"] < 0, -1, np.where(odds_df["fighter1_odds_change"] > 0, 1, 0))
        
        # Add on a flag to say whether the odds have changed by more than 0.1
        odds_df["fighter2_odds_change_gt_0"] = np.where(odds_df["fighter2_odds_change"] < 0, -1, np.where(odds_df["fighter2_odds_change"] > 0, 1, 0)) 
    
    # Recalulate true odds
    odds_df = add_true_odds(odds_df, "mean_closing_odds")

    # Add diff columns
    odds_df = features.add_fighter_differentials(odds_df) 
    
    # Now merge the new odds with the old data
    new_df = df.drop(columns=odds_cols).merge(odds_df[["event", "bout"] + odds_cols], on=["event", "bout"], how="left")
    
    # check the data is the right shape
    stop_if_not(new_df.shape == df.shape, "Add noise - shape check")
    
    return new_df
    
    

# This function adds true odds based on a column suffix
def add_true_odds(df, col_suffix="mean_closing_odds"):
    
    # Change odds into true odds
    df[["fighter1_true_closing_odds", "fighter2_true_closing_odds"]] = calculate_true_odds(df, "fighter1_" + col_suffix, "fighter2_" + col_suffix)
    df[["fighter1_true_open_odds", "fighter2_true_open_odds"]] = calculate_true_odds(df, "fighter1_open_odds", "fighter2_open_odds")

    # Recalculate change
    df["fighter1_true_odds_change"] = df["fighter1_true_closing_odds"] - df["fighter1_true_open_odds"]
    df["fighter2_true_odds_change"] = df["fighter2_true_closing_odds"] - df["fighter2_true_open_odds"]
    
    return df

# This function takes american style odds and converts them to decimals

def american_odds_to_decimal(american_odds):
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1

def back_odds_to_lay_odds(back_odds_series):
    """
    
    A function that takes in a series or array of back odds for an outcome 1 and converts it into lay odds for outcome 2. ONLY WORKS FOR A 2 OUTCOME EVENT


    """
    
    
    return back_odds_series / (back_odds_series - 1)

def calculate_all_true_odds_columns(merged_data, impute_missing_betfair, odds_suffix_for_true_odds, impute_one_sided_missing_odds):
    
    """
    This function calculates all the required true odds columns on the processed merged dataset
    
    Parameters
    -------
    merged_data : data frame
    merged data with fighter1 and fighter2 columns
    
    impute_missing_betfair: bool
    whether missing betfair odds will be imputed with best_closing_odds
    
    odds_suffix_for_true_odds: string
    the column suffix for the odds column to be used in calculating true odds
    
    impute_one_sided_missing_odds: bool
    whether to infer the odds where one fighter is missing but we have odds for the other
    
    Returns
    -------
    Data frame with the true odds columns added
    
    """
    
    # pdb.set_trace()
        
    # Fill in missing columns if Betfair has nothing
    if "fighter1_notformodel_betfair_odds_datetime" not in merged_data.columns:
        merged_data["fighter1_notformodel_betfair_odds_datetime"] = np.nan
        merged_data["fighter2_notformodel_betfair_odds_datetime"] = np.nan
        
    # Impute missing odds if they are missing for one fighter but not the other
    if impute_one_sided_missing_odds:
        for odds in ["betfair_exch_odds", "best_closing_odds", "open_odds"]:
            merged_data["fighter1_" + odds] = np.where(merged_data["fighter1_" + odds].isna(), merged_data["fighter2_" + odds] / (merged_data["fighter2_" + odds] - 1), merged_data["fighter1_" + odds])
            merged_data["fighter2_" + odds] = np.where(merged_data["fighter2_" + odds].isna(), merged_data["fighter1_" + odds] / (merged_data["fighter1_" + odds] - 1), merged_data["fighter2_" + odds])
    
    # Impute Betfair odds if required
    if impute_missing_betfair:
        merged_data["fighter1_betfair_exch_odds_imputed"] = np.where(merged_data["fighter1_betfair_exch_odds"].isna(), merged_data["fighter1_best_closing_odds"], merged_data["fighter1_betfair_exch_odds"])
        merged_data["fighter2_betfair_exch_odds_imputed"] = np.where(merged_data["fighter2_betfair_exch_odds"].isna(), merged_data["fighter2_best_closing_odds"], merged_data["fighter2_betfair_exch_odds"])
        
        # If we're using these in true odds, use the imputed
        if odds_suffix_for_true_odds == "betfair_exch_odds":
            odds_suffix_for_true_odds = "betfair_exch_odds_imputed"
    
    # Change odds into true odds
    merged_data[["fighter1_true_closing_odds", "fighter2_true_closing_odds"]] = calculate_true_odds(merged_data, "fighter1_" + odds_suffix_for_true_odds, "fighter2_" + odds_suffix_for_true_odds)
    merged_data[["fighter1_true_open_odds", "fighter2_true_open_odds"]] = calculate_true_odds(merged_data, "fighter1_open_odds", "fighter2_open_odds")
    
    # Recalculate change
    merged_data["fighter1_true_odds_change"] = merged_data["fighter1_true_closing_odds"] - merged_data["fighter1_true_open_odds"]
    merged_data["fighter2_true_odds_change"] = merged_data["fighter2_true_closing_odds"] - merged_data["fighter2_true_open_odds"]
    
    # Get the true odds scrape date
    for fighter in ["fighter1", "fighter2"]:
        if "betfair_exch_odds" in odds_suffix_for_true_odds:
            merged_data[fighter + "_notformodel_true_odds_asat_datetime"] = np.where(
                merged_data[fighter + "_betfair_exch_odds"].isna(),
                merged_data[fighter + "_notformodel_date_fighter_odd_scrape"].astype(str),
                merged_data[fighter + "_notformodel_betfair_odds_datetime"].astype(str))
        else:
            merged_data[fighter + "_notformodel_true_odds_asat_datetime"] = merged_data[fighter + "_notformodel_date_fighter_odd_scrape"].astype(str)
    
    # # Clean up columns NEED THESE IN DATA PROCESSING FOR BACKTESTING ODDS
    # merged_data.drop(columns = ['fighter1_betfair_exch_odds_imputed', 'fighter2_betfair_exch_odds_imputed'], inplace = True)
    
    return merged_data

def calculate_target_lay_odds(probabilities, target_ev, commission=0):
    
    """
    This function takes a series of outcome probabilities and a set of target EVs and calculates the LAY odds required to achieve that with a given rate of commission on winnings. NB The probabilities should be of the event happening i.e. the lay bet LOSING
    
    Parameters
    -------
    probabilities : series
    A series of probabilities of an event occuring
    
    target_ev: float64
    The EV we want to target
    
    commission: float64
    A proportion between 0 and 1 for the brokers commission on winnings
    
    
    Returns
    -------
    Series of lay odds to get to target EV in decimal format

    """
    
    return (1 - commission + commission * probabilities - target_ev) / probabilities

def calculate_target_odds(probabilities, target_ev, commission=0):
    
    """
    This function takes a series of outcome probabilities and a set of target EVs and calculates the odds required to achieve that with a given rate of commission on winnings
    
    Parameters
    -------
    probabilities : series
    A series of probabilities of events
    
    target_ev: float64
    The EV we want to target
    
    commission: float64
    A proportion between 0 and 1 for the brokers commission on winnings
    
    
    Returns
    -------
    Series of odds in decimal format

    """
    
    return (target_ev - commission * probabilities + 1) / (probabilities * (1 - commission))

# This function takes a dataframe with two odds columns and normalises them so that there is no spread or gap between them. This creates the true odds


def calculate_true_odds(df, col1, col2):
    """
    Calculates the true odds in decimal form from two columns of decimal odds.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        col1 (str): The name of the first column with decimal odds.
        col2 (str): The name of the second column with decimal odds.

    Returns:
        pd.DataFrame: A DataFrame with the true odds for the specified columns.
    """
    # Calculate implied probabilities
    implied_prob_col1 = 1 / df[col1]
    implied_prob_col2 = 1 / df[col2]

    # Calculate the total implied probability (overround)
    overround = implied_prob_col1 + implied_prob_col2

    # Calculate true probabilities
    true_prob_col1 = implied_prob_col1 / overround
    true_prob_col2 = implied_prob_col2 / overround

    # Convert true probabilities back to decimal odds
    true_odds_col1 = 1 / true_prob_col1
    true_odds_col2 = 1 / true_prob_col2

    # # Replace the original odds
    # df[col1] = true_odds_col1
    # df[col2] = true_odds_col2

    # Create and return a new DataFrame with the results
    return pd.DataFrame({"col1": true_odds_col1, "col2": true_odds_col2})

# This function cleans the date columns on the fighters betting odds data. It takes in an odds data frame

# and returns a cleaned data frame

# AB: DO NOT USE THIS FUNCTION. THERE ARE LOTS OF EVENTS CALLED JUST 'UFC' AND THIS JUST AMALGAMATES THEM ALL TOGETHER

def clean_odds_dates(odds_df):
    
    cleaned_dates_df = odds_df.groupby("event").agg({"date": utils.safe_mode})
    
    # Now merge this back onto the odds data
    odds_df.drop("date", axis=1, inplace=True)
    odds_df2 = odds_df.merge(cleaned_dates_df, on="event")

    del(cleaned_dates_df)
    
    return odds_df

# This function calculates the inverse of odds

def invert_odds(odds):
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1

def odds_to_probabilities(odds_fighter1, odds_fighter2, normalise=True):
    """
    Convert the closing odds of two fighters into the probabilities of win, loss, or draw.

    Parameters:
    - odds_fighter1 (float): Closing odds for fighter 1 (decimal).
    - odds_fighter2 (float): Closing odds for fighter 2 (decimal).

    Returns:
    - probabilities (dict): Dictionary containing probabilities for fighter 1 win, fighter 2 win, and draw.
    """
    # Calculate implied probabilities
    prob_fighter1 = 1 / odds_fighter1
    prob_fighter2 = 1 / odds_fighter2
    
    # Total implied probability
    total_prob = prob_fighter1 + prob_fighter2

    if normalise:
        # Calculate normalized probabilities
        prob_fighter1 = prob_fighter1 / total_prob
        prob_fighter2 = prob_fighter2 / total_prob

    # Assuming there is a draw, calculate the draw probability (just for demonstration)
    # This could be based on additional logic or odds if available
    prob_draw = 1 - (prob_fighter1 + prob_fighter2)

    # Create a DataFrame to hold the results
    probabilities = pd.DataFrame({
        "L/W": np.round(prob_fighter2, 8),
        "OTH": np.round(prob_draw, 8),
        "W/L": np.round(prob_fighter1, 8)
    })

    return probabilities

# This function processes the cleaned odds data so that it is ready for merging with the other datasets

def process_odds_data(fight_df, event_df, betfair_df, odds_scrape_config, average_odds_model=None, impute_missing_averages=True):
    
    # pdb.set_trace()
     
    new_event_df = event_df.copy()
    
    # Convert opening odds to decimal
    fight_df.loc[:, ["open_odds", "closing_odds_low", "closing_odds_high"]] = fight_df[["open_odds", "closing_odds_low", "closing_odds_high"]].map(american_odds_to_decimal)
    new_event_df[odds_scrape_config["events_colnames"]] = new_event_df[odds_scrape_config["events_colnames"]].map(american_odds_to_decimal)
    
    # Where we have more than 4 odds values for a fight, calculate the mean
    new_event_df = new_event_df[new_event_df[odds_scrape_config["events_colnames"]].isna().sum(axis=1) <= 5]
    new_event_df["mean_closing_odds"] = np.nanmean(new_event_df[odds_scrape_config["events_colnames"]], axis=1)
    
    # # Remove any duplicates. New odds data will be on top so keep that
    # new_event_df[new_event_df.duplicated(subset = ['event', 'bout', 'fighter'], keep = False)].sort_values(['event', 'bout', 'fighter'])
    new_event_df = new_event_df.drop_duplicates(subset=["event", "bout", "fighter"])
    
    # Merge these means onto the fight df
    new_fight_df = fight_df.merge(new_event_df[["event", "bout", "fighter", "mean_closing_odds"]], on=["event", "bout", "fighter"], how="left")
    
    # Check row number
    stop_if_not(len(new_fight_df) == len(fight_df), "Odds merge check")
    
    if impute_missing_averages:
        # Fill in missing odds with modelled results
        new_fight_df.loc[new_fight_df["mean_closing_odds"].isna(), "mean_closing_odds"] = average_odds_model.predict(new_fight_df[average_odds_model.features])[new_fight_df["mean_closing_odds"].isna()]
    
    # Record the best odds as the highest (as that is what will give the biggest return)
    new_fight_df["best_closing_odds"] = new_fight_df["closing_odds_high"].copy()
    
    # Calculate the odds change
    new_fight_df["odds_change"] = new_fight_df["mean_closing_odds"] - new_fight_df["open_odds"]
    
    # Keep only the columns we need, and remove duplicates (these duplicates are where the event and bout are empty)
    new_fight_df = new_fight_df[["event", "bout", "fighter", "open_odds", "mean_closing_odds", "best_closing_odds", "odds_change", "notformodel_date_fighter_odd_scrape"]].drop_duplicates(subset=["event", "bout", "fighter"])
    
    # Process the betfair data ready for merging
    betfair_to_merge = betfair_df[~betfair_df["event"].isna()].drop_duplicates(subset=["event", "bout", "fighter"])
    
    # Get the columns needed
    if "notformodel_betfair_odds_datetime" in betfair_to_merge.columns:
        betfair_to_merge = betfair_to_merge[["event", "bout", "fighter", "betfair_exch_odds", "notformodel_betfair_odds_datetime", "notformodel_best_betfair_odds_after_cutoff"]]
    else:
        betfair_to_merge = betfair_to_merge[["event", "bout", "fighter", "betfair_exch_odds"]]
    
    # Merge on the betfair odds
    new_fight_df = new_fight_df.merge(betfair_to_merge, on=["event", "bout", "fighter"],  how="outer")
    
    # Add on a flag to say whether the odds have changed by more than 0.1
    new_fight_df["odds_change_gt_0"] = np.where(new_fight_df["odds_change"] < 0, -1, np.where(new_fight_df["odds_change"] > 0, 1, 0))     
    
    # Check the number of rows
    stop_if_not(len(new_fight_df) >= len(fight_df), "Process odds length check")
    
    return new_fight_df

# This function takes the cleaned odds data and replaces events and bouts with those from a fight stats dataset. It merges on fighter, opponent and date, but can also give a time delta for the date.

# You can set whether to take the odds data as priority or the stats data as prioirty when determining event and bout

def replace_event_and_bout_odds(odds_df, stats_df, join_on=["fighter", "date", "opponent"], day_shift=0, odds_event_priority=False):
    
    # pdb.set_trace()
    
    new_odds_df = odds_df.copy()
    
    # Perform the shift in date
    new_odds_df["date"] = new_odds_df["date"] + timedelta(days=day_shift)
    
    new_odds_df.rename(columns={"event": "event_old", "bout": "bout_old"}, inplace=True)
    new_odds_df = new_odds_df.merge(stats_df, how="left", left_on=join_on, right_on=["event_date" if item == "date" else item for item in join_on])
    
    if odds_event_priority:
        # Now fix the event and bout columns based on the matches
        new_odds_df["event"] = np.where(~new_odds_df["event_old"].isna(), 
                                        new_odds_df["event_old"],
                                        np.where(~new_odds_df["event"].isna(), 
                                                 new_odds_df["event"], 
                                                 new_odds_df["event_old"]))
        new_odds_df["bout"] = np.where(~new_odds_df["bout_old"].isna(), 
                                       new_odds_df["bout_old"],
                                        np.where(~new_odds_df["bout"].isna(), 
                                                 new_odds_df["bout"], 
                                                 new_odds_df["bout_old"]))
    else:
        # Now fix the event and bout columns based on the matches
        new_odds_df["event"] = np.where(~new_odds_df["event"].isna(), 
                                        new_odds_df["event"],
                                        new_odds_df["event_old"])
        new_odds_df["bout"] = np.where(~new_odds_df["bout"].isna(), 
                                       new_odds_df["bout"],
                                       new_odds_df["bout_old"])

    # Shift the date back to where it was originally
    new_odds_df["date"] = new_odds_df["date"] - timedelta(days=day_shift)
    
    # Drop extra columns
    new_odds_df.drop(columns=["event_old", "bout_old"], inplace=True)
    try:
        new_odds_df.drop(columns=["event_date"], inplace=True)
    except:
        pass
    
    # Do some final checks on the data
    stop_if_not(new_odds_df["date"].tolist() == odds_df["date"].tolist(), "Date check")
    stop_if_not(new_odds_df.shape == odds_df.shape, "Dataset shape check")
    
    return new_odds_df

# This function replaces the odds data on the final dataset with new cleaned odds data

def replace_odds_data(final_data, cleaned_odds_path):
    
    ## Debug
    final_data = forecasting_data.copy()
    cleaned_odds_path = "./data/" + config["fighter_odds_file_name"].replace(".parquet", "_clean.parquet")
    ###
    
    # Read in the new odds data
    fight_odds = fp.ParquetFile(cleaned_odds_path).to_pandas()
    fight_data = final_data.copy()
    
    # Rename the odds cols
    fight_odds = process_odds_data(fight_odds)
    
    # Get all the odds columns
    odds_cols = [col for col in final_data.columns if "odd" in col]
    
    # Merged with the fight data with the odds columns removed
    fight_data = dataset.merge_df_both_fighters(
        left_df=final_data[[col for col in final_data.columns if col not in odds_cols]],
        df_list=[
            fight_odds
        ]
    ).reset_index(drop=True)
    
    # Add on the new fighter1 and fighter2 odds
    fighter1_data = fight_odds.copy().add_prefix("fighter1_")
    fighter2_data = fight_odds.copy().add_prefix("fighter2_")
    
    # Merge the fighter data
    
    # NOT FINISHED BECAUSE DECIDED FULL PROCESSING NEEDS TO BE RUN EVERY TIME
    
    return 

# This function takes new cleaned data and adds the odds to the existing merged data, recalculating differentials where needed

# It returns the data frame with new odds data.

# THIS FUNCTION SHOULD ONLY BE USED FOR UPCOMING EVENTS DUE TO CONFIG CHOICES

def update_odds_on_merged_data(
        cleaned_fight_odds_data_path: str,
        cleaned_event_odds_data_path: str,
        cleaned_betfair_odds_data_path: str,
        final_data_folder: str,
        average_odds_model_path,
        config,
        data_processing_config,
        upcoming_only=False
        ):
    
    # #########
    # cleaned_fight_odds_data_path = './data/' + config['fighter_odds_file_name'].replace('.parquet', '_clean.parquet')
    # cleaned_event_odds_data_path = './data/' + config['event_odds_file_name'].replace('.parquet', '_clean.parquet')
    # final_data_folder = './data/final_dataset/'
    # average_odds_model_path = './results/average_odds_model.joblib'
    # #########
    
    # Load in the datasets
    fight_odds = fp.ParquetFile(cleaned_fight_odds_data_path).to_pandas()
    event_odds = fp.ParquetFile(cleaned_event_odds_data_path).to_pandas()
    betfair_odds = fp.ParquetFile(cleaned_betfair_odds_data_path).to_pandas()
    merged_data = fp.ParquetFile(final_data_folder).to_pandas() # pq.ParquetDataset(final_data_folder).read().to_pandas()
    average_odds_model = joblib.load(average_odds_model_path)
    
    # pdb.set_trace()
    
    if upcoming_only:
        merged_data = merged_data[merged_data["outcome"].isna()].copy()
    
    # Save original shape for later
    original_shape = merged_data.shape
        
    # Assume that the event and bout have been added correctly as part of the data cleaning process.
    # But need to remove any with event or bout still missing
    fight_odds = fight_odds[(~fight_odds["event"].isna()) & (~fight_odds["bout"].isna())].copy()

    # Replace any older odds on the data
    fight_odds.sort_values("notformodel_date_fighter_odd_scrape", ascending=False, inplace=True)
    fight_odds = fight_odds.drop_duplicates(subset=["event", "bout", "fighter"])
    
    # Give placeholder column for betfair odds
    if "notformodel_best_betfair_odds_after_cutoff" not in betfair_odds.columns:
        betfair_odds["notformodel_best_betfair_odds_after_cutoff"] = np.nan
    
    # Process the data
    fight_odds = process_odds_data(fight_odds, event_odds, betfair_odds, odds_scrape_config, average_odds_model)
    
    # Remove the columns that will be replaced
    cols_to_remove = [col for col in merged_data.columns if col.replace("fighter1_", "").replace("fighter2_", "") in fight_odds.columns]
    cols_to_remove = [col for col in cols_to_remove if col not in ["event", "bout", "date"]]

    # Remove old columns
    merged_data.drop(columns=cols_to_remove, inplace=True)
    
    # Merge on the new odds
    merged_data = dataset.merge_df_both_fighters(
        left_df=merged_data,
        df_list=[
            fight_odds
        ]
    ).reset_index(drop=True)

    # Recalculate all true odds
    merged_data = calculate_all_true_odds_columns(merged_data, impute_missing_betfair=data_processing_config["impute_missing_betfair"], odds_suffix_for_true_odds=data_processing_config["upcoming_odds_suffix_for_true_odds"], impute_one_sided_missing_odds=data_processing_config["impute_one_sided_missing_odds"])
    
    if (data_processing_config["impute_missing_betfair"] == True) & (model_hyper_params["odds_to_use"] == "betfair_exch_odds"):
        model_hyper_params["odds_to_use"] = "betfair_exch_odds_imputed"

    # Add what odds are to be used in model training and backtesting
    merged_data["fighter1_backtesting_odds"] = merged_data["fighter1_" + model_hyper_params["odds_to_use"]].copy()
    merged_data["fighter2_backtesting_odds"] = merged_data["fighter2_" + model_hyper_params["odds_to_use"]].copy()
    
    # Recalculate differentials
    merged_data = features.add_fighter_differentials(merged_data)
    
    # Check on data
    # cols_that_should_be_in_original = [col for col in merged_data.columns if 'notformodel_true_odds_asat_datetime' not in col]
    stop_if_not(len(merged_data) == original_shape[0])
    stop_if_not(len(merged_data.columns) == original_shape[1])

    return merged_data
