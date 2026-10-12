from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from mma_functions import odds, pipelines
from mma_functions.config import odds_scrape_config

# This function checks five data frames containing events, fight results and stats. It checks that the same
# events are covered in each one.

# It returns a data frame detailing the checks done and whether they passed (True) or not (False)

def check_data_coverage(event_details_df, fight_details_df, fight_results_df, fight_stats_df, fight_odds_df, fighter_details_df, fighter_tott_df, event_odds_df, betfair_odds_df, upcoming_events_df, upcoming_fight_details_df, betfair_upcoming_odds_df, config):
    
    # # Debug
    # event_details = parsed_event_details_df.copy()
    # fight_details = parsed_fight_details_df.copy()
    # fight_results = parsed_fight_results_df.copy()
    # fight_stats = parsed_fight_stats_df.copy()
    # fight_odds = parsed_odds_df.copy()
    # fighter_details = parsed_fighter_details_df.copy()
    # fighter_tott = parsed_fighter_tott_df.copy()
    
    # Create a flag to say whether checks passed
    all_checks_passed = True
    # pdb.set_trace()
    # Make copies so we don't overwrite any of the original data frames
    event_details = event_details_df.copy()
    fight_details = fight_details_df.copy()
    fight_results = fight_results_df.copy()
    fight_stats = fight_stats_df.copy()
    fight_odds = fight_odds_df.copy()
    event_odds = event_odds_df.copy()
    fighter_details = fighter_details_df.copy()
    fighter_tott = fighter_tott_df.copy()
    betfair_odds = betfair_odds_df.copy()
    upcoming_events = upcoming_events_df.copy()
    upcoming_fight_details = upcoming_fight_details_df.copy()
    betfair_upcoming_odds = betfair_upcoming_odds_df.copy()
    
    # Clean all the datasets
    cleaned_dict = pipelines.clean_all_raw_data(event_details, upcoming_events, upcoming_fight_details, fighter_tott, fight_results, fight_stats, fight_odds, event_odds, betfair_odds, betfair_upcoming_odds)
    
    # Change the column names to lower
    for df in [event_details, fight_details, fight_results, fight_stats, fight_odds, fighter_details, fighter_tott, event_odds_df, betfair_odds, betfair_upcoming_odds]:
        df.columns = df.columns.str.lower()
                
            
    ##################################
    #### Data cleaning
    ##################################
    
    # # Clean all the columns that we wish to use
    # for df in [event_details, fight_details, fight_results, fight_stats, fight_odds, fighter_details, fighter_tott, event_odds]:
    #     for col in ['event', 'bout', 'fighter']:
    #         try:
    #             df[[col]] = clean_text_columns(df[[col]])
    #         except:
    #             continue
            
    # Process the odds data to combine events
    fight_odds.loc[:, ["open_odds", "closing_odds_low", "closing_odds_high"]] = fight_odds[["open_odds", "closing_odds_low", "closing_odds_high"]].apply(pd.to_numeric, errors="coerce")
    fight_odds = odds.process_odds_data(cleaned_dict["fighter_odds"].drop_duplicates(subset=["event", "bout", "fighter"]), cleaned_dict["event_odds"], cleaned_dict["betfair_odds"], odds_scrape_config, impute_missing_averages=False)
    
    ## The events data is the first updated, so is assumed to be complete
    # Check that all the events have some fight details in the data
    missing_events_details = event_details["event"][~event_details["event"].isin(fight_details["event"].tolist())]
    
    if (len(missing_events_details) > 0):
        print(f"Error: The following events are missing from the fight_details data: {missing_events_details}")
        all_checks_passed = False
                    
    # Check that all events are covered in the fight results data
    missing_events_results = event_details["event"][~event_details["event"].isin(fight_results["event"].tolist())]
    
    if (len(missing_events_results) > 0):
        print(f"Error: The following events are missing from the fight_results data: {missing_events_results}")
        all_checks_passed = False
        
    # Check that all events are covered in the fight stats data
    missing_events_stats = event_details["event"][~event_details["event"].isin(fight_stats["event"].tolist())]
    
    if (len(missing_events_stats) > 0):
        print(f"Error: The following events are missing from the fight_stats data: {missing_events_stats}")
        all_checks_passed = False

    ## Next, check the fight results details and stats have the same bouts in them
    
    # Get a series detailing all bouts covered in the data
    all_bouts = pd.Series(list(set(fight_details["bout"].tolist() + fight_results["bout"].tolist() + fight_stats["bout"].tolist())))
    
    missing_bouts_on_results = list(set(all_bouts[~all_bouts.isin(fight_results["bout"].tolist())]))
    missing_bouts_on_details = list(set(all_bouts[~all_bouts.isin(fight_details["bout"].tolist())]))
    missing_bouts_on_stats = list(set(all_bouts[~all_bouts.isin(fight_stats["bout"].tolist())]))
    
    if (len(missing_bouts_on_details) > 0):
        print(f"Error: The following bouts are missing from the fight_stats data: {missing_bouts_on_details}")
        all_checks_passed = False
    if (len(missing_bouts_on_results) > 0):
        print(f"Error: The following bouts are missing from the fight_stats data: {missing_bouts_on_results}")
        all_checks_passed = False
    if (len(missing_bouts_on_stats) > 0):
        print(f"Error: The following bouts are missing from the fight_stats data: {missing_bouts_on_stats}")
        all_checks_passed = False
        
    # Check the event dates on the odds data to see if any are missing
    # We only consider events after 2010 as betting odds generally aren't available in early UFC
    # event_details['date'] = event_details['date'].apply(lambda x: datetime.strptime(x, '%B %d, %Y'))
    post_2010_events = event_details["event"][event_details["date"] > config["start_date_validation_checks"]]
    
    # Only keep UFC events
    post_2010_events = pd.Series([event for event in post_2010_events if "ufc" in event])
    
    missing_events_from_odds = post_2010_events[~post_2010_events.isin(fight_odds["event"].tolist())]
    
    if (len(missing_events_from_odds) > 0):
        print(f"Error: The following events are missing from the odds data: {missing_events_from_odds}")
        all_checks_passed = False
    
    # Check if there are any bouts missing from the odds data
    post_2010_bouts = fight_stats["bout"][fight_stats["event"].isin(post_2010_events)]
    
    missing_bouts_from_odds = set(post_2010_bouts[~post_2010_bouts.isin(fight_odds["bout"].tolist())])
    
    if (len(missing_bouts_from_odds) > 0):
        print(f"Warning: {len(missing_bouts_from_odds)} UFC bouts out of {len(all_bouts)} are missing from the odds data.")
        # all_checks_passed = False # Not setting to false because some bouts have no odds data because the fighters names are wrong. See UFC 298 for e.g.
        
    # Find which recent events have missing average odds data
    recent_events = event_details["event"][event_details["date"] > "2024-01-01"]
    
    # Find which events are missing mean closing odds
    missing_events_average_odds = recent_events[~recent_events.isin(fight_odds[~fight_odds["mean_closing_odds"].isna()]["event"])].tolist()
    if (len(missing_events_average_odds) > 0):
        print(f"Warning: The following events are missing from the event odds data: {missing_events_average_odds}")
        
    # Find which fighters appear most in missed bouts
    # Split each element into two lists
    list1, list2 = zip(*(char.split("vs.") for char in missing_bouts_from_odds))
    # Combine both lists into a single list
    combined_list = list(list1) + list(list2)
    # Count occurrences of each item
    pd.Series(list(list1) + list(list2)).value_counts()
        
    ## Check that all the fighters have fighter details data
    # Add a fighter column
    fighter_details["fighter"] = fighter_details["first"] + fighter_details["last"]
    
    # Get a list of all fighters after 2010
    all_fighters = pd.Series(list(set(fight_stats[fight_stats["event"].isin(post_2010_events)]["fighter"])))
    
    ## AB: commented this out as it doesn't seem to matter if some missing. The tott data is what's used
    # # Find which fighters are missing
    # missing_fighter_details = all_fighters[~all_fighters.isin(fighter_details['fighter'].to_list())].to_list()
    
    # if (len(missing_fighter_details) > 0):
    #     print(f"Error: Missing fighters from fighter_details data: {missing_fighter_details}")
    # pdb.set_trace()
    # Find which fighters are missing from tott data
    missing_fighter_tott = all_fighters[~all_fighters.isin(fighter_tott["fighter"].to_list())].to_list()
    if (len(missing_fighter_tott) > 0):
        print(f"Error: The following fighters are missing from the fighter_tott data: {missing_fighter_tott}")
        if (len(missing_fighter_tott) > 6):
            all_checks_passed = False
    
    # Check where we have missing Betfair odds
    merged_betfair_odds = cleaned_dict["fight_stats_agg"][["event", "bout", "fighter", "date"]].merge(
        cleaned_dict["betfair_odds"][["event", "bout", "fighter", "betfair_exch_odds"]],
        on=["event", "bout", "fighter"],
        how="left"
    )
    
    # Check how many fighters after 2017 have missing betfair odds
    merged_betfair_odds = merged_betfair_odds[merged_betfair_odds["date"] > config["min_training_date"]]
    # Find the percentage missing where it is a UFC event
    percent_missing_betfair = merged_betfair_odds[merged_betfair_odds["event"].str[0:3] == "ufc"]["betfair_exch_odds"].isna().mean() * 100
    
    print(f"There are {percent_missing_betfair} % of fighters with missing Betfair exchange odds in UFC events since {config['min_training_date']}")
    
    # Get the most missing fighters
    missing_betfair_fighter_counts = merged_betfair_odds[merged_betfair_odds["betfair_exch_odds"].isna()]["fighter"].value_counts()
    
    print("The most missing fighters from Betfair exchange odds are:")
    print(missing_betfair_fighter_counts.head(20))

    # te = merged_betfair_odds[merged_betfair_odds['betfair_exch_odds'].isna()]
    
    # Check how many upcoming bouts have missing odds
    merged_upcoming = cleaned_dict["upcoming_fight_details"].merge(cleaned_dict["betfair_upcoming_odds"], on=["event", "bout"], how="left")
    
    missing_percent_upcoming = merged_upcoming["betfair_exch_odds"].isna().mean() * 100
    
    print(f"There are {missing_percent_upcoming} % missing Betfair odds from upcoming fights")
    # if missing_percent_upcoming > 50:
    #     all_checks_passed = False
    
    return all_checks_passed

# This function cleans infinite values and large numbers

def clean_inf_and_large(data):
    
    data = data.infer_objects(copy=False)  # Ensures data types remain unchanged
    data.replace([np.inf, -np.inf], [99999, -99999], inplace=True)
    
    # Replace large float values with 99999
    data[data.select_dtypes(include=[np.float64]) > 99999] = 99999
    data[data.select_dtypes(include=[np.float64]) < -99999] = -99999
    
    return data

def clean_target_column(target_column):
    """
    Cleans the target column from a text entry to a categorised, allowing consistent formatting across function

    Args:
        target_column (series): the input column
    
    Returns
        A series to be input as the new column

    """
    # pdb.set_trace()
    
    # Recode anything that isn't Wins or losses as other
    cleaned_target = np.where(target_column.isna(), np.nan, np.where((target_column == "D/D") | (target_column == "NC/NC"), "OTH", target_column))
    
    # Convert the target into a labelled column ready for multiclass modelling
    oe = OrdinalEncoder(categories=[["L/W", "OTH", "W/L"]])
    cleaned_target = oe.fit_transform(np.array(cleaned_target).reshape(-1, 1)).astype(int).ravel()
    
    # # Stop if the classes aren't as expected. These are hardcoded in other functions so need to be in this order
    # stop_if_not(le.classes_.tolist() == ['L/W', 'OTH', 'W/L'], "Target class check")

    return cleaned_target

# This function takes the final dataset (after processing) and logs some key metrics so they can be tracked with each update

def log_final_data_stats(df, log_csv_path, config):
    
    # ## For debug
    # df = forecasting_data.copy()
    # log_csv_path = 
    
    # pdb.set_trace()
    
    # Define which key columns we want to track NAs for
    # For all these cols, don't need to put fighter1_ fighter2_ prefix, 'diff' as they will be added later
    na_cols = ["open_odds", "best_closing_odds", "mean_closing_odds", "betfair_exch_odds", "start_elo", "reach", "hist_times_knocked_out", "hist_round3_body_landed_against_per_minute", "hist_last_3_fights_round1_distance_accuracy_percent", "hist_wins_per_fight", "hist_opp_stance_round1_leg_accuracy_percent_against", "hist_opp_type_allrounds_sig.str._landed_against",  "hist_last_3_fights_round1_distance_bayes_accuracy_percent", "hist_opp_stance_round1_kd_per_minute_bayes_adjmean", "hist_last_3_fights_round1_body_landed_against_per_minute_bayes_adjsum", "date_minus_dob", "subs_per_attempt"]
    
    # Remove any cols to track if the col isn't in the data frame
    na_cols = [col for col in na_cols if ("fighter1_" + col) in df.columns]

    # Start a new data frame for the final data stats
    results_df = pd.DataFrame({"Date": [datetime.today()]}, columns=["Date"])

    for fighter in ["fighter1_", "fighter2_"]:  
        
        # Choose only fights that are after validation checks date, and where we have a fight history to avoid NAs
        test_df = df[(df["date"] >= config["start_date_validation_checks"]) & (df[fighter + "hist_fights"] > 0)][[fighter + col for col in na_cols]]
    
        # Calculate the missing values
        missing_values_percentage = test_df.isna().sum() / len(test_df) * 100
        # Calculate column means (ignoring NaNs)
        column_means = test_df.mean(skipna=True)
        
        # Convert the result into a DataFrame and transpose
        results_df = pd.concat([
                results_df,
                pd.DataFrame([missing_values_percentage.tolist() + column_means.tolist()], 
                             columns=[col + "_NA_Percent" for col in missing_values_percentage.index.tolist()] + [col + "_Mean" for col in column_means.index.tolist()])
            ],
            axis=1)

    # Add this onto the existing csv
    existing_results = pd.read_csv(log_csv_path)
    
    results_df = pd.concat([results_df, existing_results])
    
    # Now overwrite the results
    results_df.to_csv(log_csv_path, index=False)

    return None

# This function applies filters to the data for records we aren't interested in modelling and testing.

def standard_data_filter(df, include_upcoming=False, keep_rows_with_flag=False):

    # pdb.set_trace()    

    # Create an empty flag to exception reasons
    df["notformodel_exclusion_reason"] = ""
    
    # Remove any unrealistic odds
    df.loc[(df["fighter1_true_closing_odds"] >= 15) | (df["fighter2_true_closing_odds"] >= 15), "notformodel_exclusion_reason"] = "Odds out of range"
    
    df.loc[(df["fighter1_hist_fights"] <= 2) | (df["fighter2_hist_fights"] <= 2), "notformodel_exclusion_reason"] = "Fights < 3"

    # Find anyone with missing true odds
    df.loc[(df["fighter1_true_closing_odds"].isna()) | (df["fighter2_true_closing_odds"].isna()), "notformodel_exclusion_reason"] = "Missing odds"

    df.loc[(df["womens"] == 1), "notformodel_exclusion_reason"] = "Womens"

    # Drop draws and no contests. A bet on one is voided so they say nothing about whether the model
    # makes money, and the model gives them so little probability that they dominate the logloss.
    # Upcoming fights have a null outcome rather than one of these, so they are not caught by this
    df.loc[df["outcome"].isin(["D/D", "NC/NC"]), "notformodel_exclusion_reason"] = "Draw or no contest"

    if not keep_rows_with_flag:
        df = df[df["notformodel_exclusion_reason"] == ""].drop(columns=["notformodel_exclusion_reason"]).copy()

    # filtered_df = df[(df['womens'] == 0) & 
    #                  (df['fighter1_hist_fights'] > 2) &
    #                  (df['fighter2_hist_fights'] > 2)]
    # # Remove any data were odds are missing
    # filtered_df = filtered_df[(~filtered_df['fighter1_true_closing_odds'].isna()) & (~filtered_df['fighter2_true_closing_odds'].isna())]
    
    # # Remove any unrealistic odds
    # filtered_df = filtered_df[(filtered_df['fighter1_true_closing_odds'] < 15) & (filtered_df['fighter2_true_closing_odds'] < 15)]
    
    if not include_upcoming:
        df = df[(~df["outcome"].isna())]
        
    return df

def undo_clean_inf(df, threshold=99999):
    """
    Replaces all values >= threshold with np.inf and all values <= -threshold with -np.inf.
    Operates directly on numeric columns for maximum speed.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        threshold (int, optional): The threshold value. Defaults to 99999.

    Returns:
        pd.DataFrame: A modified DataFrame with replacements.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns
    
    # Ignore dates
    numeric_cols = [col for col in numeric_cols if ("date" not in col and "dob" not in col) or "diff" in col]
    
    arr = df[numeric_cols].values  # Convert to NumPy array for speed

    # Vectorized replacement
    arr[arr >= threshold] = np.inf
    arr[arr <= -threshold] = -np.inf

    df[numeric_cols] = arr  # Assign back to DataFrame
    return df

def update_data_summary(results_csv_path, data_frames, summary_date, save_output=False):
    """
    Updates the results dataframe with summary statistics of the provided dataframes.
    Each date will be a column with the summary for each dataframe.

    Parameters:
    - results_df (pd.DataFrame): The dataframe where the summary will be added.
    - data_frames (dict): A dictionary of dataframe names as keys and the dataframes as values.
    - summary_dates (list): A list of dates for each summary to be logged. Example: ['2024-12-09', '2024-12-10']

    Returns:
    - pd.DataFrame: The updated results dataframe with new summary columns.
    """
    
    # pdb.set_trace()
    
    results_df = pd.read_csv(results_csv_path, index_col=0)
    
    # Create an empty list to store the summary data for each dataframe
    summary_data = []
    
    for df_name, df in data_frames.items():
        # Get the number of rows and columns
        num_rows, num_cols = df.shape
        
        # Count the number of NaNs in each column
        na_count = df.isna().sum()
        
        # Create a dictionary for the summary of this dataframe
        summary_row = {
            "date": summary_date,
            "dataframe": df_name,
            "rows": num_rows,
            "columns": num_cols,
            **na_count.to_dict()  # Add NaN counts as columns
        }
        
        # Append the summary row to the summary_data list
        summary_data.append(summary_row)
    
    # Convert the summary data to a DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Ensure the dates all correct
    summary_df["date"] = summary_date
    
    # Append the new summary to the results_df
    updated_results_df = pd.concat([results_df, summary_df], ignore_index=True)

    # Save to file before outputting
    if save_output:
        updated_results_df.to_csv(results_csv_path, index=True)

    return updated_results_df
