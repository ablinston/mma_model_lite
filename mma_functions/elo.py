from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import mlflow
import numpy as np
import pandas as pd
import pdb
import polars as pl
from scipy.stats import hmean
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
import xgboost as xgb

from mma_functions import dataset, modelling, tracking, utils
from mma_functions.config import config

def add_start_elo_to_fight_results(fight_results, fight_stats, config, elo_config):
    
    """
    This function takes the fight results data and adds the start ELO to all the bouts
    
    """
    # pdb.set_trace()
    
    # Get the starting data
    start_elo, elo_fight_data = prep_stats_for_elo_calc(fight_results, fight_stats, config, start_date="2000-01-01", remove_nas=False)

    # Dderive the ELO for the params in the config
    elo_data, elo_fight_data_welo = derive_start_elos(start_elo, elo_fight_data, elo_config)

    # Create two versions to be added onto the results data
    fighter1_elo = elo_data.to_pandas().drop(["fight_number"], axis=1).rename(columns={"fighter": "fighter1", "start_elo": "fighter1_start_elo"}).sort_values(["date"]).reset_index(drop=True)
    fighter2_elo = elo_data.to_pandas().drop(["fight_number"], axis=1).rename(columns={"fighter": "fighter2", "start_elo": "fighter2_start_elo"}).sort_values(["date"]).reset_index(drop=True)

    # Merge this onto the fight results data, because this is a merge to the nearest date before or on the fight date. Can't merged elo_fight_data because draws etc were filtered out.

    fight_results = fight_results.sort_values(["date"]).reset_index(drop=True)

    fight_results = pd.merge_asof(
        fight_results,
        fighter1_elo,
        on="date",
        by="fighter1",
        direction="backward"
    )
    fight_results = pd.merge_asof(
        fight_results,
        fighter2_elo,
        on="date",
        by="fighter2",
        direction="backward"
    )

    return fight_results
            

# This function takes a data frame with fight stats and a start ELO for fighters and
# computes the ELOs at the end of the fight

# NB: This function's input must be a polars frame and it returns a new frame with the ELO columns added

# The standard ELO can be calculated by settings all constants except C and K to 0

def calculate_end_elo(pl_frame, constants):

    # pdb.set_trace()

    ## Debug Kwargs

    # C = 400  # Elo rating scaling constant
    # K = 30   # K-factor (rate at which the rating adjusts)

    # # Coefficients for fight metrics differences
    # constant_control_proportion_diff = 0.5
    # constant_sig_strikes_landed_per_minute_diff = 0.3
    # constant_kds_diff = 0.7
    # constant_submissions_diff = 0.6
    # constant_knockouts_diff = 0.4
    # constant_unanimous_decs_diff = 0.2
    # constant_split_decs_diff = 0.1

    # Calculate prerequisites for the start ELO formula
    pl_frame = pl_frame.with_columns(
        fighter1_S=pl.when(pl.col("outcome") == "W/L").then(1).otherwise(0),
        fighter1_E=(10 ** (pl.col("fighter1_start_elo") / constants["C"])) /
                     ((10 ** (pl.col("fighter1_start_elo") / constants["C"])) +
                      (10 ** (pl.col("fighter2_start_elo") / constants["C"])))
    )

    # Calculate fighter1_D, a custom metric for the ELO
    pl_frame = pl_frame.with_columns(
        fighter1_D=(
            constants["constant_control_proportion_diff"] * pl.col("control_proportion_diff") +
            constants["constant_sig_strikes_landed_per_minute_diff"] * pl.col("sig_strikes_landed_per_minute_diff") +
            constants["constant_kds_diff"] * pl.col("kds_diff") +
            constants["constant_submissions_diff"] * pl.col("submissions_diff") +
            constants["constant_knockouts_diff"] * pl.col("knockouts_diff") +
            # constants['constant_split_decs_diff'] * pl.col('split_decs_diff') +
            constants["constant_unanimous_decs_diff"] * pl.col("unanimous_decs_diff") +
            constants["constant_inverse_fight_length_proportion"] * pl.col("inverse_fight_length_proportion")
        )
    )

    # Convert the D stat for the case when fighter 2 wins. If the D stat is positive it means fighter1 was dominant, and fighter 1 will get a boost to their positive points and fighter 2 will get an increase to the negative points by using fighter1_D as well. But if fighter 2 wins and dominates, fighter1_D will be negative and cause fighter 2 ELO to decline (will be a +ve * negative D). So it needs to be -fighter1_D if fighter 2 won.
    pl_frame = pl_frame.with_columns(
        dominance_stat=pl.when(pl.col("outcome") == "L/W").then(-pl.col("fighter1_D")).otherwise(pl.col("fighter1_D")))

    # Calculate fighter1_end_elo and fighter2_end_elo
    # Shrinkage is applied within the calculation directly
    pl_frame = pl_frame.with_columns(
        fighter1_end_elo=pl.col("fighter1_start_elo") + (constants["starting_elo"] - pl.col("fighter1_start_elo")) * constants["constants_shrinkage_factor"] + (1 + (constants["K_factor"] / pl.col("fighter1_fight_number"))) * constants["K"] * (pl.col("fighter1_S") - pl.col("fighter1_E")) * (1 + pl.col("dominance_stat")),
        fighter2_end_elo=pl.col("fighter2_start_elo") + (constants["starting_elo"] - pl.col("fighter2_start_elo")) * constants["constants_shrinkage_factor"] + (1 + (constants["K_factor"] / pl.col("fighter2_fight_number"))) * constants["K"] *  ((1 - pl.col("fighter1_S")) - (1 - pl.col("fighter1_E"))) * (1 + pl.col("dominance_stat"))
    )

    # Overwrite where the outcome wasn't W/L or L/W so that the ELO remains unchanged
    pl_frame = pl_frame.with_columns(
        fighter1_end_elo=pl.when(pl.col("outcome") == "OTH").then(pl.col("fighter1_start_elo")).otherwise(pl.col("fighter1_end_elo")),
        fighter2_end_elo=pl.when(pl.col("outcome") == "OTH").then(pl.col("fighter2_start_elo")).otherwise(pl.col("fighter2_end_elo")))

    return pl_frame

# This function takes a data frame of fighters and dates along with a data frame of fight
# results and stats and calculates

# This function requires that all pre-requisite stats for the elos have been derived using the function
# prep_stats_for_elo_calc()

def derive_start_elos(elo_df, fight_stats_df, elo_constants):

    # pdb.set_trace()

    merged_data = fight_stats_df.clone()
    start_elo = elo_df.clone()

    # Do a check for NAs in the stats used for ELO
    na_count = merged_data["sig_strikes_landed_per_minute_diff"].is_null().sum()
    if na_count > 0:
        print("Error: NAs found in stats used for calculating ELO")
        return 1, 1

    # Convert the constants to numeric
    elo_constants = utils.dict_to_numeric(elo_constants)

    # Calculate ELO for fighters first fights
    merged_data = calculate_end_elo(
        merged_data,
        constants=elo_constants)

    # # Do a check on the ELO for the first fights. All should be True
    # # Check 1: Max end elo for fighter1 where outcome is 'L/W' is less than start elo

    # # Check 1: Max end elo for fighter1 where outcome is 'L/W' is less than start elo
    # check1 = merged_data.filter(pl.col('outcome') == 'L/W')['fighter1_end_elo'].max() < config['start_elo']
    # # Check 2: Min end elo for fighter1 where outcome is 'W/L' is greater than start elo
    # check2 = merged_data.filter(pl.col('outcome') == 'W/L')['fighter1_end_elo'].min() > config['start_elo']
    # # Check 3: Max end elo for fighter2 where outcome is 'W/L' is less than start elo
    # check3 = merged_data.filter(pl.col('outcome') == 'W/L')['fighter2_end_elo'].max() < config['start_elo']
    # # Check 4: Min end elo for fighter2 where outcome is 'L/W' is greater than start elo
    # check4 = merged_data.filter(pl.col('outcome') == 'L/W')['fighter2_end_elo'].min() > config['start_elo']

    # if not check1 == check2 == check3 == check4 == True:
    #     print("Checks failed. Initial ELO for this configuration does not rise and fall as expected from fight 1")
    #     return 1, 1

    # Now loop through the rest of the fights in a career and fill in the ELO
    for fight in range(2, 500): #start_elo['fight_number'].max() + 1):

        # First, match on the end ELO from the results data onto the next fight as the starting ELO
        join_df = merged_data.select(["date", "fighter1", "fighter1_end_elo"]).rename({"fighter1": "fighter"})
        new_start_elo = start_elo.join(join_df, on=["date", "fighter"], how="left", maintain_order="left")

        # Repeat for fighter2
        join_df = merged_data.select(["date", "fighter2", "fighter2_end_elo"]).rename({"fighter2": "fighter"})
        new_start_elo = new_start_elo.join(join_df, on=["date", "fighter"], how="left", maintain_order="left")

        # Now combine the ELOs regardless of whether fighter 1 or 2
        new_start_elo = new_start_elo.with_columns(
            end_elo=pl.when(pl.col("fighter1_end_elo").is_null())
                        .then(pl.col("fighter2_end_elo"))
                        .otherwise(pl.col("fighter1_end_elo")))

        # Now shift the ending elo to the starting elo of the next fight. The frame is in
        # date order within a fighter, so a plain lag gives the previous fight
        new_start_elo = new_start_elo.with_columns(
            next_start_elo=pl.col("end_elo").shift().over("fighter"))

        # Update start_elo where the fighter has a previous fight to carry the ELO over from
        new_start_elo = new_start_elo.with_columns(
            start_elo=pl.when((pl.col("fight_number") > 1) & pl.col("next_start_elo").is_not_null())
                          .then(pl.col("next_start_elo"))
                          .otherwise(pl.col("start_elo")))

        # Drop columns no longer needed
        new_start_elo = new_start_elo.select(["date", "fighter", "fight_number", "start_elo"])

        ## Now merge the start ELOs onto the merged data

        # Drop the fighter ELO columns if they exist on the data already
        try:
            new_merged_data = merged_data.drop(["fighter1_start_elo", "fighter2_start_elo"])
        except:
            continue

        # Now join on the updated fighter start ELOs
        join_df = new_start_elo.select(["date", "fighter", "start_elo"]).rename({"fighter": "fighter1", "start_elo": "fighter1_start_elo"})
        new_merged_data = new_merged_data.join(join_df, on=["date", "fighter1"], how="left", maintain_order="left")

        # Repeat for fighter2
        join_df = new_start_elo.select(["date", "fighter", "start_elo"]).rename({"fighter": "fighter2", "start_elo": "fighter2_start_elo"})
        new_merged_data = new_merged_data.join(join_df, on=["date", "fighter2"], how="left", maintain_order="left")

        # Now calculate the end ELOs again
        new_merged_data = calculate_end_elo(
            new_merged_data,
            constants=elo_constants)

        # Now repeat the loop until all ELOs have been calculated
        # Clean up memory for efficiency
        start_elo = new_start_elo
        merged_data = new_merged_data
        del[new_start_elo, new_merged_data, join_df]
        gc.collect()

        # Check how many start ELOs are missing. If none, finish the loop
        na_count = start_elo["start_elo"].is_null().sum()
        if na_count > 0:
            pass
            # print(f"{na_count} ELOs missing. Continuing loop {fight}")
        else:
            return start_elo, merged_data


    print("ELOs failed to compute after 500 rounds of iteration")
    pdb.set_trace()
    return 1, 1

# This function is used to predict fight outcomes based only on the elo and assess how good it is at
# predicting the fight outcomes on it's own. It takes the highest ELO as the predicted winner.

# It relies on the following objects being available in memory:
    # - start_elo, merged_data: these are the standard outputs of the prep_stats_for_elo_calc() function
  

def objective_cv_elo_basic_mlflow(params):

    # pdb.set_trace()
    print("Params received:", params)    
    
    # Check for duplicates
    if tracking.is_duplicate_run(params):
        
        # If the run existing, find what the loss was and return it without logging in mlflow
        existing_metric, all_params = tracking.get_existing_metric_and_params(params, "Logloss")
        
        print(f"Duplicate run found, skipping logging. Logloss: {existing_metric:.4f}")
        return {"loss": existing_metric, "params": all_params, "status": STATUS_OK}
        
    else:
    
        try:
        
            # Begin new model in MLflow
            with mlflow.start_run():
                
                # Derive the ELO for the selected constants
                elo_data, fight_data = derive_start_elos(start_elo, merged_data, params)
            
                # Create the training dataset. This only contains the outcome and ELO variables. Also filters for the training start date
                train = fight_data.to_pandas()
                train = train[train["date"] >= config["min_training_date"]][["outcome", "fighter1_start_elo", "fighter2_start_elo"]]
        
                mlflow.set_tag("Description", "Win probability from the ELO difference, scored on logloss")
        
                # Log the training parameters
                mlflow.log_param("model", "Basic")
                mlflow.log_params(params)
                mlflow.log_param("Fighter 1 Max ELO", train["fighter1_start_elo"].max())
                mlflow.log_param("Fighter 1 Min ELO", train["fighter1_start_elo"].min())
                        
                # Only fights with a winner can be scored against a win probability
                train = train[train["outcome"].isin(["W/L", "L/W"])]
                
                # The probability fighter 1 wins. This is the same logistic form the rating update
                # uses for the expected score, so it has to use the same C, otherwise the ratings and
                # the probabilities would be on different scales
                fighter1_win_prob = 1 / (1 + 10 ** ((train["fighter2_start_elo"] - train["fighter1_start_elo"]) / params["C"]))
                
                # The outcome as a 1 where fighter 1 won
                fighter1_won = np.where(train["outcome"] == "W/L", 1, 0)
                
                elo_logloss = log_loss(fighter1_won, fighter1_win_prob, labels=[0, 1])
                
                # Keep the accuracies as well so runs can still be compared with the older ones
                y_pred = np.where(train["fighter1_start_elo"] >= train["fighter2_start_elo"], "W/L", "L/W")
                correct_binary = y_pred == train["outcome"]
            
                # Now calculate the accuracies
                wl_accuracy = np.mean(correct_binary[y_pred == "W/L"])
                lw_accuracy = np.mean(correct_binary[y_pred == "L/W"])
        
                # Log the outcome in MLflow
                mlflow.log_metric("Logloss", elo_logloss)
                mlflow.log_metric("W/L accuracy", wl_accuracy)
                mlflow.log_metric("L/W accuracy", lw_accuracy)
                mlflow.log_metric("H.Mean of two accuracies", hmean([wl_accuracy, lw_accuracy]))
                mlflow.log_metric("Overall accuracy", np.mean(correct_binary))
         
            # Return whatever metric we wish to optimise for the model
            return {"loss": elo_logloss, "params": params, "status": STATUS_OK} 
        
        except Exception as e:
            print("Error encountered. Model fit abandoned")
            return {"loss": float("inf"), "params": params, "status": STATUS_FAIL, "error": str(e)}

# This function is used to train decision trees on the elo and assess how good it is at
# predicting the fight outcomes on it's own

# It relies on the following objects being available in memory:
    # - start_elo, merged_data: these are the standard outputs of the prep_stats_for_elo_calc() function
    # elo_n_folds to set the number of folds
    

def objective_cv_elo_dtree_mlflow(params):

    # pdb.set_trace()
    print("Params received:", params)    
    
    # Check for duplicates
    if tracking.is_duplicate_run(params):
        
        # If the run existing, find what the loss was and return it without logging in mlflow
        existing_metric, all_params = tracking.get_existing_metric_and_params(params, "Avg of two accuracies")
        
        print(f"Duplicate run found, skipping logging. Accuracy: {existing_metric:.4f}")
        return {"loss": 1 - existing_metric, "params": all_params, "status": STATUS_OK}
        
    else:
    
        try:
        
            # Dderive the ELO for the selected constants
            elo_data, fight_data = derive_start_elos(start_elo, merged_data, params)
        
            # Create the training dataset. This only contains the outcome and ELO variables. Also filters for the training start date
            train = fight_data.to_pandas()
            train = train[train["date"] >= config["min_training_date"]][["outcome", "fighter1_start_elo", "fighter2_start_elo"]]
            train["elo_diff"] = train["fighter1_start_elo"] - train["fighter2_start_elo"]
                
            # Create n folds of the training data
            kf = KFold(n_splits=elo_n_folds, shuffle=True, random_state=config["seed_all"])
        
            # Create lists that will be used to store models and test datasets
            wl_accuracies = []
            lw_accuracies = []
            accuracies = []
        
            # Print out progress
            print("Training CV models...")
            
            # Convert the target into a labelled column ready for multiclass modelling
            le = LabelEncoder()
            train["outcome"] = le.fit_transform(train["outcome"])
            
            # Split into train and test
            X_train, X_test, y_train, y_test = train_test_split(
                train.drop(columns=["outcome"]),
                train["outcome"],
                test_size=0.1,
                random_state=config["seed_all"]
            )
            
            # Precompute DMatrix objects using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=config["num_threads"]) as executor:
                futures = {executor.submit(modelling.create_dmatrix, X_train, y_train, train_idx, test_idx): (train_idx, test_idx) for train_idx, test_idx in kf.split(X_train)}
                dmatrix_data = [future.result() for future in as_completed(futures)]
            
            # Begin new model in MLflow
            with mlflow.start_run():
        
                mlflow.set_tag("Description", "Cross-validated Decision Tree")
        
                # Log the training parameters
                mlflow.log_param("model", "Decision Tree")
                mlflow.log_params(params)
                mlflow.log_param("Fighter 1 Max ELO", train["fighter1_start_elo"].max())
                mlflow.log_param("Fighter 1 Min ELO", train["fighter1_start_elo"].min())
                
                # Loop through precomputed DMatrix objects for each fold
                for train_dmatrix, test_dmatrix in dmatrix_data:
        
                    # # For debugging
                    # train_dmatrix = xgb.DMatrix(X_train, label = y_train)
                    # test_dmatrix = xgb.DMatrix(X_test, label = y_test)
                    
                    # Train the model with early stopping
                    model = xgb.train(params, train_dmatrix,
                                      num_boost_round=params["num_iterations"],
                                      evals=[(test_dmatrix, "eval")],
                                      early_stopping_rounds=params["early_stopping_rounds"],
                                      verbose_eval=True)
                        
                    # Make predictions on the test set
                    y_pred_prob = model.predict(test_dmatrix)
                    
                    # Get the outcomes of the test data
                    y_test_fold = test_dmatrix.get_label().astype(int)
        
                    # Convert actual results to binarized format
                    y_test_bin = label_binarize(y_test_fold, classes=np.array([0, 1, 2]))
                    y_pred_bin = label_binarize(y_pred_prob, classes=np.array([0, 1, 2]))
        
                    # Calculate the accuracy of wins and losses. Check the classes of the target column. Was 1 = L/W and 3 = W/L (should be alphabetical)
                    wl_accuracy = (y_test_bin[:, 2] == y_pred_bin[:, 2]).mean()
                    lw_accuracy = (y_test_bin[:, 0] == y_pred_bin[:, 0]).mean()
                    lw_accuracies.append(lw_accuracy)
                    wl_accuracies.append(wl_accuracy)
                    
                    # Calculate the AUC as the average of the W/L and L/W
                    accuracies.append(np.mean([lw_accuracy, wl_accuracy]))
        
                    # Clean up memory
                    del model, train_dmatrix, test_dmatrix, y_pred_prob, wl_accuracy, lw_accuracy
                    gc.collect()
                    
                # Now all folds have been calculated, summarise the metrics
        
                # Log the outcome in MLflow
                mlflow.log_metric("W/L accuracy", np.mean(wl_accuracies))
                mlflow.log_metric("L/W accuracy", np.mean(lw_accuracies))
                mlflow.log_metric("Avg of two accuracies", np.mean(accuracies))
         
            # Return whatever metric we wish to optimise for the model
            return {"loss": 1 - np.mean(accuracies), "params": params, "status": STATUS_OK} 
        
        except:
            print("Error encountered. Model fit abandoned")
            return {"loss": float("inf"), "params": params, "status": STATUS_FAIL, "error": str(e)}

# This function prepares a stats dataset for the elo calcs. It takes in pandas data frames,
# the cleaned fight results and fights stats data,
# then outputs two polars frames because that will be used for the ELO calc.

# The first output is a frame with each fighter and the ELO for each fight ready to be populated
# The second output is a frame with each fight and the results/stats

# remove_nas is a setting on whether to remove NA stats from the dataset. Some fights have no stats. If they are not remove, NAs will be filled by 0 so a basic version of ELO will apply.

def prep_stats_for_elo_calc(fight_results_df, fight_stats_df, config, start_date="2000-01-01", remove_nas=False):
    
    # pdb.set_trace()
    
    fight_results = fight_results_df.copy()
    fight_stats = fight_stats_df.copy()
    
    # Recode the target. Don't need to distinguish between D and NC
    fight_results["outcome"] = np.where((fight_results["outcome"] == "D/D") | (fight_results["outcome"] == "NC/NC"), "OTH", fight_results["outcome"])
    
    # Remove any results that are wins or losses by DQ. Just recode as OTH
    fight_results["outcome"] = np.where((fight_results["method"] == "Other"), "OTH", fight_results["outcome"])
    
    # Add the fighter columns needed to the data
    fight_results[["fighter1", "fighter2"]] = fight_results["bout"].str.split("vs\\.", expand=True)

    # Merge the two datasets together to get stats for the fight
    merged_data = dataset.merge_df_both_fighters(
        left_df=fight_results[["event", "bout", "fighter1", "fighter2", "outcome", "method", "weightclass", "date", "time format"]],
        df_list=[
            fight_stats
        ]
    ).reset_index().drop(columns=["index"])

    # # Remove any results that aren't wins or losses by DQ. Remove anything before start_date because of multiple fights per date.
    # # Keep np.nan because these are future events we want ELOs for
    # merged_data = merged_data[(merged_data['method'].isin(['Decision-Unanimous', 'KO/TKO', 'Submission', 'Decision-Split', np.nan, None])) & (merged_data['date'] >= start_date)]

    # Cleanup
    del([fight_results, fight_stats])
    
    # Get rid of any rows where we don't have stats
    if remove_nas:
        # Count NaN values in 'round1_fight_minutes' per event
        na_counts_per_event = merged_data["fighter1_round1_fight_minutes"].isna().groupby(merged_data["event"]).sum().reset_index()
        # Rename columns for clarity
        na_counts_per_event.columns = ["event", "na_count_round1_fight_minutes"]
        na_counts_per_event = na_counts_per_event[na_counts_per_event["na_count_round1_fight_minutes"] > 0]
        if len(na_counts_per_event) > 0:
            print("Removing bouts from the following events where no fight stats available")
            print(na_counts_per_event)
            merged_data = merged_data[~merged_data["fighter1_round1_fight_minutes"].isna()]

    # Remove any duplicates of fighters on dates
    # Sort so we keep records with results first
    merged_data = merged_data.sort_values(["outcome"], kind="stable")
    merged_data = dataset.remove_duplicate_fighter_pairs(merged_data)

    # Now create a data frame that contains only the fighters and the dates they fought. This will be used to track the existing ELO of a fighter on that date
    start_elo = pd.concat([
        merged_data[["fighter1", "date"]].rename(columns={"fighter1": "fighter"}),
        merged_data[["fighter2", "date"]].rename(columns={"fighter2": "fighter"})
        ]).sort_values(["fighter", "date"]).reset_index(drop=True)

    # # Remove any remaining duplicates # Shouldn't be needed
    # start_elo = start_elo.drop_duplicates(['date', 'fighter'])

    # Add the fight number for each fighter
    start_elo["fight_number"] = start_elo.groupby("fighter").cumcount() + 1

    # Compute the starting ELO for each fighter
    start_elo["start_elo"] = np.nan
    start_elo.loc[start_elo["fight_number"] == 1,"start_elo"] = config["start_elo"]

    # Now merge these onto the merged fight data to being calculating the ELO at the end of the first fights
    merged_data = merged_data.merge(
        start_elo[["fighter", "date", "start_elo", "fight_number"]].rename(columns={"start_elo": "fighter1_start_elo", "fight_number": "fighter1_fight_number"}),
        left_on=["fighter1", "date"], 
        right_on=["fighter", "date"],
        how="left"
    ).drop(columns="fighter").merge(
        start_elo[["fighter", "date", "start_elo", "fight_number"]].rename(columns={"start_elo": "fighter2_start_elo", "fight_number": "fighter2_fight_number"}),
        left_on=["fighter2", "date"], 
        right_on=["fighter", "date"],
        how="left"
    ).drop(columns="fighter")

    ##############################################################################
    # Create the columns that will be used in the ELO calculation

    merged_data["control_proportion_diff"] = (merged_data["fighter1_allrounds_ctrl_minutes"] / merged_data["fighter1_allrounds_fight_minutes"]) - (merged_data["fighter2_allrounds_ctrl_minutes"] / merged_data["fighter2_allrounds_fight_minutes"])

    merged_data["sig_strikes_landed_per_minute_diff"] = np.where(
        merged_data["fighter1_allrounds_sig.str._landed"] > merged_data["fighter2_allrounds_sig.str._landed"],
        1,
        np.where(
            merged_data["fighter1_allrounds_sig.str._landed"] < merged_data["fighter2_allrounds_sig.str._landed"],
            -1,
            0)
    )

    # merged_data['strikes_landed_per_minute_head_diff'] = (merged_data['fighter1_allrounds_head_landed'] / merged_data['fighter1_allrounds_fight_minutes']) - (merged_data['fighter2_allrounds_head_landed'] / merged_data['fighter2_allrounds_fight_minutes'])

    # merged_data['strikes_landed_per_minute_body_diff'] = (merged_data['fighter1_allrounds_body_landed'] / merged_data['fighter1_allrounds_fight_minutes']) - (merged_data['fighter2_allrounds_body_landed'] / merged_data['fighter2_allrounds_fight_minutes'])

    # merged_data['strikes_landed_per_minute_leg_diff'] = (merged_data['fighter1_allrounds_leg_landed'] / merged_data['fighter1_allrounds_fight_minutes']) - (merged_data['fighter2_allrounds_leg_landed'] / merged_data['fighter2_allrounds_fight_minutes'])

    # merged_data['strikes_landed_per_minute_ground_diff'] = (merged_data['fighter1_allrounds_ground_landed'] / merged_data['fighter1_allrounds_fight_minutes']) - (merged_data['fighter2_allrounds_ground_landed'] / merged_data['fighter2_allrounds_fight_minutes'])

    # merged_data['strikes_landed_per_minute_clinch_diff'] = (merged_data['fighter1_allrounds_clinch_landed'] / merged_data['fighter1_allrounds_fight_minutes']) - (merged_data['fighter2_allrounds_clinch_landed'] / merged_data['fighter2_allrounds_fight_minutes'])

    merged_data["kds_diff"] = np.where(
        merged_data["fighter1_allrounds_kd"] > merged_data["fighter2_allrounds_kd"],
        1,
        np.where(
            merged_data["fighter1_allrounds_kd"] < merged_data["fighter2_allrounds_kd"],
            -1,
            0)
    )

    merged_data["submissions_diff"] = np.where(merged_data["method"] == "Submission",
                                               np.where(merged_data["outcome"] == "W/L", 1, -1),
                                               0)

    merged_data["knockouts_diff"] = np.where(merged_data["method"] == "KO/TKO",
                                               np.where(merged_data["outcome"] == "W/L", 1, -1),
                                               0)

    merged_data["unanimous_decs_diff"] = np.where(merged_data["method"] == "Decision-Unanimous",
                                               np.where(merged_data["outcome"] == "W/L", 1, -1),
                                               0)

    merged_data["split_decs_diff"] = np.where(merged_data["method"] == "Decision-Split",
                                               np.where(merged_data["outcome"] == "W/L", 1, -1),
                                               0)
    
    # Add the fight number for each fighter
    merged_data.sort_values("date", inplace=True)
    # merged_data['fighter1_fight_number'] = merged_data.groupby('fighter1').cumcount() + 1
    # merged_data['fighter2_fight_number'] = merged_data.groupby('fighter2').cumcount() + 1
    
    # Calculate the inverse of the fight length proportion (doing all as 25 minutes). Positive for a win, negative for loss
    merged_data["fight_end_in_round"] = np.ceil(merged_data["fighter1_allrounds_fight_minutes"] / 5)
    merged_data["inverse_fight_length_proportion"] = np.where(
        merged_data["outcome"] == "W/L", 
        1 / merged_data["fight_end_in_round"],
        - 1 / merged_data["fight_end_in_round"])
    
    # pdb.set_trace()
    # Fill any NAs with 0 so functions run
    merged_data = merged_data.fillna({
        col: 0 if merged_data[col].dtype in ["float64", "int64"] else ""
        for col in [colu  for colu in merged_data.columns if colu not in ["fighter1_start_elo", "fighter2_start_elo"]]
    })
    
    # Convert to polars for speed. start_elo goes in date then fighter order because the
    # ELO calc relies on each fighter's rows being in date order
    merged_data = pl.from_pandas(merged_data)
    start_elo = pl.from_pandas(start_elo).select(["date", "fighter", "fight_number", "start_elo"]).sort(["date", "fighter"])

    return start_elo, merged_data
