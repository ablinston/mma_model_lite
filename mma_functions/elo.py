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

    # Squash the dominance into the multiplier rather than using 1 + D directly. A raw 1 + D goes
    # negative once D drops below -1, which flipped the sign of the update and docked the winner
    # while rewarding the loser. 1 + tanh(D) matches 1 + D for small D, is strictly positive for any
    # D, and caps the amplification at twice K
    pl_frame = pl_frame.with_columns(
        dominance_multiplier=1 + pl.col("dominance_stat").tanh())

    # Calculate fighter1_end_elo and fighter2_end_elo
    # Shrinkage is applied within the calculation directly
    pl_frame = pl_frame.with_columns(
        fighter1_end_elo=pl.col("fighter1_start_elo") + (1 + (constants["K_factor"] / pl.col("fighter1_fight_number"))) * constants["K"] * (pl.col("fighter1_S") - pl.col("fighter1_E")) * pl.col("dominance_multiplier"),
        fighter2_end_elo=pl.col("fighter2_start_elo") + (1 + (constants["K_factor"] / pl.col("fighter2_fight_number"))) * constants["K"] *  ((1 - pl.col("fighter1_S")) - (1 - pl.col("fighter1_E"))) * pl.col("dominance_multiplier")
    )

    # Leave the ELO alone unless the fight actually had a winner. A whitelist rather than checking for
    # OTH, because an upcoming fight has a null outcome and used to fall through here and move both
    # ratings as though fighter 2 had won
    pl_frame = pl_frame.with_columns(
        fighter1_end_elo=pl.when(pl.col("outcome").is_in(["W/L", "L/W"])).then(pl.col("fighter1_end_elo")).otherwise(pl.col("fighter1_start_elo")),
        fighter2_end_elo=pl.when(pl.col("outcome").is_in(["W/L", "L/W"])).then(pl.col("fighter2_end_elo")).otherwise(pl.col("fighter2_start_elo")))

    return pl_frame

# This function takes a data frame of fighters and dates along with a data frame of fight
# results and stats and calculates

# This function requires that all pre-requisite stats for the elos have been derived using the function
# prep_stats_for_elo_calc()

def derive_start_elos(elo_df, fight_stats_df, elo_constants):

    merged_data = fight_stats_df.clone()
    start_elo = elo_df.clone()

    # Do a check for NAs in the stats used for ELO
    na_count = merged_data["sig_strikes_landed_per_minute_diff"].is_null().sum()
    if na_count > 0:
        print("Error: NAs found in stats used for calculating ELO")
        return 1, 1

    # Convert the constants to numeric
    elo_constants = utils.dict_to_numeric(elo_constants)

    # Every fight only ever depends on fights that happened before it, so one pass in date order is
    # enough. Walk the fights oldest first holding each fighter's running ELO, and whatever is in the
    # lookup when a fight comes round is that fighter's starting ELO for it. This used to iterate on
    # fight number instead, which moved every fighter forward one career fight per pass and so took as
    # many passes over the whole dataset as the longest career
    ordered = merged_data.with_row_index("elo_row_order").sort("date")

    fighter1 = ordered["fighter1"].to_list()
    fighter2 = ordered["fighter2"].to_list()
    outcome = ordered["outcome"].to_list()
    fighter1_fight_number = ordered["fighter1_fight_number"].to_numpy()
    fighter2_fight_number = ordered["fighter2_fight_number"].to_numpy()

    # The dominance stat doesn't depend on the ELOs, so it can be worked out for every fight up front
    fighter1_D = (
        elo_constants["constant_control_proportion_diff"] * ordered["control_proportion_diff"] +
        elo_constants["constant_sig_strikes_landed_per_minute_diff"] * ordered["sig_strikes_landed_per_minute_diff"] +
        elo_constants["constant_kds_diff"] * ordered["kds_diff"] +
        elo_constants["constant_submissions_diff"] * ordered["submissions_diff"] +
        elo_constants["constant_knockouts_diff"] * ordered["knockouts_diff"] +
        elo_constants["constant_unanimous_decs_diff"] * ordered["unanimous_decs_diff"] +
        elo_constants["constant_inverse_fight_length_proportion"] * ordered["inverse_fight_length_proportion"]
    ).to_numpy()

    starting_elo = elo_constants["starting_elo"]
    C = elo_constants["C"]
    K = elo_constants["K"]
    K_factor = elo_constants["K_factor"]
    half_life = elo_constants["shrinkage_half_life_days"]

    date = ordered["date"].to_list()

    current_elo = {}
    last_fight_date = {}
    fighter1_start = np.empty(len(ordered))
    fighter2_start = np.empty(len(ordered))

    for i in range(len(ordered)):

        elo1 = current_elo.get(fighter1[i], starting_elo)
        elo2 = current_elo.get(fighter2[i], starting_elo)

        # Regress towards the starting ELO for the time out of competition, before the fight rather
        # than after it, so the ELO the model sees is the one a layoff has already been priced into.
        # Decaying by days rather than by fights means someone out for three years is regressed hard
        # and someone fighting every eight weeks is barely touched, which is the way round it should be
        last1 = last_fight_date.get(fighter1[i])
        if last1 is not None:
            elo1 = starting_elo + (elo1 - starting_elo) * 0.5 ** ((date[i] - last1).days / half_life)

        last2 = last_fight_date.get(fighter2[i])
        if last2 is not None:
            elo2 = starting_elo + (elo2 - starting_elo) * 0.5 ** ((date[i] - last2).days / half_life)

        fighter1_start[i] = elo1
        fighter2_start[i] = elo2

        # The clock restarts whatever the result, so this happens even when the ELO doesn't move
        last_fight_date[fighter1[i]] = date[i]
        last_fight_date[fighter2[i]] = date[i]

        # Only a fight with a winner moves the ELO. A draw, a no contest or an upcoming fight leaves
        # both where they are
        if outcome[i] not in ("W/L", "L/W"):
            current_elo[fighter1[i]] = elo1
            current_elo[fighter2[i]] = elo2
            continue

        S = 1.0 if outcome[i] == "W/L" else 0.0
        E = (10 ** (elo1 / C)) / ((10 ** (elo1 / C)) + (10 ** (elo2 / C)))
        dominance_stat = -fighter1_D[i] if outcome[i] == "L/W" else fighter1_D[i]
        dominance_multiplier = 1 + np.tanh(dominance_stat)

        current_elo[fighter1[i]] = elo1 + (1 + (K_factor / fighter1_fight_number[i])) * K * (S - E) * dominance_multiplier
        current_elo[fighter2[i]] = elo2 + (1 + (K_factor / fighter2_fight_number[i])) * K * ((1 - S) - (1 - E)) * dominance_multiplier

    # Put the starting ELOs back in the original row order
    ordered = ordered.with_columns(
        fighter1_start_elo=pl.Series(fighter1_start),
        fighter2_start_elo=pl.Series(fighter2_start))

    merged_data = ordered.sort("elo_row_order").drop("elo_row_order")

    # Run the normal end ELO calculation so every derived column is produced the same way as before
    merged_data = calculate_end_elo(merged_data, constants=elo_constants)

    # Carry the starting ELOs back onto the per fighter frame
    both_sides = pl.concat([
        merged_data.select(["date", pl.col("fighter1").alias("fighter"), pl.col("fighter1_start_elo").alias("new_start_elo")]),
        merged_data.select(["date", pl.col("fighter2").alias("fighter"), pl.col("fighter2_start_elo").alias("new_start_elo")])
    ])

    start_elo = start_elo.join(both_sides, on=["date", "fighter"], how="left", maintain_order="left")

    start_elo = start_elo.with_columns(
        start_elo=pl.when(pl.col("new_start_elo").is_not_null())
                    .then(pl.col("new_start_elo"))
                    .otherwise(pl.col("start_elo")))

    start_elo = start_elo.select(["date", "fighter", "fight_number", "start_elo"])

    return start_elo, merged_data

# This function is used to predict fight outcomes based only on the elo and assess how good it is at
# predicting the fight outcomes on it's own. It takes the highest ELO as the predicted winner.

# start_elo and merged_data are the two outputs of prep_stats_for_elo_calc()
  

def objective_cv_elo_basic_mlflow(params, start_elo, merged_data):

    # Imported here rather than at the top. hyperopt is not in requirements.txt, and pipelines.py
    # imports this module for the main pipeline, which has no need of it
    from hyperopt import STATUS_FAIL, STATUS_OK

    # pdb.set_trace()
    print("Params received:", params)    
    
    # Check for duplicates
    if tracking.is_duplicate_run(params, use_cache=True):
        
        # If the run existing, find what the loss was and return it without logging in mlflow
        existing_metric, all_params = tracking.get_existing_metric_and_params(params, "Logloss", use_cache=True)
        
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
        
                # Log the outcome in MLflow. One call rather than one per metric, because each is a
                # round trip to the server and the trial itself now takes a fraction of a second
                elo_metrics = {
                    "Logloss": elo_logloss,
                    "W/L accuracy": wl_accuracy,
                    "L/W accuracy": lw_accuracy,
                    "H.Mean of two accuracies": hmean([wl_accuracy, lw_accuracy]),
                    "Overall accuracy": np.mean(correct_binary)
                }

                mlflow.log_metrics(elo_metrics)

            # Keep the cache in step so repeating these parameters is caught without asking the server
            tracking.record_run(params, elo_metrics)

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
