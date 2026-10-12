from ablinston_ml import stop_if_not
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hmean
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import label_binarize

from mma_functions import modelling, odds

def calculate_metrics(y_pred_prob, y_test_df, fighter1_odds, fighter2_odds, classes=np.array([0, 1, 2])):
    
    """
    Take a set of predictions and actuals and calculates a series of metrics
    
    Args:
        y_pred_prob (array): an array of probabilities for the predictions
        y_test_df: a series of actuals to test against
        classes (array): the classes of the predicted variable
        
    Returns:
        A dictionary of metrics
    
    """
    # pdb.set_trace()

    y_pred = np.argmax(y_pred_prob, axis=1)

    # Convert the predictions to object in case not already
    y_test_fold = pd.Series(y_test_df.copy(), dtype="int")
    y_pred = pd.Series(y_pred.copy(), dtype="int")
    
    # Convert actual results to binarized format
    y_test_bin = label_binarize(y_test_fold, classes=classes)
    y_pred_bin = label_binarize(y_pred, classes=classes)

    # AUC fails when only one class in observations, so try except
    try:
        # Calculate the AUC of wins and losses. Check the classes of the target column. Was 1 = L/W and 3 = W/L
        wl_auc = roc_auc_score(y_test_bin[:, 2], y_pred_prob[:, 2], labels=classes)
        lw_auc = roc_auc_score(y_test_bin[:, 0], y_pred_prob[:, 0], labels=classes)
    except:
        wl_auc = np.nan
        lw_auc = np.nan

    # Calculate the accuracies
    wl_pred = y_pred_bin[:, 2]
    wl_test = y_test_bin[:, 2]
    lw_pred = y_pred_bin[:, 0]
    lw_test = y_test_bin[:, 0]

    wl_accuracy = np.nan if wl_pred.sum() == 0 else (wl_pred[wl_pred == 1] == wl_test[wl_pred == 1]).sum() / wl_pred.sum()
    lw_accuracy = np.nan if lw_pred.sum() == 0 else (lw_pred[lw_pred == 1] == lw_test[lw_pred == 1]).sum() / lw_pred.sum()

    # **Calculate LogLoss**
    log_loss_result = log_loss(y_test_df, y_pred_prob, labels=classes)

    ev_mse = modelling.ev_loss_function(
        y_pred_prob,
        y_test_df,
        fighter1_odds, 
        fighter2_odds,
        return_loss=True,
        input_probs=True)

    # Return the metrics    
    return {
        "Accuracy": accuracy_score(y_test_df, y_pred),
        "W/L AUC": wl_auc,
        "L/W AUC": lw_auc,
        "W/L Accuracy": wl_accuracy,
        "L/W Accuracy": lw_accuracy,
        "Avg WL only AUC": np.mean([wl_auc, lw_auc]),
        "Hmean WL only Accuracy": hmean([lw_accuracy, wl_accuracy]),
        "Logloss": log_loss_result,
        "EV MSE": ev_mse,
        "Sample Size": len(y_pred)
    }

# This function takes the dataset and calculates the set of metrics based on the odds only

def calculate_metrics_odds(X_test, y_test, classes=np.array([0, 1, 2])):

    # pdb.set_trace()

    # First get the predictions from the ensemble
    y_pred_prob = odds.odds_to_probabilities(X_test["fighter1_backtesting_odds"], X_test["fighter2_backtesting_odds"])

    # # Fill in nas
    # y_pred_prob['L/W'] = y_pred_prob['L/W'].fillna(0.499)
    # y_pred_prob['OTH'] = y_pred_prob['OTH'].fillna(0)
    # y_pred_prob['W/L'] = y_pred_prob['W/L'].fillna(0.501)

    # Convert the probabilities to predictions
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Convert the probabilities into a Numpy array
    y_pred_prob = y_pred_prob.to_numpy()

    # Convert the predictions to object in case not already
    y_test_fold = pd.Series(y_test.copy(), dtype="int")

    # Convert actual results to binarized format
    y_test_bin = label_binarize(y_test_fold, classes=classes)
    y_pred_bin = label_binarize(y_pred, classes=classes)

    # AUC fails when only one class in observations, so try except
    try:
        # Calculate the AUC of wins and losses. Check the classes of the target column. Was 1 = L/W and 3 = W/L
        wl_auc = roc_auc_score(y_test_bin[:, 2], y_pred_prob[:, 2], labels=classes)
        lw_auc = roc_auc_score(y_test_bin[:, 0], y_pred_prob[:, 0], labels=classes)
    except:
        wl_auc = np.nan
        lw_auc = np.nan

    # Calculate the accuracies
    wl_pred = y_pred_bin[:, 2]
    wl_test = y_test_bin[:, 2]
    lw_pred = y_pred_bin[:, 0]
    lw_test = y_test_bin[:, 0]

    wl_accuracy = np.nan if wl_pred.sum() == 0 else (wl_pred[wl_pred == 1] == wl_test[wl_pred == 1]).sum() / wl_pred.sum()
    lw_accuracy = np.nan if lw_pred.sum() == 0 else (lw_pred[lw_pred == 1] == lw_test[lw_pred == 1]).sum() / lw_pred.sum()

    # **Calculate LogLoss**
    log_loss_result = log_loss(y_test, y_pred_prob, labels=classes)    

    # Calculate the MSE of the Expected value
    ev_mse = modelling.ev_loss_function(
        y_pred_prob,
        y_test,
        X_test["fighter1_backtesting_odds"], 
        X_test["fighter2_backtesting_odds"],
        return_loss=True,
        input_probs=True)

    # Return the metrics
    
    return {
        "X Odds Accuracy": accuracy_score(y_test, y_pred),
        "X Odds W/L AUC": wl_auc,
        "X Odds L/W AUC": lw_auc,
        "X Odds W/L Accuracy": wl_accuracy,
        "X Odds L/W Accuracy": lw_accuracy,
        "X Odds Avg WL only AUC": np.mean([wl_auc, lw_auc]),
        "X Odds Logloss": log_loss_result,
        "X Odds EV MSE": ev_mse,
        "X Odds Hmean WL only Accuracy": hmean([lw_accuracy, wl_accuracy])
    }

# This function is used to apply the calculate metrics function to a group
def group_apply_calculate_metrics(group):
    
    return calculate_metrics(group["y_pred_prob"], group["outcome"], classes)

# This function creates a seaborn chart to track a metric. It returns two objects, the figure and axes

# x and y should be pandas Series

def metric_chart(x, y_model, y_odds, x_label, y_label):
    
    # pdb.set_trace()
    
    # Set Seaborn style
    sns.set_theme(style="whitegrid")  # Clean background with grid
    
    # Put the model and odds values into a single data frame
    chart_df = pd.DataFrame({
        x_label: x.tolist() * 2,
        y_label: y_model.tolist() + y_odds.tolist(),
        "Source": ["Model"] * len(y_model) + ["Odds"] * len(y_odds)})
    
    # Create the figure and plot
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=chart_df[x_label], y=chart_df[y_label], hue=chart_df["Source"], ax=ax, palette=["lightblue", "orange"])
    
    # Set labels and title
    ax.set_ylabel(y_label)
    
    # Rotate x-axis labels (Seaborn handles spacing better)
    ax.set_xticklabels(x, rotation=90)
    
    # Adjust layout to prevent text cutoff
    plt.tight_layout() 
    
    return fig, ax

# This function calculates the log loss over a trailing window of months ending at each fight date

def rolling_log_loss(actuals_data, prediction_probabilities, target, months=8, min_sample_size=50, classes=np.array([0, 1, 2])):

    """
    Calculates the log loss of the model and of the odds over a trailing window of fights ending at
    each date in the data, so a drift in performance shows up as a trend rather than as noise.

    Args:
        actuals_data (data frame): the data the predictions are for, with date and backtesting odds columns
        prediction_probabilities (array): the probabilities for each class of the outcome
        target (series): the outcome as classes, from clean_target_column
        months (int): the length of the trailing window
        min_sample_size (int): windows holding fewer fights than this are dropped, since a log loss on a handful of fights says nothing

    Returns:
        A data frame of one row per window with the model and odds log loss and the sample size
    """

    # The probabilities and the target line up with the rows positionally, so a length mismatch means
    # they are for a different sample and the metrics would be quietly wrong rather than wrong loudly
    stop_if_not(len(prediction_probabilities) == len(actuals_data) and len(target) == len(actuals_data),
                "Predictions and target match the actuals row count in rolling_log_loss")

    # Get the probabilities implied by the odds to compare against
    odds_probabilities = odds.odds_to_probabilities(actuals_data["fighter1_backtesting_odds"], actuals_data["fighter2_backtesting_odds"])

    rolling_rows = []

    for window_end in sorted(actuals_data["date"].unique()):

        window_start = pd.Timestamp(window_end) - pd.DateOffset(months=months)

        row_filter = (actuals_data["date"] > window_start) & (actuals_data["date"] <= window_end)

        if row_filter.sum() < min_sample_size:
            continue

        rolling_rows.append({
            "date": pd.Timestamp(window_end),
            "Logloss": log_loss(target[row_filter], prediction_probabilities[row_filter], labels=classes),
            "X Odds Logloss": log_loss(target[row_filter], odds_probabilities[row_filter], labels=classes),
            "Sample Size": int(row_filter.sum()),
            # Draws and no contests are class 1 and get almost no probability, so log loss jumps when one
            # lands in the window. Counting them makes a spike explainable rather than looking like drift
            "OTH Fights": int((target[row_filter] == 1).sum())
            })

    # Name the columns so an empty result is still usable rather than a frame with nothing on it
    return pd.DataFrame(rolling_rows, columns=["date", "Logloss", "X Odds Logloss", "Sample Size", "OTH Fights"])

# This function charts the rolling log loss of the model against the odds

def plot_rolling_log_loss(rolling_log_loss_df, months):

    # Set Seaborn style
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(rolling_log_loss_df["date"], rolling_log_loss_df["Logloss"], label="Model", color="tab:blue")
    ax.plot(rolling_log_loss_df["date"], rolling_log_loss_df["X Odds Logloss"], label="Odds", color="orange")

    ax.set_xlabel("Window end date")
    ax.set_ylabel("Logloss")
    ax.set_title("Rolling " + str(months) + " month logloss")
    ax.legend()

    # Rotate the dates so they don't overlap
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    return fig, ax

# This function compares the log loss of groups of fights between two periods

def log_loss_by_group_and_period(actuals_data, prediction_probabilities, target, group_col, period_col, min_sample_size=20, classes=np.array([0, 1, 2])):

    """
    Calculates the log loss of the model and of the odds for each group of fights within each period,
    and the change between the first and last period. The odds columns are there to separate a model
    that has got worse from a kind of fight that got harder for everyone to call.

    Args:
        actuals_data (data frame): the data the predictions are for, with the group and period columns on it
        prediction_probabilities (array): the probabilities for each class of the outcome
        target (series): the outcome as classes, from clean_target_column
        group_col (str): the column splitting the fights, e.g. the method of victory
        period_col (str): the column splitting the dates. Name the values so they sort in date order
        min_sample_size (int): cells holding fewer fights than this are left out, since a log loss on a handful of fights is meaningless and a single draw in a small cell sends it into double figures

    Returns:
        A data frame of one row per group with a column per period and the change between them
    """

    # Same positional alignment as the rolling version, so check it before calculating anything
    stop_if_not(len(prediction_probabilities) == len(actuals_data) and len(target) == len(actuals_data),
                "Predictions and target match the actuals row count in log_loss_by_group_and_period")

    # Get the probabilities implied by the odds to compare against
    odds_probabilities = odds.odds_to_probabilities(actuals_data["fighter1_backtesting_odds"], actuals_data["fighter2_backtesting_odds"])

    group_rows = []

    for group in sorted(actuals_data[group_col].unique()):
        for period in sorted(actuals_data[period_col].unique()):

            row_filter = (actuals_data[group_col] == group) & (actuals_data[period_col] == period)

            if row_filter.sum() < min_sample_size:
                continue

            group_rows.append({
                group_col: group,
                period_col: period,
                "Logloss": log_loss(target[row_filter], prediction_probabilities[row_filter], labels=classes),
                "X Odds Logloss": log_loss(target[row_filter], odds_probabilities[row_filter], labels=classes),
                "Sample Size": int(row_filter.sum()),
                # A draw or no contest costs about three times what an ordinary fight does, so a cell
                # going from one to three of them looks like the model falling apart
                "OTH Fights": int((target[row_filter] == 1).sum())
                })

    group_metrics = pd.DataFrame(group_rows)

    # Put the periods side by side so the change can be read off
    period_metrics = group_metrics.pivot(index=group_col, columns=period_col, values=["Logloss", "X Odds Logloss", "Sample Size", "OTH Fights"])
    period_metrics.columns = [metric + " " + str(period) for metric, period in period_metrics.columns]

    # Add the change between the first and last period. With only one period there is nothing to
    # compare, and a change column of zeroes would read as nothing having changed
    periods = sorted(group_metrics[period_col].unique())

    if len(periods) > 1:
        for metric in ["Logloss", "X Odds Logloss"]:
            period_metrics["Change " + metric] = period_metrics[metric + " " + str(periods[-1])] - period_metrics[metric + " " + str(periods[0])]

    return period_metrics

# Converts the metrics saved in dictionaries into a pandas dataframe

def metric_table_from_metrics_dictionary(data_dict, col):
    
    # Extract the relevant data for the given 'col'
    if col not in data_dict:
        raise ValueError(f"Column '{col}' not found in the dictionary.")
    
    # Extract and convert the data to DataFrame
    groups = data_dict[col]
    df = pd.DataFrame(groups).transpose()  # Transpose so groups become rows
    
    # Reset index to have groups as rows
    df = df.reset_index().rename(columns={"index": "group"})
    
    return df

def plot_calibration_curve(y_true, y_pred, n_bins=10):
    """
    Plot a calibration curve for a set of binary outcomes and predicted probabilities.

    Args:
        y_true (array-like): True binary outcomes (0 or 1).
        y_pred (array-like): Predicted probabilities for the positive class.
        n_bins (int, optional): Number of bins to divide predictions into. Default = 10.

    Returns:
        matplotlib.figure.Figure: The created Matplotlib figure, which can be shown,
                                  saved, or logged to MLflow.
    
    Example:
        fig = plot_calibration_curve(y_true, y_pred)
        mlflow.log_figure(fig, "calibration_curve.png")
    """
    
    # pdb.set_trace()
    
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy="uniform")

    fig, ax = plt.subplots()
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration Curve")
    ax.legend()

    return fig

def plot_roi_and_sample_size(df):
    """
    Plots ROI_on_constant_bets as a bar chart and Sample_Size as a line chart 
    on a secondary axis, using the DataFrame index as x-axis labels.
    
    Args:
        df (pd.DataFrame): Must contain 'ROI_on_constant_bets' and 'Sample_Size'
                           with index as group labels.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar chart for ROI
    ax1.bar(df.index.astype(str), df["ROI_on_constant_bets"], color="skyblue", label="ROI (%)")
    ax1.set_xlabel("Group")
    ax1.set_ylabel("ROI on Constant Bets (%)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_xticklabels(df.index.astype(str), rotation=45, ha="right")

    # Secondary axis for sample size
    ax2 = ax1.twinx()
    ax2.plot(df.index.astype(str), df["Sample_Size"], color="red", marker="o", label="Sample Size")
    ax2.set_ylabel("Sample Size", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Title and layout
    fig.suptitle("ROI and Sample Size by Group", fontsize=14)
    fig.tight_layout()

    return fig

def summarise_bet_roi_by_ev(df):
    """
    This function takes a backtesting output from backtest_model() and summarise the returns
    on investment from betting on all outcomes, and calculates the ROI for each EV band
    the model predicted.
    
    It assumes a fixed bet across all fighters of 1
    """
    
    # pdb.set_trace()
    
    # Split out data for fighter 1 and fighter 2
    f1_df = df[["Fighter1_win_EV", "fighter1_backtesting_odds"]].rename(columns={"Fighter1_win_EV":"EV", "fighter1_backtesting_odds": "odds"})
    # Add on win
    f1_df["WL"] = np.where(df["outcome"] == "W/L", 1, 0)

    f2_df = df[["Fighter2_win_EV", "fighter2_backtesting_odds"]].rename(columns={"Fighter2_win_EV":"EV", "fighter2_backtesting_odds": "odds"})
    # Add on win
    f2_df["WL"] = np.where(df["outcome"] == "L/W", 1, 0)
    
    # Stack the data to get two bets on every fight
    stacked_df = pd.concat([f1_df, f2_df]).reset_index(drop=True)

    # Now cut EV into bands
    stacked_df["EV_band"] = pd.cut(stacked_df["EV"], [-10, -0.25, -0.1, 0, 0.1, 0.25, 10], include_lowest=True)
    
    stacked_df["ROI_on_constant_bets"] = np.where(stacked_df["WL"] == 1, stacked_df["odds"] - 1, -1)
    
    # Summarise the output
    output_df = stacked_df.sort_values("EV").groupby("EV_band", observed=False)[["ROI_on_constant_bets"]].mean() * 100
    
    # Add on the sample size
    output_df["Sample_Size"] = stacked_df.sort_values("EV").groupby("EV_band", observed=False)["ROI_on_constant_bets"].count()
    
    return output_df

# This function takes a dataset of actuals and prediction probabilities (which must correspond in rows to each other)
# It returns a data frame with added columns for the EVs of each bet and bet sizing
def summarise_model_predictions(
        actuals_data_df, 
        prediction_probabilities, 
        min_prob_to_bet, 
        min_ev_to_bet, 
        min_bet_size=1, 
        max_bet_size=1, 
        exchange_commission=0.02, 
        return_triggers=False, 
        keep_scrape_dates=False, 
        point_in_time_only_odds=True, # Whether to target a bet after cutoff using best odds
        target_ev_after_cutoff=0): # The EV that will be targetted after cutoff

    
    # ###
    # # Debug
    # actuals_data = X_train[['id']].merge(data, on = 'id', how = 'left')
    # prediction_probabilities, temp = predict_xgboost_ensemble(rf_xgb_model, X_train, y_train)
    # breakdown_cols = ['weightclass', 'year']
    # min_prob_to_bet = 0.5
    # min_ev_to_bet = 0.1
    # ###
    
    actuals_data = actuals_data_df.copy()
    
    # pdb.set_trace()
    
    # Do a check that the inputs are consistent if doing a non constant bets. If using non constant bets, the bet sizing should be as proportions, not £ amounts

    # First calculate the Expected Values on a W/L result and L/W result, given in betting units
    # Odds are decimal odds, so represent the payout for a 1 unit bet e.g. odds of 2 means a (2 - 1) = 1 profit.
    ev_wl = (actuals_data["fighter1_backtesting_odds"] - 1) * prediction_probabilities[:, 2] - (1 - prediction_probabilities[:, 2])
    ev_lw = (actuals_data["fighter2_backtesting_odds"] - 1) * prediction_probabilities[:, 0] - (1 - prediction_probabilities[:, 0])
        
    # Now calculate bet triggers as 0 or 1 indicators
    ev_wl_trigger = ev_wl > min_ev_to_bet
    prob_wl_trigger = prediction_probabilities[:, 2] > min_prob_to_bet
    wl_trigger = 1 * ev_wl_trigger * prob_wl_trigger
    
    ev_lw_trigger = ev_lw > min_ev_to_bet
    prob_lw_trigger = prediction_probabilities[:, 0] > min_prob_to_bet
    lw_trigger = 1 * ev_lw_trigger * prob_lw_trigger
    
    # Fix any instances where both triggers happen. Remove the bet because there must be an odds mistake
    double_trigger = (lw_trigger + wl_trigger) > 1
    lw_trigger[double_trigger] = 0
    wl_trigger[double_trigger] = 0
        
    # Set whether a bet is made
    actuals_data["bets"] = wl_trigger + lw_trigger
    
    # Add the EV for later breakdowns
    actuals_data["bet_ev"] = np.where(actuals_data["bets"] == 0, np.nan, wl_trigger * ev_wl + lw_trigger * ev_lw)
    #  Convert it into buckets
    # bucket_width = 0.1
    # col_min = np.round(np.floor(actuals_data['bet_ev'].min() / bucket_width) * bucket_width, 2)
    # col_max = np.round(np.ceil(actuals_data['bet_ev'].max() / bucket_width) * bucket_width, 2)
    actuals_data["bet_ev_band"] = pd.cut(actuals_data["bet_ev"],
                                    bins=[0, 0.05, 0.1, 0.2, np.inf],
                                    labels=["0 - 0.05", "0.05 - 0.1", "0.1 - 0.2", "0.2+"])
    
    
    ## Determine the bet size. Begin with the minimum, then add on the confidence of the bet
    
    # First get the probabilities of winning and odds, irrespective of fighter to win
    win_prob = wl_trigger * prediction_probabilities[:, 2] + lw_trigger * prediction_probabilities[:, 0]
    win_odds = wl_trigger * actuals_data["fighter1_backtesting_odds"] + lw_trigger * actuals_data["fighter2_backtesting_odds"]
    
    actuals_data["kelly_criterion"] = win_prob - (1 - win_prob) / (win_odds - 1)
    
    # Perform a check that kelly is correct if only +EV bets, it should always be positive.
    if min_ev_to_bet >= 0:
        stop_if_not((actuals_data["kelly_criterion"].max() <= 1) and (actuals_data["kelly_criterion"].min() >= 0),
                    "Kelly criterion between 0 and 1 with no negative EV bets in summarise_model_predictions")
    else:
        stop_if_not((actuals_data["kelly_criterion"].max() <= 1),
                    "Kelly criterion no greater than 1 in summarise_model_predictions")
    
    # And use it to calculate how much to increase from the min
    actuals_data["bet_size"] = actuals_data["bets"] * (min_bet_size + (max_bet_size - min_bet_size) * actuals_data["kelly_criterion"])

    # Check no bets breach the max
    stop_if_not(actuals_data["bet_size"].max() <= max_bet_size, "No bet size above max_bet_size in summarise_model_predictions")
    
    
    #######################################
    # Now calculate the target odds, for everyone, but especially those with no bets
    wl_target_odds_inc_commission = odds.calculate_target_odds(prediction_probabilities[:, 2], target_ev_after_cutoff, exchange_commission)
    lw_target_odds_inc_commission = odds.calculate_target_odds(prediction_probabilities[:, 0], target_ev_after_cutoff, exchange_commission)
    
    # And do the same for lay odds
    wl_target_lay_odds_inc_commission = odds.calculate_target_lay_odds(prediction_probabilities[:, 2], target_ev_after_cutoff, exchange_commission)
    lw_target_lay_odds_inc_commission = odds.calculate_target_lay_odds(prediction_probabilities[:, 0], target_ev_after_cutoff, exchange_commission) 
    
    # Calculate target betsizes
    actuals_data["fighter1_target_bet_size"] = (min_bet_size + (max_bet_size - min_bet_size) * (
        prediction_probabilities[:, 2] - (1 - prediction_probabilities[:, 2]) / (wl_target_odds_inc_commission - 1)))
        
    actuals_data["fighter2_target_bet_size"] = (min_bet_size + (max_bet_size - min_bet_size) * (
        prediction_probabilities[:, 0] - (1 - prediction_probabilities[:, 0]) / (lw_target_odds_inc_commission - 1)))
    
    
    #######################################
    # Summarise key outputs
    actuals_data["Bet_breakeven_odds"] = np.where(wl_trigger == 1, 1 / prediction_probabilities[:, 2],
                                                  np.where(lw_trigger == 1, 1 / prediction_probabilities[:, 0], np.nan))    # checked the maths and this is how it works out
    actuals_data["Bet_breakeven_exchange_odds"] = np.where(wl_trigger == 1, (1 / prediction_probabilities[:, 2] - 1) / (1 - exchange_commission) + 1,
                                                  np.where(lw_trigger == 1, (1 / prediction_probabilities[:, 0] - 1) / (1 - exchange_commission) + 1, np.nan))
    
    # Both wl and lw are back bets, so use back-odds formula for both
    actuals_data["Bet_target_exchange_odds"] = np.where(wl_trigger == 1, wl_target_odds_inc_commission,
                                                        np.where(lw_trigger == 1, lw_target_odds_inc_commission, np.nan))
    
    actuals_data["Fighter1_win_prob"] = np.round(prediction_probabilities[:, 2] * 100, 1)
    actuals_data["Fighter1_win_EV"] = np.round(ev_wl, 2)
    actuals_data["Fighter2_win_prob"] = np.round(prediction_probabilities[:, 0] * 100, 1)
    actuals_data["Fighter2_win_EV"] = np.round(ev_lw, 2)
    actuals_data["Bet_trigger"] = np.where(wl_trigger == 1, actuals_data["fighter1"], np.where(lw_trigger == 1, actuals_data["fighter2"], "None"))
    actuals_data["Predict_to_win"] = np.where((actuals_data["Fighter1_win_prob"] > actuals_data["Fighter2_win_prob"]) & (actuals_data["fighter1_backtesting_odds"] > actuals_data["fighter2_backtesting_odds"]), 
                                              "Underdog to win",
                                             np.where((actuals_data["Fighter2_win_prob"] > actuals_data["Fighter1_win_prob"]) & (actuals_data["fighter2_backtesting_odds"] > actuals_data["fighter1_backtesting_odds"]),
                                             "Underdog to win",
                                             "Favourite to win"))
    actuals_data["Fighter_to_win"] = np.where(actuals_data["Fighter1_win_prob"] > actuals_data["Fighter2_win_prob"], "Fighter 1 to win", "Fighter 2 to win")
    actuals_data["fighter1_target_exchange_odds"] = wl_target_odds_inc_commission
    actuals_data["fighter2_target_exchange_odds"] = lw_target_odds_inc_commission
    
    if return_triggers:
        return [actuals_data, wl_trigger, lw_trigger]
    else:
        return actuals_data
