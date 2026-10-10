
import csv
import gc
# import h2o
# from h2o.automl import H2OAutoML
from hyperopt import hp, STATUS_OK, tpe, fmin, Trials # hyperparameter optimisation functions
from hyperopt.atpe import suggest as atpe_suggest
import itertools
# import lightgbm as lgb
import math
import matplotlib.pyplot as plt
import mlflow
import pickle
import shap
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
import xgboost as xgb

# Read in global objects and packages
import fastparquet as fp
import numpy as np
import pandas as pd

from mma_functions.cleaning import standard_data_filter
from mma_functions.dataset import prep_data_for_model_training, standard_feature_removal
from mma_functions.features import switch_fighter1_fighter2
from mma_functions.modelling import create_custom_loss, get_base_margin
from mma_functions.config import config, model_hyper_params

# Read in the data for training
data = fp.ParquetFile("./data/final_dataset/final_dataset.parquet").to_pandas() # pq.ParquetDataset('./data/final_dataset/').read().to_pandas()


##########################################################################################
### Data prep for model ###
##########################################################################################

# Apply standard filters
data = standard_data_filter(data)

# Set target
target = "outcome"

# Switch fighers if necessary
if model_hyper_params["switch_fighters_with_odds"] == True:
    data = switch_fighter1_fighter2(data, data["fighter1_backtesting_odds"] > data["fighter2_backtesting_odds"])
if model_hyper_params["switch_fighters_randomly"] == True: # Commented this out because I don't think we want random for the RF
    np.random.seed(seed=config["seed_all"] + 9845)
    # Switch twice then concat to try to get some variance
    data = switch_fighter1_fighter2(data, np.random.choice([True, False], size=len(data)))
    # data2 = switch_fighter1_fighter2(data, np.random.choice([True, False], size = len(data)))
    
    # data = pd.concat([data, data2, data3], ignore_index = True)

X_train_master, X_test, y_train, y_test, y_test_bin, no_of_classes, oot = prep_data_for_model_training(data, config)

# Now remove any columns that aren't wanted in a training dataset. At this point we don't remove non-true odds in order to see the rank in feature importance
X_train_master = standard_feature_removal(X_train_master, differentials_only=model_hyper_params["differential_features_only"], true_odds_only=False, open_odds_only=False)

# Check data types
data.dtypes.value_counts()

# Set number of classes for training
no_of_classes = len(set(y_train))


# Set up experiment training
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("all_fights")


##########################################################################################
### Random Forest - For feature importance
##########################################################################################

# These initial RF runs are to get variable importance in order to check what the models will
# likely be using and to remove columns that aren't adding much value. This will speed up training
# later on when we get to hyperparameter tuning.

# Set the parameters for the model
params = {
    "nthread": 3, # Only applies to CPU cores
    "booster": "gbtree",
    "num_boost_round": 1, # to tell the package to train only 1 random forest
    "colsample_bynode": np.sqrt(len(data.columns)) / len(data.columns), # fraction of columns to select for each tree
    "num_parallel_tree": 1500, # the size of the forest. For 16k features, and choosing sqrt, 2500 gives 99.998% chance all features selected at least once
    "learning_rate": 1, # Learning rate
    "min_child_weight": 15,                 # The min observations that should be left in a leaf node    
    "subsample": 0.85,
    # 'max_depth': 6,
    "max_bin": 175, # used in the hist method to speed up, this says how many bins the vars should be split into
    "objective": "multi:softprob", # Multi classification objective
    "eval_metric": "mlogloss",       # Evaluation metric
    "tree_method": "hist",      # Use GPU accelerated algorithm
    "device": "cuda",           # Use CUDA / GPU
    "predictor": "gpu_predictor",   # Use GPU for prediction
    "verbosity": 2,
    "num_class": no_of_classes,
    "random_state": config["seed_all"] + 38
}


# Get base margin cols if needed
base_margin_train = get_base_margin(X_train_master, model_hyper_params["base_margin_cols"])

# Remove any objects not for modelling
X_train_master = X_train_master[[col for col in X_train_master if "notformodel" not in col]]

# Count the number of runs it's going to take
runs_required = math.ceil(len(X_train_master.columns) / 10000)

for i in range(0, runs_required):
    
    # Trim features if too many
    X_train = X_train_master.iloc[:, (10000 * i):np.min([10000 * (i + 1), len(X_train_master.columns) + 1])]

    # Prepare data
    train = xgb.DMatrix(X_train, label=y_train, base_margin=base_margin_train)
    
    print("Training RF " + str(i) + "...")
    # Add the odds in case we're using custom loss
    # train.set_info(fighter1_backtesting_odds=X_train['fighter1_backtesting_odds'].values,
    #                fighter2_backtesting_odds=X_train['fighter2_backtesting_odds'].values)


    with mlflow.start_run():
    
        # Log the params
        mlflow.log_params(params)
        mlflow.log_param("model", "RandomForest partial iteration" + str(i))
        mlflow.log_param("features", len(X_train.columns))
    
        # Train the model
        if model_hyper_params["loss_function"] == "ev_loss_function": # any eval metric above will be ignored
        
            # Pass the odds to the loss function
            custom_loss = create_custom_loss(X_train["fighter1_backtesting_odds"].values, X_train["fighter2_backtesting_odds"].values)
            # Train the model
            rf_model = xgb.train(params, train, obj=custom_loss)
            
        else:
            rf_model = xgb.train(params, train)
            
        # Get feature importance scores
        importance = rf_model.get_score(importance_type="total_gain")
    
        # Sort features by importance
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # Save the variable importance plot to csv
        with open("results/random_forest_variable_importance_" + str(i) + ".csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Feature", "Importance"])  # Write the header
            writer.writerows(sorted_importance)
    
        # TreeSHAP calculation
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_train)
        
        # # Plot SHAP feature importance
        # shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        # plt.title("SHAP Feature Importance (TreeSHAP)")
        # plt.savefig('results/treeshap_feature_importance.png')
        # plt.show()
        
        # Combine SHAP values with feature names into a DataFrame for each target class
        # If you have a single-target classification, shap_values will be a single array
        # For multi-class, it will be a list of arrays, one for each class
        
        # Assuming `shap_values` has the shape (6792, 18000, 3)
        # Take the mean absolute SHAP values across all samples and classes
        mean_shap_values = np.mean(np.abs(shap_values), axis=(0, 2))
        
        # Create a DataFrame for better readability
        shap_importance_df = pd.DataFrame({"Feature": X_train.columns, "Mean SHAP Value": mean_shap_values})
        
        # Sort features by importance
        shap_importance_df = shap_importance_df.sort_values(by="Mean SHAP Value", ascending=False).reset_index(drop=True)
        
        # Display or save the table
        print(shap_importance_df.head(10))  # Shows the top 10 most important features
        # Optionally, save to CSV
        shap_importance_df.to_csv("./results/shap_feature_importance_" + str(i) + ".csv", index=False)


if runs_required > 1:
    
    # Read in the baseline run and remove any columns with very low accuracy historically to speed up the RF
    for i in range(0, runs_required):
        
        if i == 0:
            var_importance = pd.read_csv("./results/shap_feature_importance_" + str(i) + ".csv").sort_values("Mean SHAP Value", ascending=False)
            # Take top 2000
            var_importance = var_importance[0:2000]
        else:
            temp_importance = pd.read_csv("./results/shap_feature_importance_" + str(i) + ".csv").sort_values("Mean SHAP Value", ascending=False)
            # Take top 2000
            temp_importance = temp_importance[0:2000]
            
            # Combine with other SHAPs
            var_importance = pd.concat([var_importance, temp_importance], axis=0)
   
    # Select only the top columns
    X_train = X_train_master[[col for col in X_train_master.columns if col in var_importance["Feature"].to_list()]]
    
    # Prepare data
    train = xgb.DMatrix(X_train, label=y_train, base_margin=base_margin_train)
    
    # Train the final random forest
    with mlflow.start_run():
    
        # Log the params
        mlflow.log_params(params)
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("features", len(X_train.columns))
    
        # Train the model
        if model_hyper_params["loss_function"] == "ev_loss_function": # any eval metric above will be ignored
        
            # Pass the odds to the loss function
            custom_loss = create_custom_loss(X_train["fighter1_backtesting_odds"].values, X_train["fighter2_backtesting_odds"].values)
            # Train the model
            rf_model = xgb.train(params, train, obj=custom_loss)
            
        else:
            rf_model = xgb.train(params, train)
    
        # Save the model in case we want to use it later
        rf_model.save_model("results/7.5_random_forest_model.xgb")
        
        # Get feature importance scores
        importance = rf_model.get_score(importance_type="total_gain")
    
        # Sort features by importance
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # Save the variable importance plot to csv
        with open("results/random_forest_variable_importance.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Feature", "Importance"])  # Write the header
            writer.writerows(sorted_importance)
    
        # TreeSHAP calculation
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_train)
    
        # # If the above doesn't work
        # shap_values = rf_model.predict(train, pred_contribs=True)    
    
        # # Plot SHAP feature importance
        # shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        # plt.title("SHAP Feature Importance (TreeSHAP)")
        # plt.savefig('results/treeshap_feature_importance.png')
        # plt.show()
        
        # Combine SHAP values with feature names into a DataFrame for each target class
        # If you have a single-target classification, shap_values will be a single array
        # For multi-class, it will be a list of arrays, one for each class
        
        # Assuming `shap_values` has the shape (6792, 18000, 3)
        # Take the mean absolute SHAP values across all samples and classes
        mean_shap_values = np.mean(np.abs(shap_values), axis=(0, 2))
        
        # Create a DataFrame for better readability
        shap_importance_df = pd.DataFrame({"Feature": X_train.columns, "Mean SHAP Value": mean_shap_values})
        
        # Sort features by importance
        shap_importance_df = shap_importance_df.sort_values(by="Mean SHAP Value", ascending=False).reset_index(drop=True)
        
        # Display or save the table
        print(shap_importance_df.head(10))  # Shows the top 10 most important features
        # Optionally, save to CSV
        shap_importance_df.to_csv("./results/shap_feature_importance.csv", index=False)
