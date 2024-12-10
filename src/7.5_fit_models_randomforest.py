

import gc
import h2o
from h2o.automl import H2OAutoML
from hyperopt import hp, STATUS_OK, tpe, fmin, Trials # hyperparameter optimisation functions
from hyperopt.atpe import suggest as atpe_suggest
import itertools
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import pickle
import shap
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
import xgboost as xgb

# Read in global objects and packages
exec(open('global.py').read())

# Read in the data for training
data = concat_parquets_fastparquet('./data/final_dataset/')


##########################################################################################
### Data prep for model ###
##########################################################################################

# Apply standard filters
data = standard_data_filter(data)

# Set target
target = 'outcome'

# Switch fighers if necessary

X_train, X_test, y_train, y_test, y_test_bin, no_of_classes, oot = prep_data_for_model_training(data, config)

# Check data types
data.dtypes.value_counts()

# Set number of classes for training
no_of_classes = len(set(y_train))


# Set up experiment training
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment('all_fights')


##########################################################################################
### Random Forest - For feature importance
##########################################################################################

# These initial RF runs are to get variable importance in order to check what the models will
# likely be using and to remove columns that aren't adding much value. This will speed up training
# later on when we get to hyperparameter tuning.

# Set the parameters for the model
params = {
    'nthread': 3, # Only applies to CPU cores
    'booster': 'gbtree',
    'num_boost_round': 1, # to tell the package to train only 1 random forest
    'colsample_bynode': np.sqrt(len(data.columns)) / len(data.columns), # fraction of columns to select for each tree
    'num_parallel_tree': 2500, # the size of the forest. For 16k features, and choosing sqrt, this gives 99.998% chance all features selected at least once
    'learning_rate': 1, # Learning rate
    'min_child_weight': 15,                 # The min observations that should be left in a leaf node    
    'subsample': 0.85,
    # 'max_depth': 6,
    'max_bin': 200, # used in the hist method to speed up, this says how many bins the vars should be split into
    'objective': 'multi:softprob', # Multi classification objective
    'eval_metric': 'mlogloss',       # Evaluation metric
    'tree_method': 'hist',      # Use GPU accelerated algorithm
    'device': 'cuda',           # Use CUDA / GPU
    'predictor': 'gpu_predictor',   # Use GPU for prediction
    'verbosity': 1,
    'num_class': no_of_classes,
    'n_folds': 5,                   # Number of folds for the cross-validation
    'random_state': config['seed_all'] + 38
}

# Prepare data
train = xgb.DMatrix(X_train, label = y_train)

with mlflow.start_run():

    # Log the params
    mlflow.log_params(params)
    mlflow.log_param('model', "RandomForest")

    # Train the model
    rf_model = xgb.train(params, train)

    # Save the model in case we want to use it later
    rf_model.save_model("results/7.5_random_forest_model.xgb")
    
    # Get feature importance scores
    importance = rf_model.get_score(importance_type='total_gain')

    # Sort features by importance
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    # Save the variable importance plot to csv
    with open('results/random_forest_variable_importance.csv', mode='w', newline='') as file:
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
    shap_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Mean SHAP Value': mean_shap_values})
    
    # Sort features by importance
    shap_importance_df = shap_importance_df.sort_values(by='Mean SHAP Value', ascending=False).reset_index(drop=True)
    
    # Display or save the table
    print(shap_importance_df.head(10))  # Shows the top 10 most important features
    # Optionally, save to CSV
    shap_importance_df.to_csv('./results/shap_feature_importance.csv', index=False)
