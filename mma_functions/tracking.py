import mlflow
import pandas as pd
import xgboost as xgb

from mma_functions import cleaning, evaluation

# This function deletes all runs in MLflow with duplicate parameters

def delete_duplicate_runs(experiment_name):

    # pdb.set_trace()

    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # Fetch all runs for the specified experiment
    all_runs = mlflow.search_runs(experiment_ids=[experiment_id], filter_string="", output_format="pandas")

    # Sort runs by start time to retain the earliest run when duplicates are found
    all_runs = all_runs.sort_values("start_time", ascending=True)

    # Identify duplicates based on the parameters
    # Convert parameter columns to a tuple to allow comparison
    param_columns = [col for col in all_runs.columns if col.startswith("params.")]
    all_runs['params_tuple'] = all_runs[param_columns].apply(tuple, axis=1)

    # Identify duplicate parameter tuples, keeping the first occurrence
    duplicate_runs = all_runs[all_runs.duplicated(subset='params_tuple', keep='first')]

    # Delete each duplicate run
    for run_id in duplicate_runs['run_id']:
        print(f"Deleting duplicate run: {run_id}")
        mlflow.delete_run(run_id)

    print(f"Deleted {len(duplicate_runs)} duplicate runs.")

# This function looks at an mlflow experiment and gets the best set of parameters

def get_best_params_mlflow(experiment_name, metric_name, maximise = True):

    # pdb.set_trace()

    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # Fetch all runs for the specified experiment
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    
    # Check if any runs are available
    if runs.empty:
        print("No runs found in the specified experiment.")
        return None
    
    # Sort by the specified metric, descending if maximizing
    sorted_runs = runs.sort_values(by=[f"metrics.{metric_name}"], ascending = not maximise)
    
    # Get the best run ID
    best_run_id = sorted_runs.iloc[0]["run_id"]
    
    # Retrieve the best run's parameters
    best_run = mlflow.get_run(best_run_id)
    best_params = best_run.data.params
    
    return best_params

# This function gets an existing metric from the MLflow experiment for a set of parameters and also return the full set of parameters

def get_existing_metric_and_params(params, metric):
    filters = " and ".join([f"params.{key} = '{value}'" for key, value in params.items()])
    existing_runs = mlflow.search_runs(filter_string=filters)
    if not existing_runs.empty:
        # Retrieve the first matched run's metric (assuming metric name is 'loss')
        
        # pdb.set_trace()
        # Get the other parameters from the run
        all_params = existing_runs[[col for col in existing_runs.columns if 'params.' in col]]
        
        # Reformat names
        all_params.columns = [col.replace('params.', '') for col in all_params.columns]
        
        # Convert numbers to numeric
        for col in all_params.columns.to_list():
            try:
                all_params.loc[:, col] = pd.to_numeric(all_params[col])
            except:
                pass
        
        # Return two objects
        return float(existing_runs.iloc[0]['metrics.' + metric]), all_params.to_dict(orient = 'records')[0]
    
    return None

# This function gets the params from an mlflow run by run name

def get_params_by_run_name(experiment_name, target_run_name):
    """
    Retrieves all parameters for a specified run name within an MLflow experiment.

    Parameters:
    - experiment_id: ID of the MLflow experiment to search within.
    - target_run_name: Name of the run to search for.

    Returns:
    - params: A dictionary of parameters for the specified run.
    """
    # Get the experiment ID
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # Initialize an empty dictionary to store parameters
    params = {}

    # Get all runs in the specified experiment
    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    # Filter for the specific run name
    target_run = runs[runs['tags.mlflow.runName'] == target_run_name]

    if target_run.empty:
        print(f"No run found with name '{target_run_name}' in experiment ID '{experiment_id}'")
    else:
        # Retrieve run_id and fetch parameters for the first matching run
        run_id = target_run.iloc[0]['run_id']
        run = mlflow.get_run(run_id)
        params = run.data.params

    return params

# This function checks if there is a duplicate run in mlflow
def is_duplicate_run(params, parameter_to_check=None):
    """
    Check if there is a duplicate run in MLflow based on a specific parameter or all parameters.

    Args:
        params (dict): Dictionary of parameters to search for.
        parameter_to_check (str, optional): The name of the parameter to check for.
                                            If None, checks all parameters.
    
    Returns:
        bool: True if a duplicate run exists, False otherwise.
    """
    if parameter_to_check:
        # Ensure the specified parameter exists in params
        if parameter_to_check not in params:
            raise ValueError(f"The parameter '{parameter_to_check}' is missing in params.")
        # Build the filter string using only the specified parameter
        filters = f"params.{parameter_to_check} = '{params[parameter_to_check]}'"
    else:
        # Build the filter string using all parameters
        filters = " and ".join([f"params.{key} = '{value}'" for key, value in params.items()])
    
    # Search for existing runs with the constructed filter
    existing_runs = mlflow.search_runs(filter_string=filters)
    
    return not existing_runs.empty

def log_metrics_mlflow(y_pred_prob, y_test_df, fighter1_odds, fighter2_odds, classes):
    
    """
    Take a set of predictions and actuals and log all required metrics in MLflow
    
    Args:
        y_pred_prob (array): an array of probabilities for the predictions
        y_test_df: a series of actuals to test against
        classes (array): the classes of the predicted variable
        
    Returns:
        Nothing, but the metrics will be logged in MLflow
    
    """
    # pdb.set_trace()

    # Calculate the metrics
    metrics = evaluation.calculate_metrics(y_pred_prob, y_test_df, fighter1_odds, fighter2_odds, classes)
    
    mlflow.log_metrics(metrics)
    

    return None

# This function takes the dataset and logs the metrics based purely on the odds in the data

def log_metrics_odds_mlflow(X_test, y_test):

    # Calculate the metrics
    metrics = evaluation.calculate_metrics_odds(X_test, y_test) 

    # Log the metrics
    mlflow.log_metrics(metrics)
    
    return None

# This function logs the SHAP values for a model's predictions on MLflow

def log_shap_values_mlflow(xgb_forest, data):
    
    # pdb.set_trace()
    
    # Just use the first model in the forest
    # Clean the test features
    X_test_df = cleaning.clean_inf_and_large(data).copy()

    # Go through each of the models to get SHAP values
    shap_values_list = []

    for model in xgb_forest:
        
        if 'calibrated' not in model.keys():
            model['calibrated'] = False
            
        # Get the booster from the model
        if model['calibrated'] == True:
            xgb_model = model['xgb_model'].get_booster()
        else:
            xgb_model = model['model']
            
        # Create the DMatrix for predictions
        dmatrix_test = xgb.DMatrix(X_test_df[model['features']], label = [0] * len(X_test_df))
        
        # Calculate shap values and add them to list
        shap_values = xgb_model.predict(dmatrix_test, pred_contribs=True)
        
        # Create a df from it
        shap_df = pd.DataFrame(
            shap_values[:, 0, :],
            columns=list(model['features']) + ["bias"]
        )
        
        shap_values_list.append(shap_df)
        
    # Union of all feature names across models
    all_features = sorted(set().union(*[df.columns for df in shap_values_list]))
    
    # Reindex each df so they all have the same columns
    aligned_dfs = [df.reindex(columns=all_features, fill_value=0) for df in shap_values_list]

    # Concatenate along new axis, then take mean
    avg_shap = sum(aligned_dfs) / len(aligned_dfs)
    
    # Separate features into three groups
    odds_features = [col for col in all_features if 'odds' in col]
    dob_features = [col for col in all_features if 'dob' in col]
    other_features = [col for col in all_features if ('odds' not in col) and ('dob' not in col) and (col != 'bias')]
    
    feature_groups = {
        "Odds": odds_features,
        "DOB": dob_features,
        "Other": other_features,
        "Bias" : ['bias']
    }
    
    grouped_shap = {}

    for group_name, features in feature_groups.items():
        # sum or mean absolute SHAP values for this group
        grouped_shap[group_name] = avg_shap[features].abs().mean().sum()  # sum across features
    
    # Convert to df then percentages
    grouped_shap_df = pd.DataFrame(grouped_shap, index = ['SHAP abs %'])
    grouped_shap_df.loc['SHAP abs sum', :] = grouped_shap_df.loc['SHAP abs %', :] / grouped_shap_df.loc['SHAP abs %', :].sum()
    
    # Save the output and log in MLflow
    grouped_shap_df.to_csv('./results/temp/SHAP_values.csv')
    
    mlflow.log_artifact('./results/temp/SHAP_values.csv', "Feature SHAP")

    return "SHAP values logged" 
