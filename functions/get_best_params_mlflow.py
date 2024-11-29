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
