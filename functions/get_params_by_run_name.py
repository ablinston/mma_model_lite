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


