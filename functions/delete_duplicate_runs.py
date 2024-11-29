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