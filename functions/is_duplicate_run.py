# This function checks if there is a duplicate run in mlflow

def is_duplicate_run(params):
    # Convert params dictionary to a search filter
    filters = " and ".join([f"params.{key} = '{value}'" for key, value in params.items()])
    # Search for existing runs with the same parameters
    existing_runs = mlflow.search_runs(filter_string=filters)
    return not existing_runs.empty