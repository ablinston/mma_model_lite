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
