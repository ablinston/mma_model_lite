# This function gets an existing metric from the MLflow experiment for a set of parameters

def get_existing_metric(params, metric):
    filters = " and ".join([f"params.{key} = '{value}'" for key, value in params.items()])
    existing_runs = mlflow.search_runs(filter_string=filters)
    if not existing_runs.empty:
        # Retrieve the first matched run's metric (assuming metric name is 'loss')
        return float(existing_runs.iloc[0]['metrics.' + metric])
    return None