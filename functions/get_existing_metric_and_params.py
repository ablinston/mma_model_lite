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