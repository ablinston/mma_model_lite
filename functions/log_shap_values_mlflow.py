# This function logs the SHAP values for a model's predictions on MLflow

def log_shap_values_mlflow(xgb_forest, data):
    
    # pdb.set_trace()
    
    # Just use the first model in the forest
    # Clean the test features
    X_test_df = clean_inf_and_large(data).copy()

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