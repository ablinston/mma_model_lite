# This function cleans infinite values and large numbers

def clean_inf_and_large(data):
    
    data.replace([np.inf, -np.inf], [99999, -99999], inplace=True)
    data = data.infer_objects(copy=False)  # Ensures data types remain unchanged
    
    # Replace large float values with 99999
    data[data.select_dtypes(include=[np.float64]) > 99999] = 99999
    data[data.select_dtypes(include=[np.float64]) < -99999] = -99999
    
    return data
