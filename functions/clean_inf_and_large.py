# This function cleans infinite values and large numbers

def clean_inf_and_large(data):
    
    # Replace infinite values with large values
    data.replace([np.inf], 99999, inplace = True)
    data.replace([-np.inf], -99999, inplace = True)
    
    # Replace large float values with 99999
    data[data.select_dtypes(include=[np.float64]) > 99999] = 99999
    data[data.select_dtypes(include=[np.float64]) < -99999] = -99999
    
    return data