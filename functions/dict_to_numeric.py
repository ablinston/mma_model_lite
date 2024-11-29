# This function converts elements of a dictionary to numeric where possible
def dict_to_numeric(dict):
    
    return {k: pd.to_numeric(v, errors='ignore') for k, v in dict.items()}

