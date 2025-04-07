# This function converts elements of a dictionary to numeric where possible
def dict_to_numeric(dict):
    
    return pd.to_numeric(pd.Series(dict), errors='coerce').fillna(pd.Series(dict)).to_dict()

