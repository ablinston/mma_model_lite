# Function to handle different cases for the mode
def safe_mode(x):
    modes = x.mode()
    if modes.empty:
        return pd.NaT  # Return NaT if no mode found
    else:
        return modes.iloc[0]  # Take the first mode if it exists