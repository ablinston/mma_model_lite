def clean_target_column(target_column):
    """
    Cleans the target column from a text entry to a categorised, allowing consistent formatting across function

    Args:
        target_column (series): the input column
    
    Returns
        A series to be input as the new column

    """
    # pdb.set_trace()
    
    # Recode anything that isn't Wins or losses as other
    cleaned_target = np.where(target_column.isna(), np.nan, np.where((target_column == "D/D") | (target_column == "NC/NC"), "OTH", target_column))
    
    # Convert the target into a labelled column ready for multiclass modelling
    le = LabelEncoder()
    cleaned_target = le.fit_transform(cleaned_target)
    
    # Stop if the classes aren't as expected. These are hardcoded in other functions so need to be in this order
    stop_if_not(le.classes_.tolist() == ['L/W', 'OTH', 'W/L'], "Target class check")

    return cleaned_target
