# This function is used to apply the calculate metrics function to a group
def group_apply_calculate_metrics(group):
    
    return calculate_metrics(group['y_pred_prob'], group['outcome'], classes)