
def ev_loss_function(y_pred, y_true, odds_fighter_1, odds_fighter_2, return_loss = False, for_feval = False, input_probs = False):
    """
    Custom loss function for XGBoost that optimizes based on Expected Value (EV) from betting odds.

    Parameters:
    - y_true (np.array): True fight outcome (0 = Fighter 2 win, 1 = Draw, 2 = Fighter 1 win)
    - y_pred (np.array): Raw logits from XGBoost (before softmax transformation) i.e. NOT PROBABILITIES
    - odds_fighter_1 (np.array): Betting odds for Fighter 1
    - odds_fighter_2 (np.array): Betting odds for Fighter 2
    - return_loss (bool): If True, returns the actual EV loss for monitoring
    - for_feval (bool): If True, the function returns a tuple. This is for the xgb.train function
    - input_probs (bool): Set to True if y_pred is an array of probabilities rather than logits

    Returns:
    - grad (np.array): First derivative of the loss function
    - hess (np.array): Second derivative (Hessian) for XGBoost
    """
    # pdb.set_trace()
    
    # Convert from dmatrix if needed
    try:
        y_true = y_true.get_label()
    except:
        pass
    
    if for_feval:
        np.savetxt("E:/predictions.csv", y_pred, delimiter=",")
        np.savetxt("E:/y_true.csv", y_true, delimiter=",")
    #     probs = y_pred # Using the function for evaluation in xgb.train seems to pass probabilities through rather than logits
    # else:
    if input_probs:
        probs = y_pred.copy()
    else:
        # Convert logits into probabilities using softmax
        exp_preds = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))  # Stability trick
        probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

    # np.savetxt("E:/odds_fighter_1.csv", odds_fighter_1, delimiter=",")

    # Extract individual probabilities
    p_fighter_1 = probs[:, 2]  # Probability of Fighter 1 win
    p_draw = probs[:, 1]  # Probability of Draw
    p_fighter_2 = probs[:, 0]  # Probability of Fighter 2 win

    # Expected value calculations
    ev_fighter_1 = p_fighter_1 * (odds_fighter_1 - 1) - p_fighter_2 - p_draw
    ev_fighter_2 = p_fighter_2 * (odds_fighter_2 - 1) - p_fighter_1 - p_draw
    ev_draw = -np.ones_like(y_true)  # Draw results in a loss of stake

    # Now calculate the actual outcome
    ev_fighter1_actual = np.where(y_true == 2, odds_fighter_1 - 1, -1)
    ev_fighter2_actual = np.where(y_true == 0, odds_fighter_2 - 1, -1)
    ev_draw_actual = np.where(y_true == 1, 0, -1)

    # Select EV based on true outcome difference to the predicted EV
    # Removed draw from EV as will never be betting on it and have no odds for it
    ev_loss = np.stack((
        (ev_fighter2_actual - ev_fighter_2) ** 2,
        # (ev_draw_actual - ev_draw) ** 2,
        (ev_fighter1_actual - ev_fighter_1) ** 2
        ), axis = 1)

    # Compute the gradient and Hessian (vectorized)
    
    # Calculate errors (difference between actual and predicted EV)
    error_fighter_2 = ev_fighter_2 - ev_fighter2_actual
    # error_draw = ev_draw - ev_draw_actual
    error_fighter_1 = ev_fighter_1 - ev_fighter1_actual
    
    # Gradients for each class. Got these from chatgpt differentiation
    grad_fighter_2 = 2 * error_fighter_2 * p_fighter_2 * (1 - p_fighter_2)
    # grad_draw = 2 * error_draw * p_draw * (1 - p_draw)
    grad_draw = np.zeros(len(p_fighter_1))
    grad_fighter_1 = 2 * error_fighter_1 * p_fighter_1 * (1 - p_fighter_1)
    
    # Combine gradients into a single matrix
    grad = np.stack([grad_fighter_2, grad_draw, grad_fighter_1], axis=1)

    # Hessians for each class
    hess_fighter_2 = 2 * p_fighter_2 * (1 - p_fighter_2) * (1 - 2 * p_fighter_2)
    # hess_draw = 2 * p_draw * (1 - p_draw) * (1 - 2 * p_draw)
    hess_draw = np.zeros(len(p_fighter_1))
    hess_fighter_1 = 2 * p_fighter_1 * (1 - p_fighter_1) * (1 - 2 * p_fighter_1)
    
    # Combine Hessians into a single matrix
    hess = np.stack([hess_fighter_2, hess_draw, hess_fighter_1], axis=1)

    # Get the overall loss
    ev_loss = np.array(ev_loss, dtype=np.float64)
    overall_loss = np.mean(ev_loss)

    if for_feval:
        # print(ev_loss.mean())
        # joblib.dump(overall_loss, 'E:/overall_loss.pkl')
        return 'EV_MSE', overall_loss
    elif return_loss:
        return overall_loss
    return grad, hess

def create_custom_loss(odds_fighter_1, odds_fighter_2):
    """
    Wrapper function to pass odds dynamically to the loss function.
    
    Returns:
    - A custom loss function that can be used in XGBoost training.
    """
    return lambda y_pred, y_true: ev_loss_function(y_pred, y_true, odds_fighter_1, odds_fighter_2)

def ev_loss_function_test(odds_fighter_1, odds_fighter_2):
    """
    Wrapper function to pass odds dynamically to the loss function.
    
    Returns:
    - A custom loss function that can be used in XGBoost training.
    """
    # pdb.set_trace()
    
    return lambda y_pred, y_true: ev_loss_function(y_pred, y_true, odds_fighter_1, odds_fighter_2, for_feval = True)

