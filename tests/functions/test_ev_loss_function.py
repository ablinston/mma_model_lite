
# Load all the packages and functions needed
exec(open('global.py').read())

def test_ev_loss_function():
    # Create dummy data for testing
    y_true = np.array([2, 0, 1, 2, 0])  # Fighter 1 win (2), Fighter 2 win (0), Draw (1)
    y_pred = np.array([
        [0.1, 0.3, 0.6],  # Probabilities (raw logits, for example)
        [0.2, 0.4, 0.4],
        [0.05, 0.8, 0.15],
        [0.3, 0.4, 0.3],
        [0.4, 0.4, 0.2]
    ])
    odds_fighter_1 = np.array([2.5, 3.0, 1.8, 2.0, 2.2])  # Example odds for Fighter 1
    odds_fighter_2 = np.array([1.5, 2.8, 2.2, 1.6, 2.0])  # Example odds for Fighter 2

    # Call the function without returning loss
    grad, hess = ev_loss_function(y_pred, y_true, odds_fighter_1, odds_fighter_2, return_loss=False)

    # Check gradients and Hessians are of correct shape (same as y_pred)
    assert grad.shape == y_pred.shape
    assert hess.shape == y_pred.shape

    # Call the function with return_loss=True to also get the actual loss
    loss = ev_loss_function(y_pred, y_true, odds_fighter_1, odds_fighter_2, return_loss=True)

    # Check that loss is a scalar
    assert np.issubdtype(type(loss), np.floating)

    # Check that for more accurate predictions, the loss is lower. (columns are fighter2 win, draw, fighter1 win respectively)
    y_pred_improved = np.array([
        [0.1, 0.1, 0.8],  # Probabilities (raw logits, for example)
        [0.4, 0.3, 0.3],
        [0.05, 0.8, 0.15],
        [0.3, 0.3, 0.4],
        [0.6, 0.3, 0.1]
    ])
    
    # Call the function with return_loss=True to also get the actual loss
    loss_improved = ev_loss_function(y_pred_improved, y_true, odds_fighter_1, odds_fighter_2, return_loss=True)
    
    assert loss_improved < loss
    
