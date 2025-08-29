def plot_calibration_curve(y_true, y_pred, n_bins=10):
    """
    Plot a calibration curve for a set of binary outcomes and predicted probabilities.

    Args:
        y_true (array-like): True binary outcomes (0 or 1).
        y_pred (array-like): Predicted probabilities for the positive class.
        n_bins (int, optional): Number of bins to divide predictions into. Default = 10.

    Returns:
        matplotlib.figure.Figure: The created Matplotlib figure, which can be shown,
                                  saved, or logged to MLflow.
    
    Example:
        fig = plot_calibration_curve(y_true, y_pred)
        mlflow.log_figure(fig, "calibration_curve.png")
    """
    
    # pdb.set_trace()
    
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy="uniform")

    fig, ax = plt.subplots()
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration Curve")
    ax.legend()

    return fig