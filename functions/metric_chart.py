# This function creates a seaborn chart to track a metric. It returns two objects, the figure and axes

# x and y should be pandas Series

def metric_chart(x, y_model, y_odds, x_label, y_label):
    
    # pdb.set_trace()
    
    # Set Seaborn style
    sns.set_theme(style="whitegrid")  # Clean background with grid
    
    # Put the model and odds values into a single data frame
    chart_df = pd.DataFrame({
        x_label: x.tolist() * 2,
        y_label: y_model.tolist() + y_odds.tolist(),
        'Source': ['Model'] * len(y_model) + ['Odds'] * len(y_odds)})
    
    # Create the figure and plot
    fig, ax = plt.subplots(figsize = (6, 5))
    sns.barplot(x = chart_df[x_label], y = chart_df[y_label], hue = chart_df['Source'], ax = ax, palette = ["lightblue", "orange"])
    
    # Set labels and title
    ax.set_ylabel(y_label)
    
    # Rotate x-axis labels (Seaborn handles spacing better)
    ax.set_xticklabels(x, rotation=90)
    
    # Adjust layout to prevent text cutoff
    plt.tight_layout() 
    
    return fig, ax
