def plot_roi_and_sample_size(df):
    """
    Plots ROI_on_constant_bets as a bar chart and Sample_Size as a line chart 
    on a secondary axis, using the DataFrame index as x-axis labels.
    
    Args:
        df (pd.DataFrame): Must contain 'ROI_on_constant_bets' and 'Sample_Size'
                           with index as group labels.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar chart for ROI
    ax1.bar(df.index.astype(str), df['ROI_on_constant_bets'], color='skyblue', label='ROI (%)')
    ax1.set_xlabel("Group")
    ax1.set_ylabel("ROI on Constant Bets (%)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticklabels(df.index.astype(str), rotation=45, ha='right')

    # Secondary axis for sample size
    ax2 = ax1.twinx()
    ax2.plot(df.index.astype(str), df['Sample_Size'], color='red', marker='o', label='Sample Size')
    ax2.set_ylabel("Sample Size", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Title and layout
    fig.suptitle("ROI and Sample Size by Group", fontsize=14)
    fig.tight_layout()

    return fig