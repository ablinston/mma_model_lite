# This function applies filters to the data for records we aren't interested in modelling and testing.

def standard_data_filter(df):

    filtered_df = df[(df['womens'] == 0) & 
                     (df['fighter1_hist_fights'] > 2) &
                     (df['fighter2_hist_fights'] > 2)]
    
    # Remove any data were odds are missing
    filtered_df = filtered_df[(~filtered_df['fighter1_best_closing_odds'].isna()) & (~filtered_df['fighter2_best_closing_odds'].isna()) & (~filtered_df['outcome'].isna())]

    return filtered_df