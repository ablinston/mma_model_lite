# This function applies filters to the data for records we aren't interested in modelling and testing.

def standard_data_filter(df):

    filtered_df = df[(df['womens'] == 0) & 
                     (df['fighter1_hist_fights'] > 3) &
                     (df['fighter2_hist_fights'] > 3)]

    return filtered_df