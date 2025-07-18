# This function applies filters to the data for records we aren't interested in modelling and testing.

def standard_data_filter(df, include_upcoming = False, keep_rows_with_flag = False):

    # pdb.set_trace()    

    # Create an empty flag to exception reasons
    df['notformodel_exclusion_reason'] = ''
    
    # Remove any unrealistic odds
    df.loc[(df['fighter1_true_closing_odds'] >= 15) | (df['fighter2_true_closing_odds'] >= 15), 'notformodel_exclusion_reason'] = 'Odds out of range'
    
    df.loc[(df['fighter1_hist_fights'] <= 2) | (df['fighter2_hist_fights'] <= 2), 'notformodel_exclusion_reason'] = 'Fights < 3'

    # Find anyone with missing true odds
    df.loc[(df['fighter1_true_closing_odds'].isna()) | (df['fighter2_true_closing_odds'].isna()), 'notformodel_exclusion_reason'] = 'Missing odds'

    df.loc[(df['womens'] == 1), 'notformodel_exclusion_reason'] = 'Womens'

    if not keep_rows_with_flag:
        df = df[df['notformodel_exclusion_reason'] == ''].drop(columns = ['notformodel_exclusion_reason']).copy()

    # filtered_df = df[(df['womens'] == 0) & 
    #                  (df['fighter1_hist_fights'] > 2) &
    #                  (df['fighter2_hist_fights'] > 2)]
    # # Remove any data were odds are missing
    # filtered_df = filtered_df[(~filtered_df['fighter1_true_closing_odds'].isna()) & (~filtered_df['fighter2_true_closing_odds'].isna())]
    
    # # Remove any unrealistic odds
    # filtered_df = filtered_df[(filtered_df['fighter1_true_closing_odds'] < 15) & (filtered_df['fighter2_true_closing_odds'] < 15)]
    
    if not include_upcoming:
        df = df[(~df['outcome'].isna())]
        
    return df
