# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 13:16:14 2025

@author: Andy
"""

te = pd.concat([merged_data[(merged_data['date'] > '2016-02-01') & (merged_data['fighter1_betfair_exch_odds'].isna())]['fighter1'].value_counts(),  merged_data[(merged_data['date'] > '2016-02-01') & (merged_data['fighter2_betfair_exch_odds'].isna())]['fighter2'].value_counts()], axis = 0).sort_values()


merged_data['fighter1_open_odds'].mean()


backtesting_data[backtesting_data['fighter1'] == 'michaelchandler']['fighter1_dob']
