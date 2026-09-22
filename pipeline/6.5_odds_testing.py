# This script is to look at the fighter odds and event odds to determine which to use

from sklearn.linear_model import LinearRegression

# imports
from ablinston_ml import stop_if_not
import fastparquet as fp
import joblib
import numpy as np
import pandas as pd

from mma_functions.odds import american_odds_to_decimal
from mma_functions.config import config, odds_scrape_config

# Get the two datasets
fight_odds = fp.ParquetFile('./data/' + config['fighter_odds_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()
event_odds = fp.ParquetFile('./data/' + config['event_odds_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()

event_odds = event_odds.drop_duplicates(subset = ['fighter', 'event', 'bout'])

# Merge the datasets
all_odds = fight_odds.merge(event_odds.drop(columns = ['date', 'opponent']), on = ['fighter', 'event', 'bout'], how = 'left')

# Check rows in data
stop_if_not(len(fight_odds) == len(fight_odds))

# Look at how many cols are na
event_odds_cols = odds_scrape_config['events_colnames']
all_odds[event_odds_cols].isna().sum(axis = 1).value_counts()
all_odds[event_odds_cols].isna().sum(axis = 1).value_counts(normalize = True)

# Get all the fighters where we have at least 4 non-NA odds
all_complete_odds_info = all_odds[all_odds[event_odds_cols].isna().sum(axis=1) <= 3]

# COnvert all odds to decimal
all_complete_odds_info.loc[:, ['closing_odds_low', 'closing_odds_high'] + event_odds_cols] = all_complete_odds_info.loc[:, ['closing_odds_low', 'closing_odds_high'] + event_odds_cols].applymap(american_odds_to_decimal)

# Compare the averages to the best_closing_odds
all_complete_odds_info.loc[:, 'closing_odds_mean'] = np.nanmean(all_complete_odds_info[event_odds_cols], axis = 1)

# Check where they are outside the range. This might happen where a fight has happened twice
all_complete_odds_info['outside_range'] = np.where(all_complete_odds_info['closing_odds_mean'] > all_complete_odds_info['closing_odds_high'],
                                                   1,
                                                   np.where(all_complete_odds_info['closing_odds_mean'] < all_complete_odds_info['closing_odds_low'],
                                                                                                      1, 0))

all_complete_odds_info['outside_range'].value_counts(normalize = True)

# Select only those within range
all_complete_odds_info = all_complete_odds_info[all_complete_odds_info['outside_range'] == 0]

# Fit a regression to see how far from the mid point of the high and low the average has been historically
lm = LinearRegression(fit_intercept = False)
lm.fit(X = all_complete_odds_info[['closing_odds_low', 'closing_odds_high']], y = all_complete_odds_info['closing_odds_mean'])

# Summarise the coefficients
coefficients = pd.Series(lm.coef_, index=['closing_odds_low', 'closing_odds_high'])
coefficients
lm.intercept_

# Testing
all_complete_odds_info['estimated_closing_odds_mean'] = lm.predict(all_complete_odds_info[['closing_odds_low', 'closing_odds_high']])

# Save the features for future reference
lm.features = ['closing_odds_low', 'closing_odds_high']

# Output this model to use later to impute missing average odds
joblib.dump(lm, './results/average_odds_model.joblib', compress = 9)


######################
## Randomised odds both directions


# Work out the differences between high and lows at closing to get a rough idea how much odds vary leading up to a fight
rands = np.random.uniform(0, 1, len(fight_odds))

# Take random swapping of odds to try to get differences mean 0
odds_1 = np.vectorize(american_odds_to_decimal)(np.where(rands > 0.5, fight_odds['closing_odds_low'], fight_odds['closing_odds_high']))
odds_2 = np.vectorize(american_odds_to_decimal)(np.where(rands <= 0.5, fight_odds['closing_odds_low'], fight_odds['closing_odds_high']))

# Work out mean and std of population
np.mean(odds_1 / odds_2) 
np.std(odds_1 / odds_2, ddof = 0) # to get population rather than sample std


######################
## Randomised odds one directions

# Take random swapping of odds to try to get differences mean 0
odds_1 = np.vectorize(american_odds_to_decimal)(fight_odds['closing_odds_low'])
odds_2 = np.vectorize(american_odds_to_decimal)(fight_odds['closing_odds_high'])

# Work out mean and std of population
np.mean(odds_1 / odds_2) 
np.std(odds_1 / odds_2, ddof = 0) # to get population rather than sample std

np.random.normal(0.96123, 0.051772, 100)
