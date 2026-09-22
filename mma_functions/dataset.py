from ablinston_ml import stop_if_not
import fastparquet as fp
import numpy as np
import random
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample

from mma_functions import cleaning, evaluation
from mma_functions.config import config

import pandas as pd
import polars as pl

# This function adds formatted names to a data frame based on a cleaned fighter column

def add_formatted_fighter_names(df, on = ['fighter'], formatted_col_names = ['Fighter'], config = config):
    
    # pdb.set_trace()
    
    # Get formatted names data
    formatted_names = fp.ParquetFile('./data/' + config['fighter_names_file_name']).to_pandas()
    
    new_df = df.copy()
    
    for i in range(0,len(on)):
        # Merge on the names
        new_df = new_df.merge(formatted_names, left_on = on[i], right_on = 'fighter', how = 'left')
        
        # Rename the fighter column
        new_df[formatted_col_names[i]] = new_df['Fighter'].copy()
        new_df.drop(columns = ['Fighter'], inplace = True)
        # new_df.rename(columns = {'fighter': formatted_col_name[i]}, inplace = True)
    
    # Check rows correct
    stop_if_not(len(new_df) == len(df))
    
    return new_df

def add_opponent_column(df):
    """
    Adds an 'opponent' column to the DataFrame based on the 'fighter' and 'bout' columns.
    Assumes 'bout' is in 'fighter vs. opponent' or 'opponent vs. fighter' format.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing 'fighter' and 'bout' columns.

    Returns:
    pd.DataFrame: The DataFrame with an added 'opponent' column.
    """
    # pdb.set_trace()
    
    # Split 'bout' into two columns, 'fighter_1' and 'fighter_2'
    fighter_pairs = df['bout'].str.split('vs\\.', expand=True)
    df['fighter_1'] = fighter_pairs[0]
    df['fighter_2'] = fighter_pairs[1]

    # Use np.where to assign 'opponent' based on whether 'fighter' matches 'fighter_1' or 'fighter_2'
    df['opponent'] = np.where(df['fighter'].str.replace(' ', '') == df['fighter_1'].str.replace(' ', ''), df['fighter_2'], df['fighter_1'])

    # Drop the temporary columns used for the split
    df.drop(columns=['fighter_1', 'fighter_2'], inplace=True)
    
    return df

# Example usage
# new_odds_df = add_opponent_column(new_odds_df)
# print(new_odds_df[['fighter', 'bout', 'opponent']])

# This function takes the fight stats and upcoming fight details data and adds empty upcoming fight stats onto the fight stats data frame
# for the upcoming fights

def add_upcoming_to_fight_stats(fight_stats, upcoming_fight_details_df, events, upcoming_events):
        
    # Add upcoming events
    all_events = pd.concat([upcoming_events, events], axis = 0)

    # Add the date to the upcoming fights
    upcoming_fight_details = upcoming_fight_details_df.merge(all_events[['event', 'date']], on = 'event', how = 'left')

    # Create empty stats rows for each fight
    upcoming_fight_details[['fighter1', 'fighter2']] = upcoming_fight_details['bout'].str.split('vs\\.', expand = True)
    upcoming_fight_stats = pd.concat([
        upcoming_fight_details[['event', 'bout', 'date', 'fighter1']].rename(columns = {'fighter1': 'fighter'}),
        upcoming_fight_details[['event', 'bout', 'date', 'fighter2']].rename(columns = {'fighter2': 'fighter'})
        ])
    
    return pd.concat([upcoming_fight_stats, fight_stats])
    
    

def assign_opponents(df):
    """
    Assigns opponent to each fighter in the DataFrame.
    """
    df = df.copy().reset_index(drop = True)
    df["opponent"] = df["fighter"].shift(-1)
    df.loc[df.index % 2 == 1, "opponent"] = df["fighter"].shift(1)
    return df

# This function takes a pd series of bouts (without cleaning so WITH spaces) and transforms it into a unique
# list of fighters fighting in those bouts

def fighter_list_from_bouts(bout_series):
    
    # bout_series = upcoming_fight_details_df['BOUT']
    
    df = pd.DataFrame()
    
    # First split the bouts
    df[['f1', 'f2']] = bout_series.str.split(' vs\\. ', expand = True)
    
    return list(set(df['f1'].to_list() + df['f2'].to_list()))

# This function takes a data frame and flags the data with train/test/oot for a given config

def flag_train_test_oot(data, config, test_fighter_list = []):
    
    # # pdb.set_trace()
    # data = df.copy()
    
    # Flag the oot sample and set the rest to train
    data['train_test_oot_flag'] = np.where(
        (data['date'] >= config['min_training_date']) & (data['date'] < config['oot_start_date']),
        'train',
        np.where(data['date'] >= config['oot_start_date'], 'oot', pd.NA)
    )
    
    # Get a separate data frame with only non-oot fighters
    data_without_oot = data[data['train_test_oot_flag'] == 'train'][['fighter1', 'fighter2']].copy()
        
    # Get a list of all fighters to select the test data
    if len(test_fighter_list) == 0:
        random.seed(config['seed_all'] + 45)
        
        ## Generate a random number for each fighter in a way that keeps them consistent when new ones are added
        # First generate one for every fight, then drop duplicates to take the first random number allocated for each fighter. Need two sets so that fighter2 always starts at the same point too
        
        rand_set_1 = np.random.rand(len(data_without_oot))
        
        random.seed(config['seed_all'] + 97502)
        rand_set_2 = np.random.rand(len(data_without_oot))
        
        rands = pd.DataFrame({'fighter': data_without_oot['fighter1'].tolist() + data_without_oot['fighter2'].tolist(),
                             'rand': np.concatenate((rand_set_1, rand_set_2))}).drop_duplicates(subset = ['fighter'])
        
        # Set test fighters as those below a certain threshold, multiplied by a factor to get roughly the right proportion of fights
        test_fighter_list = rands[rands['rand'] < config['test_proportion'] * 0.35]['fighter'].to_list()
    
    # pdb.set_trace()
    # with open("./configs/test_fighters.yaml", "w") as file:
    #     dump(test_fighters, file, default_flow_style=False)
    
    # test_fighters = random.sample(
    #     list(set(data_without_oot['fighter1'].tolist() + data_without_oot['fighter2'].tolist())),
    #     int(config['test_proportion'] * len(data_without_oot) * 0.15) # multiply by this to get roughly the correct number of rows as test_size wishes
    # )
    
    # Set the test fighters to test
    data['train_test_oot_flag'] = np.where(
        (data['train_test_oot_flag'] == 'train') & (data['fighter1'].isin(test_fighter_list) | data['fighter2'].isin(test_fighter_list)),
        'test',
        data['train_test_oot_flag']
    )
    
    # Check the values
    print(data['train_test_oot_flag'].value_counts(normalize = True))
    
    # Check the backtesting metrics of the chosen sample
    train_df = data[(data['train_test_oot_flag'] == 'train') & ~(data['outcome'].isna()) & ~(data['fighter1_backtesting_odds'].isna()) & ~(data['fighter2_backtesting_odds'].isna())]
    
    test_df = data[(data['train_test_oot_flag'] == 'test') & ~(data['outcome'].isna()) & ~(data['fighter1_backtesting_odds'].isna()) & ~(data['fighter2_backtesting_odds'].isna())]
    
    backtesting_metrics_train = evaluation.calculate_metrics_odds(train_df, cleaning.clean_target_column(train_df['outcome']))
    backtesting_metrics_test = evaluation.calculate_metrics_odds(test_df, cleaning.clean_target_column(test_df['outcome']))
    
    print(f"Train odds logloss: {backtesting_metrics_train['X Odds Logloss']}")
    print(f"Test odds logloss: {backtesting_metrics_test['X Odds Logloss']}")
    
    del [train_df, test_df]
    
    # Return the data
    return data

# This function sets consistent column formats for modelling

def format_cols_for_model(data):
   
    # Convert all dates to integers
    date_cols = [col for col in data.columns if data.dtypes[col] == 'datetime64[ns]']
    data[date_cols] = data[date_cols].astype('int64')
    
    return data

# This function takes a list and swaps 'fighter1' and 'fighter2' around.

def list_swap_fighter1_fighter2(swap_list):
    
    new_list = pd.Series(swap_list).str.replace('fighter2', 'fighterX')
    new_list = new_list.str.replace('fighter1', 'fighter2')
    new_list = new_list.str.replace('fighterX', 'fighter1')
    
    return new_list.tolist()

def merge_df_both_fighters(left_df: pd.DataFrame, df_list: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge fighter-related dataframes onto a main dataframe using Polars for efficiency.

    This function:
    1. Converts the input Pandas DataFrames to Polars DataFrames.
    2. Concatenates all fighter1 merges into one big table.
    3. Concatenates all fighter2 merges into one big table.
    4. Performs only two joins in Polars (one for fighter1, one for fighter2).
    5. Converts the result back into a Pandas DataFrame.

    Parameters
    ----------
    left_df : pd.DataFrame
        The base dataframe containing columns:
        ['event', 'bout', 'fighter1', 'fighter2'].
        This is the "main" dataframe we will merge onto.

    df_list : list[pd.DataFrame]
        A list of dataframes to merge. Each dataframe must contain:
        ['event', 'bout', 'fighter'] plus any additional columns.

    Returns
    -------
    pd.DataFrame
        A new Pandas DataFrame with the merged fighter1 and fighter2 data.

    Notes
    -----
    - Uses Polars under the hood for faster, more memory-efficient joins.
    - Reduces memory pressure compared to Pandas by doing only 2 joins total,
      instead of 2 per dataframe.
    - Ideal for low-memory environments like Raspberry Pi.
    """

    # Convert left_df to Polars
    left_pl = pl.from_pandas(left_df)

    for df in df_list:
        if len(df) == 0:
            continue

        df_pl = pl.from_pandas(df)

        # Fighter 1 prefixed version
        f1 = df_pl.rename({c: f"fighter1_{c}" for c in df_pl.columns}).rename({
            "fighter1_event": "event", "fighter1_bout": "bout", "fighter1_fighter":"fighter1"})

        # Fighter 2 prefixed version
        f2 = df_pl.rename({c: f"fighter2_{c}" for c in df_pl.columns}).rename({
            "fighter2_event": "event", "fighter2_bout": "bout", "fighter2_fighter":"fighter2"})
        
        # Merge fighter1 data
        left_pl = left_pl.join(
                f1,
                on=["event", "bout", "fighter1"],
                how="left",
            )

        # Merge fighter2 data
        left_pl = left_pl.join(
                f2,
                on=["event", "bout", "fighter2"],
                how="left",
            )
        
        del [f1, f2]
        
    # Convert back to Pandas
    return left_pl.to_pandas()

# This function takes a dataset and prepares the data for model training by splitting it into train and test

# It returns 7 objects that can be output into env objects like
# X_train, X_test, y_train, y_test, y_test_bin, no_of_classes, oot = prep_data_for_model_training(data)

def prep_data_for_model_training(df, split_or_other_flag = False, upcoming_fights = False, oot_testing = False):
    
    # pdb.set_trace()
        
    data = cleaning.clean_inf_and_large(df.copy())
    
    # If we're prepping upcoming fights, take whole sample
    if oot_testing or upcoming_fights:
        data['train_test_oot_flag'] = 'train'
    
    # Filter the OOT data and training data to be ignored
    oot = data[data['train_test_oot_flag'] == 'oot']
    data = data[data['train_test_oot_flag'] != 'oot']
        
    # Create a flag for exclusion of split decision and other in training data if required
    if (split_or_other_flag):
        data['notformodel_split_dec_or_other'] = np.where(data['method'].isin(['Decision-Split', 'Other']), 1, 0)

    # Remove column that has results, or that contain information we don't want included.
    data.drop(columns = ['method', 'index', 'level_0', 'date'], inplace = True, errors = 'ignore')
    data.rename(columns = {'fighter1': 'notformodel_fighter1', 'fighter2': 'notformodel_fighter2', 'train_test_oot_flag': 'notformodel_train_test_oot_flag'}, inplace = True)
    
    # Set target
    target = 'outcome'
    
    if not upcoming_fights:
        # Recode the target. Don't need to distinguish between D and NC
        data[target] = np.where((data[target] == "D/D") | (data[target] == "NC/NC"), "OTH", data[target])
    
    data = format_cols_for_model(data)
    
    # Convert object columns into series' of dummies
    obj_cols = [col for col in data.columns if data.dtypes[col] == 'object' and 'notformodel' not in col]
    try:
        obj_cols.remove(target)
    except:
        pass
    data = pd.get_dummies(data, columns = obj_cols)
    
    if not upcoming_fights:
        # Convert the target into a labelled column ready for multiclass modelling
        data[target] = cleaning.clean_target_column(data[target])
            
    X_train = data[data['notformodel_train_test_oot_flag'] == 'train'].drop(columns = [target])
    y_train = data[data['notformodel_train_test_oot_flag'] == 'train'][target]
    X_test = data[data['notformodel_train_test_oot_flag'] == 'test'].drop(columns = [target])
    y_test = data[data['notformodel_train_test_oot_flag'] == 'test'][target]
    
    if not upcoming_fights and len(X_test) > 0:
        # Convert the test output to a binarised data frame. Can help with AUC calculations
        y_test_bin = label_binarize(y_test, classes = np.unique(y_test))
    
        # # Exclude split decisions and other if required
        # if exclude_split_other_from_train:
        #     X_train = X_train[X_train['split_dec_or_other'] == 0]
        #     X_train.drop(columns = ['split_dec_or_other'], inplace = True)
        
        # Check data types
        data.dtypes.value_counts()
    else:
        y_test_bin = None
    
    # Set number of classes for training
    no_of_classes = len(set(y_train))
    
    
    # Return the objects needed as a list
    return [X_train, X_test, y_train, y_test, y_test_bin, no_of_classes, oot]

# This removes duplicates where a fighter appears in a data frame on the same date more than once, as either fighter1 or fighter2

def remove_duplicate_fighter_pairs(df):
    """
    Removes duplicate rows from the DataFrame where each fighter appears only once per date,
    considering both 'fighter1' and 'fighter2' columns.

    Parameters:
    - data (dict): Dictionary containing 'date', 'fighter1', and 'fighter2' columns as lists.

    Returns:
    - pd.DataFrame: Cleaned DataFrame with duplicates removed.
    """
    # pdb.set_trace()
    
    # Initialize a set to track unique fighter-date pairs
    seen_fighters = set()
    rows_to_keep = []

    # Iterate over each row and check if fighters have appeared on that date
    for idx, row in df.iterrows():
        date = row['date']
        fighter1 = row['fighter1']
        fighter2 = row['fighter2']

        # Create unique keys for each fighter with the date
        key1 = (date, fighter1)
        key2 = (date, fighter2)

        # Check if both fighters are new for this date
        if key1 not in seen_fighters and key2 not in seen_fighters:
            rows_to_keep.append(idx)  # Keep this row
            seen_fighters.add(key1)
            seen_fighters.add(key2)

    # Create a cleaned DataFrame with only the rows we decided to keep
    df_cleaned = df.loc[rows_to_keep].reset_index(drop=True)

    return df_cleaned

# This function takes a data frame and selects a random sample of rows, but it does this by randomly selecting fighters then
# selecting all their fights for the data.

# rows_with_bagging is how many rows the final data should have after bagging 

def sample_fighters(df_complete = None, X = None, y = None, n_rows = 100, rows_with_bagging = None):
    
    # pdb.set_trace()
    
    if df_complete is None:
        df = X.copy()
        df['y'] = y
    else:
        df = df_complete.copy()
    
    # Create a series of all the fighters
    fighters = pd.concat([df['notformodel_fighter1'], df['notformodel_fighter2']])

    # Count how many fights they have and shuffle
    fighters_counts = pd.DataFrame(fighters.value_counts().sample(frac = 1)).reset_index()
    
    # Now do cumulative sums until I get the required rows
    fighters_counts['cumsums'] = fighters_counts['count'].cumsum()
    
    # Create a flag for which fighters will be in the selected data. Multiply by 1.1 because there will be overlap in these fighters fighting each other
    fighters_counts['select_flag'] = np.where(fighters_counts['cumsums'] < (n_rows * 1.2), 1, 0)

    # Get the selected fighters
    selected_fighters = fighters_counts[fighters_counts['select_flag'] == 1]['index'].tolist()
    
    final_df = df[df['notformodel_fighter1'].isin(selected_fighters) | df['notformodel_fighter2'].isin(selected_fighters)]

    # Do a check to see if n rows is reasonable
    if abs(len(final_df) - n_rows) > (n_rows * 0.05):
        print(f"WARNING: sample_fighters() function returned a row number > 5% different from target. Sample is {(len(final_df) / n_rows) - 1} of data.")

    # If doing bagging, expand the sample
    if rows_with_bagging is not None:
        final_df = resample(final_df, replace = True, n_samples = rows_with_bagging)

    # Now return the selected data
    if df_complete is None:
        # Return an X and y
        return final_df.drop(columns = 'y'), final_df['y']
    else:
        return final_df

# This function removes columns from training data that seem arbitrary and could lead to over-fitting

def standard_feature_removal(df, open_odds_only, differentials_only = False, true_odds_only = False):

    data = df.copy()
    
    # pdb.set_trace()
    
    # Remove cols that shouldn't really affect the fight.
    # Remove dates because they can overfit. Leave date minuses, which are days since
    cols_to_remove = [col for col in data.columns if ('date' in col) & (('diff' not in col) & ('minus' not in col) & (col != 'date'))]

    # Remove event
    cols_to_remove = cols_to_remove + ['event', 'bout']
    
    # Get rid of dob and reach. Seem arbitrary
    cols_to_remove = cols_to_remove + ['fighter1_dob', 'fighter2_dob', 'fighter1_reach', 'fighter2_reach', 'diff_reach', 'fighter1_open_odds', 'fighter2_open_odds']

    # # Remove cols that are just totals for a fighter, like total kicks etc. Leave percentages and ratios and per minute stats
    cols_to_remove = cols_to_remove + [col for col in data.columns if ('_round' in col) and not (('percent' in col) or ('ratio' in col) or ('per_minute' in col) or ('per_fight' in col) or ('bayes' in col) or ('adj' in col))]
    cols_to_remove = cols_to_remove + [col for col in data.columns if ('allrounds' in col) and not (('percent' in col) or ('ratio' in col) or ('per_minute' in col) or ('per_fight' in col) or ('bayes' in col) or ('adj' in col) or (('against' in col) and (('hist_round' in col) or ('hist_last_3_fights' in col))))]
    # Remove the columns that are single results
    cols_to_remove = cols_to_remove + [col for col in data.columns if ('result_' in col)]

    # Now we want to remove any column that is not a differential or a comparison between fighters
    # This bit is optional. Just testing.
    if differentials_only:
        cols_to_remove = cols_to_remove + [col for col in data.columns if ('diff' not in col) and ('fight_elevation' not in col) and ('weightclass' not in col) and ('matchup' not in col)]

    if true_odds_only:
        # Remove any odds that aren't true
        cols_to_remove += [col for col in data.columns if 'odds' in col and 'true' not in col and 'backtesting' not in col] 
    
    # Remove columns not wanted but select specific ones to keep
    data = data[[col for col in data.columns if not col in cols_to_remove or col in ['notformodel_split_dec_or_other']]]

    return data

# This function takes a list of columns, from the random forest, and tries to remove duplicates or very similar columns.
# The list should be ordered in terms of priority to keep the columns, with name 0 being the highest priority.

def trim_column_list(column_list):
    
    # pdb.set_trace()
    
    # First create a new version and replace any bayes and adjust names
    trunc_column_series = pd.Series(column_list).str.replace(r'_ratio|_bayes', '', regex = True).replace(r'_adjsum|_adjmean', '_adj', regex = True).replace(r'last_3_fights|last_2_fights|last_1_fight', 'last_x_fight', regex = True)
    
    # Now return the original list but without any duplicates across those strings
    
    return  pd.Series(column_list)[(~trunc_column_series.duplicated()).tolist()].tolist()
