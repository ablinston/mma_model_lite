# This script fits XGBoost models to the final data

import gc
import h2o
from h2o.automl import H2OAutoML
from hyperopt import hp, STATUS_OK, tpe, fmin, Trials # hyperparameter optimisation functions
from hyperopt.atpe import suggest as atpe_suggest
import itertools
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import pickle
from sklearn.model_selection import train_test_split, KFold
import seaborn as sns
import xgboost as xgb

# Read in global objects and packages
exec(open('global.py').read())

# Set the model location
# model_path = './results/final_xgboost_forest.joblib'
model_path = './models/xgb_forest_v2.joblib'

# # Set the version number to be used in outputs
# new_version_number = ''

# Read in the data for training
backtesting_data = fp.ParquetFile('./data/final_dataset/final_dataset.parquet').to_pandas() # pq.ParquetDataset('./data/final_dataset/').read().to_pandas()

# Load files
rf_xgb_model = joblib.load(model_path)
# rf_xgb_model = mlflow.pyfunc.load_model('models:/xgb_forest_model/1')._model_impl.python_model.model


# Get an individual fight
backtesting_data = backtesting_data[(backtesting_data['date']=='2025-07-26') & (backtesting_data['fighter1']=='nikitakrylov')]

backtesting_data_dup = backtesting_data.copy()
# Edit the odds

backtesting_data['fighter1_betfair_exch_odds']
backtesting_data['fighter2_betfair_exch_odds']

backtesting_data_dup['fighter1_betfair_exch_odds'] = 1.56
backtesting_data_dup['fighter2_betfair_exch_odds'] = 2.74



# Recalculate all true odds
backtesting_data_dup = calculate_all_true_odds_columns(backtesting_data_dup, impute_missing_betfair = data_processing_config['impute_missing_betfair'], odds_suffix_for_true_odds = data_processing_config['upcoming_odds_suffix_for_true_odds'], impute_one_sided_missing_odds = data_processing_config['impute_one_sided_missing_odds'])
 
if (data_processing_config['impute_missing_betfair'] == True) & (model_hyper_params['odds_to_use'] == 'betfair_exch_odds'):
    model_hyper_params['odds_to_use'] = 'betfair_exch_odds_imputed'

# Add what odds are to be used in model training and backtesting
backtesting_data_dup['fighter1_backtesting_odds'] = backtesting_data_dup['fighter1_' + model_hyper_params['odds_to_use']].copy()
backtesting_data_dup['fighter2_backtesting_odds'] = backtesting_data_dup['fighter2_' + model_hyper_params['odds_to_use']].copy()

# Recalculate differentials
backtesting_data_dup = add_fighter_differentials(backtesting_data_dup)


# Combine the fights
backtesting_data = pd.concat([backtesting_data, backtesting_data_dup])

# Check
backtesting_data['fighter1_true_closing_odds'] # Should be different



##########################################################################################
### Data prep for model ###
##########################################################################################

# Get the data ready for the model
X_upcoming, temp1, y_upcoming, temp2, temp3, temp4, temp5 = prep_data_for_model_training(
    backtesting_data,
    upcoming_fights = True)

# Get the model predictions
# prediction_probabilities, temp = rf_xgb_model.predict(X_upcoming)
prediction_probabilities, temp = predict_xgboost_ensemble(rf_xgb_model, X_upcoming, [0] * len(y_upcoming))


gc.collect();

# Summarise the model predictions and betting decisions using the config
betting_data = summarise_model_predictions(
    actuals_data_df = backtesting_data, 
    prediction_probabilities = prediction_probabilities, 
    min_prob_to_bet = bet_config['min_prob_to_bet'], 
    min_ev_to_bet = bet_config['min_ev_to_bet'], 
    min_bet_size = bet_config['min_bet_size'], 
    max_bet_size = bet_config['max_bet_size'],
    exchange_commission = bet_config['exchange_commission'])

# Add the final bet size based on balance
if bet_config['constant_bets'] == False:
    betting_data['Final_bet'] = np.round((betting_data['bets'] * betting_data['bet_size'] * bet_config['start_balance']) / 5) * 5
else:
    betting_data['Final_bet'] = np.round((betting_data['bets'] * betting_data['bet_size']) / 5) * 5

betting_data['Potential_win'] = np.round(betting_data['bets'] * betting_data['Final_bet'] * np.where(betting_data['Bet_trigger'] == betting_data['fighter1'], betting_data['fighter1_backtesting_odds'] - 1, betting_data['fighter2_backtesting_odds'] - 1), 2)

betting_data['bet_assumed_odds'] = betting_data['bets'] * np.where(betting_data['Bet_trigger'] == betting_data['fighter1'], betting_data['fighter1_backtesting_odds'], betting_data['fighter2_backtesting_odds'])

# Round odds columns
for col in [cols for cols in betting_data.columns if 'odds' in cols]:
    betting_data.loc[:, col] = np.round(betting_data[col], 2)

betting_data.rename(columns = {'fighter1_notformodel_true_odds_asat_datetime': 'f1_scrape_date', 'fighter2_notformodel_true_odds_asat_datetime': 'f2_scrape_date'}, inplace = True)

# Flag whether odds Betfair imputed


summary_data = betting_data[['date', 'event', 'f1_scrape_date', 'f2_scrape_date', 'fighter1', 'fighter1_backtesting_odds', 'Fighter1_win_prob', 'Fighter1_win_EV', 'fighter2', 'fighter2_backtesting_odds', 'Fighter2_win_prob', 'Fighter2_win_EV', 'Bet_trigger', 'bet_assumed_odds', 'Bet_breakeven_odds', 'Final_bet', 'Potential_win']]











backtesting_data = fp.ParquetFile('./data/final_dataset/final_dataset.parquet').to_pandas() # pq.ParquetDataset('./data/final_dataset/').read().to_pandas()

# Get an individual fight
backtesting_data = backtesting_data[(backtesting_data['date']=='2025-07-26') & (backtesting_data['fighter1']=='nikitakrylov')]


backtesting_data = pd.concat([backtesting_data, backtesting_data, backtesting_data])
backtesting_data['train_test_oot_flag'] = ['test', 'train', 'oot']

# Prep the data ready for model training as it was done in the overall model training
X_train, X_test, y_train, y_test, y_test_bin, no_of_classes, oot = prep_data_for_model_training(backtesting_data)


##########################################################################################
### Calculating metrics ###
##########################################################################################


        
# Get the model predictions
prediction_probabilities, temp = predict_xgboost_ensemble(rf_xgb_model, X_train, y_train)

# Get the model predictions
prediction_probabilities_test, temp = predict_xgboost_ensemble(rf_xgb_model, X_test, y_test)

# Get the model input data
X_oot, temp1, y_oot, temp2, temp3, temp4, temp5 = prep_data_for_model_training(
    oot,
    oot_testing = True)

# Get the model predictions
prediction_probabilities_oot, temp = predict_xgboost_ensemble(rf_xgb_model, X_oot, y_oot)
        
prediction_probabilities
prediction_probabilities_test
prediction_probabilities_oot








##########################################################################################
# TESTING PREDICTIONS ON ACTUAL DATA AND DATA FROM BEFORE EVENT
##########################################################################################

backtesting_data = fp.ParquetFile('./data/final_dataset/final_dataset.parquet').to_pandas() # pq.ParquetDataset('./data/final_dataset/').read().to_pandas()

backtesting_data_before = fp.ParquetFile('./data/final_dataset/final_dataset.parquet').to_pandas() # pq.ParquetDataset('./data/final_dataset/').read().to_pandas()

event = 'ufcfightnight:imavovvsborralho'

actuals = backtesting_data[backtesting_data['event'] == event]
before = backtesting_data_before[backtesting_data_before['event'] == event].sort_values(by = ['fighter1']).reset_index(drop = True)

# Remove fights not in the before for whatever reason
actuals = actuals[actuals['fighter1'].isin(before['fighter1'].to_list())].sort_values(by = ['fighter1']).reset_index(drop = True)

X_upcoming_actual, temp1, y_upcoming, temp2, temp3, temp4, temp5 = prep_data_for_model_training(
    actuals,
    upcoming_fights = True)

X_upcoming_before, temp1, y_upcoming, temp2, temp3, temp4, temp5 = prep_data_for_model_training(
    before,
    upcoming_fights = True)


prediction_probabilities_actual, temp = predict_xgboost_ensemble(rf_xgb_model, X_upcoming_actual, [0] * len(X_upcoming_actual))
prediction_probabilities_before, temp = predict_xgboost_ensemble(rf_xgb_model, X_upcoming_before, [0] * len(X_upcoming_before))


betting_data_actual = summarise_model_predictions(
    actuals_data_df = actuals, 
    prediction_probabilities = prediction_probabilities_actual, 
    min_prob_to_bet = bet_config['min_prob_to_bet'], 
    min_ev_to_bet = bet_config['min_ev_to_bet'], 
    min_bet_size = bet_config['min_bet_size'], 
    max_bet_size = bet_config['max_bet_size'],
    exchange_commission = bet_config['exchange_commission'])

betting_data_before = summarise_model_predictions(
    actuals_data_df = before, 
    prediction_probabilities = prediction_probabilities_before, 
    min_prob_to_bet = bet_config['min_prob_to_bet'], 
    min_ev_to_bet = bet_config['min_ev_to_bet'], 
    min_bet_size = bet_config['min_bet_size'], 
    max_bet_size = bet_config['max_bet_size'],
    exchange_commission = bet_config['exchange_commission'])


summary_data_actual = betting_data_actual[['date', 'event', 'fighter1', 'fighter1_backtesting_odds', 'Fighter1_win_prob', 'Fighter1_win_EV', 'fighter2', 'fighter2_backtesting_odds', 'Fighter2_win_prob', 'Fighter2_win_EV', 'Bet_trigger', 'Bet_breakeven_odds']]

summary_data_before = betting_data_before[['date', 'event', 'fighter1', 'fighter1_backtesting_odds', 'Fighter1_win_prob', 'Fighter1_win_EV', 'fighter2', 'fighter2_backtesting_odds', 'Fighter2_win_prob', 'Fighter2_win_EV', 'Bet_trigger', 'Bet_breakeven_odds']]

actual_model = actuals.sort_values(by = ['fighter1']).reset_index(drop = True)[final_model_features]
before_model = before.sort_values(by = ['fighter1']).reset_index(drop = True)[final_model_features]

cols_to_check = [col for col in actuals.columns if "diff" not in col and 'bayes' not in col]

# Compare the data going into the model
comparison = actuals.sort_values(by = ['fighter1']).reset_index(drop = True)[cols_to_check].compare(before.sort_values(by = ['fighter1']).reset_index(drop = True)[cols_to_check])

te = comparison.columns

comparison['fighter1_hist_opp_stance_round1_ctrl_minutes']
actuals[['fighter1', 'fighter1_hist_opp_stance_round1_ctrl_minutes']]

before[['fighter1', 'fighter1_hist_opp_stance_round1_ctrl_minutes']]



###############
# Do an ELO check on single fighter

fighter_name = 'andreasgustafsson'

cols_to_see = ['event', 'bout', 'date', 'fighter1', 'fighter2', 'fighter1_stance', 'fighter2_stance', 'fighter1_hist_opp_stance_round1_ctrl_minutes', 'fighter2_hist_opp_stance_round1_ctrl_minutes', 'fighter1_hist_round1_ctrl_minutes', 'fighter2_hist_round1_ctrl_minutes']

brian_before = backtesting_data[(backtesting_data['fighter1'] == fighter_name) | ((backtesting_data['fighter2'] == fighter_name))].sort_values('date')[cols_to_see]

brian_after = backtesting_data_before[(backtesting_data_before['fighter1'] == fighter_name) | ((backtesting_data_before['fighter2'] == fighter_name))].sort_values('date')[cols_to_see]



westin_before = backtesting_data[(backtesting_data['fighter1'] == 'westinwilson') | ((backtesting_data['fighter2'] == 'westinwilson'))].sort_values('date')[['fighter1', 'fighter2', 'outcome', 'method', 'fighter1_hist_times_knocked_out_round_1', 'fighter2_hist_times_knocked_out_round_1']]

westin_after = backtesting_data_before[(backtesting_data_before['fighter1'] == 'westinwilson') | ((backtesting_data_before['fighter2'] == 'westinwilson'))].sort_values('date')[['fighter1', 'fighter2', 'outcome', 'method', 'fighter1_hist_times_knocked_out_round_1', 'fighter2_hist_times_knocked_out_round_1']]










##########################################################################################
# TESTING FIGHT STATS AGGREGATION BEFORE
##########################################################################################

fight_stats_before = fp.ParquetFile('./data/' + config['fight_stats_file_name'].replace('.parquet', '_clean.parquet')).to_pandas()