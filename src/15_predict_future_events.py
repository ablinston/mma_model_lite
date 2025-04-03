# This script fits XGBoost models to the future events data after refreshing data scrape

# import gc
# import h2o
# from h2o.automl import H2OAutoML
# from hyperopt import hp, STATUS_OK, tpe, fmin, Trials # hyperparameter optimisation functions
# from hyperopt.atpe import suggest as atpe_suggest
# import itertools
# import lightgbm as lgb
# import mlflow
# import pickle
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.preprocessing import LabelEncoder, label_binarize

# Read in global objects and packages
exec(open('global.py').read())

# Replace the firefox location if we're not on windows
if not os.name == "nt":

    # Use this to skip update of data if it's already been done recently
    skip_data_update = False
    skip_full_data_update = True
    skip_save_data = True
else:
    skip_data_update = False
    skip_full_data_update = False
    skip_save_data = False


# # Load files
# mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
# rf_xgb_model = mlflow.pyfunc.load_model('models:/xgb_forest_model/3')


##########################################################################################
### Update the upcoming event and odds data ###
##########################################################################################

if not skip_data_update:
    
    # Run these scripts to scrape then clean the raw data
    exec(open('./src/4_scrape_all_unparsed_data.py').read())
    exec(open('./src/5_data_cleaning.py').read())
        
    # Optional, can either rerun all the processing, or just update the odds. By default will rerun all processing if new events were found to have completed
    if skip_full_data_update:
        
        # Clear out memory
        globals().clear()
        # Read in global objects and packages
        exec(open('global.py').read())
        gc.collect()
        
        forecasting_data = update_odds_on_merged_data(
                cleaned_fight_odds_data_path = './data/' + config['fighter_odds_file_name'].replace('.parquet', '_clean.parquet'),
                cleaned_event_odds_data_path = './data/' + config['event_odds_file_name'].replace('.parquet', '_clean.parquet'),
                events_path = './data/' + config['event_details_file_name'].replace('.parquet', '_clean.parquet'),
                final_data_folder = config['final_data_path'],
                average_odds_model_path = './results/average_odds_model.joblib',
                config = config
                )
    else:
        exec(open('./src/7_data_processing.py').read())
        # Refresh the dataset
        forecasting_data = merged_data.copy()
else:
    forecasting_data = pq.ParquetDataset(config['final_data_path']).read().to_pandas()
    # add in only odds update
    

# Apply standard filters
forecasting_data = standard_data_filter(forecasting_data, include_upcoming = True)

##########################################################################################
### Choose which events to predict ###
##########################################################################################

upcoming_fights = forecasting_data[forecasting_data['outcome'].isna()]
# upcoming_fights = forecasting_data[forecasting_data['event'] == 'ufc311:makhachevvs.moicano']

##########################################################################################
### Get the models predictions ###
##########################################################################################

# Get the data ready for the model
X_upcoming, temp1, y_upcoming, temp2, temp3, temp4, temp5 = prep_data_for_model_training(
    upcoming_fights,
    upcoming_fights = True)

# Load files
# rf_xgb_model = mlflow.pyfunc.load_model('models:/xgb_forest_model/2')
rf_xgb_model = joblib.load(bet_config['model_to_use'])

# Get the model predictions
# prediction_probabilities, temp = rf_xgb_model.predict(X_upcoming)
prediction_probabilities, temp = predict_xgboost_ensemble(rf_xgb_model, X_upcoming, [0] * len(y_upcoming))

# Add formatted fighters
upcoming_fights = add_formatted_fighter_names(upcoming_fights, on = ['fighter1', 'fighter2'], formatted_col_names = ['fighter1', 'fighter2'], config = config)

# Summarise the model predictions and betting decisions using the config
betting_data = summarise_model_predictions(
    actuals_data_df = upcoming_fights, 
    prediction_probabilities = prediction_probabilities, 
    min_prob_to_bet = bet_config['min_prob_to_bet'], 
    min_ev_to_bet = bet_config['min_ev_to_bet'], 
    min_bet_size = bet_config['min_bet_size'], 
    max_bet_size = bet_config['max_bet_size'])

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

betting_data.rename(columns = {'fighter1_notformodel_date_fighter_odd_scrape': 'f1_scrape_date', 'fighter2_notformodel_date_fighter_odd_scrape': 'f2_scrape_date'}, inplace = True)

summary_data = betting_data[['date', 'event', 'f1_scrape_date', 'f2_scrape_date', 'fighter1', 'fighter1_backtesting_odds', 'Fighter1_win_prob', 'Fighter1_win_EV', 'fighter2', 'fighter2_backtesting_odds', 'Fighter2_win_prob', 'Fighter2_win_EV', 'Bet_trigger', 'bet_assumed_odds', 'Bet_breakeven_odds', 'Final_bet', 'Potential_win']]


print(
      summary_data.to_string()
)

# Get only the bets to place
bets_to_place = betting_data[(betting_data['bets'] == 1) & (betting_data['date'] == betting_data['date'].min())].copy()
bets_to_place['bet_win_prob'] = np.where(bets_to_place['Bet_trigger'] == bets_to_place['fighter1'], bets_to_place['Fighter1_win_prob'], bets_to_place['Fighter2_win_prob'])

try:
    del round
except:
    pass

for i in range(0, len(bets_to_place)):
    print("")
    print(f"Bet: {bets_to_place.iloc[i]['Bet_trigger']}")
    print(f"Win probability: {bets_to_place.iloc[i]['bet_win_prob']:.1f}%")
    print(f"EV: {bets_to_place.iloc[i]['bet_ev']:.2f} at {bets_to_place.iloc[i]['bet_assumed_odds']:.2f} odds")
    print(f"Breakeven odds: {bets_to_place.iloc[i]['Bet_breakeven_odds']:.2f}")

# Output the results
summary_data.to_csv('./results/upcoming_predictions.csv')

# If on Raspberry Pi, output to Gdrive
if not os.name == "nt":
    
    summary_data.to_csv(f'/home/andy/to_gdrive/upcoming_predictions_{datetime.today()}.csv')
    
    # Create text file with the bet summaries
    text_file_path = "/home/andy/to_gdrive/bets_to_place.txt"
    
    with open(text_file_path, "w") as file:
        for i in range(0, len(bets_to_place)):
            print("", file = file)
            print(f"Bet: {bets_to_place.iloc[i]['Bet_trigger']}", file = file)
            print(f"Win probability: {bets_to_place.iloc[i]['bet_win_prob']:.1f}%", file = file)
            print(f"EV: {bets_to_place.iloc[i]['bet_ev']:.2f} at {bets_to_place.iloc[i]['bet_assumed_odds']:.2f} odds", file = file)
            print(f"Breakeven odds: {bets_to_place.iloc[i]['Bet_breakeven_odds']:.2f}", file = file)