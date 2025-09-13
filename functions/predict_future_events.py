# This function predicts future UFC events

def predict_future_events(
        skip_data_update,
        skip_full_data_update,
        skip_save_data,
        skip_betfair_history,
        config,
        data_processing_config,
        model_hyper_params,
        betfair_login_config,
        test_fighters,
        output_path = None,
        ):
    
    """
    This function predicts all the future UFC events. It first scrapes the latest data, then if required, reprocesses
    the data to get the latest forecasting dataset, and then forecasts the upcoming fights based on the standard data filters
    
    Parameters
    ----------
    skip_data_update: bool
        whether to skip updating the data and go straight to predictions
    
    skip_full_data_update: bool
        whether to skip re-running the processing script on cleaned data. If set to False, only the odds will be updated
        
    skip_save_data: bool
        set whether the saving of processed data should be skipped i.e. only used for the predictions
        
    skip_betfair_history: bool
        set whether to skip downloading the latest betfair odds history data
    
    output_path: str
        the path where the outputs from the predictions will be saved. This is in addition to automatic outputs, which will be saved in ./results/
    
    """
    
    # Set the odds to use as upcoming instead of as used in the regular data
    data_processing_config['odds_suffix_for_true_odds'] = data_processing_config['upcoming_odds_suffix_for_true_odds']
    model_hyper_params['odds_to_use'] = data_processing_config['upcoming_odds_suffix_for_true_odds']

    ##########################################################################################
    ### Update the upcoming event and odds data ###
    ##########################################################################################

    if not skip_data_update:
        
        # Run these scripts to scrape then clean the raw data
        scrape_all_unparsed_data(skip_betfair_history, config, betfair_login_config)
        clean_all_raw_data_and_save(config)
        
        gc.collect();
        
        # Optional, can either rerun all the processing, or just update the odds. By default will rerun all processing if new events were found to have completed
        if skip_full_data_update:
            
            # Clear out memory
            # globals().clear()
            # # Read in global objects and packages
            # exec(open('global.py').read())
            # gc.collect();
            
            # # Make sure for the backtesting odds we use whatever upcoming because they're used to assess the model predictions later
            # model_hyper_params['odds_to_use'] = data_processing_config['upcoming_odds_suffix_for_true_odds']
            
            forecasting_data = update_odds_on_merged_data(
                    cleaned_fight_odds_data_path = './data/' + config['fighter_odds_file_name'].replace('.parquet', '_clean.parquet'),
                    cleaned_event_odds_data_path = './data/' + config['event_odds_file_name'].replace('.parquet', '_clean.parquet'),
                    cleaned_betfair_odds_data_path = './data/' + config['betfair_upcoming_odds_file_name'].replace('.parquet', '_clean.parquet'),
                    # events_path = './data/' + config['event_details_file_name'].replace('.parquet', '_clean.parquet'),
                    final_data_folder = config['final_data_path'],
                    average_odds_model_path = './results/average_odds_model.joblib',
                    config = config,
                    data_processing_config = data_processing_config,
                    upcoming_only = True
                    )
        else:
            # Refresh the dataset
            forecasting_data = merge_and_process_all_clean_data(
                    config,
                    elo_config,
                    data_processing_config,
                    odds_scrape_config,
                    model_hyper_params,
                    test_fighter_list = test_fighters,
                    skip_save_data = skip_save_data # whether to skip the save data and just return merged data
                    )
            
    else:
        forecasting_data = fp.ParquetFile('./data/final_dataset/final_dataset.parquet').to_pandas() # pq.ParquetDataset(config['final_data_path']).read().to_pandas()
        # add in only odds update
     
    gc.collect();
    # pdb.set_trace()
            
    # Apply standard filters
    forecasting_data = standard_data_filter(forecasting_data, include_upcoming = True, keep_rows_with_flag = True)

    ##########################################################################################
    ### Choose which events to predict ###
    ##########################################################################################

    upcoming_fights = forecasting_data[forecasting_data['outcome'].isna()]
    # upcoming_fights = forecasting_data[forecasting_data['event'] == 'ufc311:makhachevvs.moicano']

    upcoming_fights_for_model = upcoming_fights[upcoming_fights['notformodel_exclusion_reason'] == ''].copy()

    ##########################################################################################
    ### Get the models predictions ###
    ##########################################################################################

    # Get the data ready for the model
    X_upcoming, temp1, y_upcoming, temp2, temp3, temp4, temp5 = prep_data_for_model_training(
        upcoming_fights_for_model,
        upcoming_fights = True)

    # Load files
    # rf_xgb_model = mlflow.pyfunc.load_model('models:/xgb_forest_model/2')
    rf_xgb_model = joblib.load(bet_config['model_to_use'])

    # Get the model predictions
    # prediction_probabilities, temp = rf_xgb_model.predict(X_upcoming)
    prediction_probabilities, temp = predict_xgboost_ensemble(rf_xgb_model, X_upcoming, [0] * len(y_upcoming), model_hyper_params)

    # Add formatted fighters
    upcoming_fights_for_model = add_formatted_fighter_names(upcoming_fights_for_model, on = ['fighter1', 'fighter2'], formatted_col_names = ['fighter1', 'fighter2'], config = config)
    upcoming_fights = add_formatted_fighter_names(upcoming_fights, on = ['fighter1', 'fighter2'], formatted_col_names = ['fighter1', 'fighter2'], config = config)

    gc.collect();

    # Summarise the model predictions and betting decisions using the config
    betting_data = summarise_model_predictions(
        actuals_data_df = upcoming_fights_for_model, 
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

    # Add the fights that haven't been modelled along with exclusion reason for info
    summary_data = pd.concat([
         summary_data,
         upcoming_fights[upcoming_fights['notformodel_exclusion_reason'] != ''][['date', 'event', 'fighter1', 'fighter2', 'notformodel_exclusion_reason']]
    ])

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
        print(f"Breakeven exchange odds: {bets_to_place.iloc[i]['Bet_breakeven_exchange_odds']:.2f}")

    # Output the results
    summary_data.to_csv('./results/upcoming_predictions.csv')

    # If on Raspberry Pi, output to Gdrive
    if not output_path == None:
        
        summary_data.to_csv(f'{output_path}/upcoming_predictions_{datetime.today()}.csv')
        
        # Create text file with the bet summaries
        text_file_path = f"{output_path}/bets_to_place.txt"
        
        with open(text_file_path, "w") as file:
            for i in range(0, len(bets_to_place)):
                print("", file = file)
                print(f"Bet: {bets_to_place.iloc[i]['Bet_trigger']}", file = file)
                print(f"Win probability: {bets_to_place.iloc[i]['bet_win_prob']:.1f}%", file = file)
                print(f"EV: {bets_to_place.iloc[i]['bet_ev']:.2f} at {bets_to_place.iloc[i]['bet_assumed_odds']:.2f} odds", file = file)
                print(f"Breakeven odds: {bets_to_place.iloc[i]['Bet_breakeven_odds']:.2f}", file = file)
                print(f"Breakeven exchange odds: {bets_to_place.iloc[i]['Bet_breakeven_exchange_odds']:.2f}", file = file)

    return summary_data