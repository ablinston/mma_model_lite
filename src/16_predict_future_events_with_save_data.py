# This script fits XGBoost models to the future events data after refreshing data scrape


# Read in global objects and packages
exec(open('global.py').read())

# Replace the firefox location if we're not on windows
if not os.name == "nt":

    # Use this to skip update of data if it's already been done recently
    predict_future_events(
            skip_data_update = False,
            skip_full_data_update = False,
            skip_save_data = False,
            skip_betfair_history = True,
            config = config,
            data_processing_config = data_processing_config,
            model_hyper_params = model_hyper_params,
            betfair_login_config = betfair_login_config,
            output_path = None
            )
    
else:
    
    predict_future_events(
            skip_data_update = False,
            skip_full_data_update = False,
            skip_save_data = False,
            skip_betfair_history = False,
            config = config,
            data_processing_config = data_processing_config,
            model_hyper_params = model_hyper_params,
            betfair_login_config = betfair_login_config,
            output_path = None
            )

