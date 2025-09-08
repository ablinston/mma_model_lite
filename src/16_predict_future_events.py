# This script fits XGBoost models to the future events data after refreshing data scrape


# Read in global objects and packages
exec(open('global.py').read())

# Slight different settings on windows and R Pi
if not os.name == "nt":

    # Use this to skip update of data if it's already been done recently
    predict_future_events(
            skip_data_update = False,
            skip_full_data_update = True,
            skip_save_data = True,
            skip_betfair_history = True,
            config = config,
            data_processing_config = data_processing_config,
            model_hyper_params = model_hyper_params,
            betfair_login_config = betfair_login_config,
            test_fighters = test_fighters,
            output_path = '/home/andy/to_gdrive'
            )
    
else:
    
    predict_future_events(
            skip_data_update = False,
            skip_full_data_update = False,
            skip_save_data = False,
            skip_betfair_history = True,
            config = config,
            data_processing_config = data_processing_config,
            model_hyper_params = model_hyper_params,
            betfair_login_config = betfair_login_config,
            test_fighters = test_fighters,
            output_path = None
            )

