# This script fits XGBoost models to the future events data after refreshing data scrape


# Read in global objects and packages
from mma_functions.pipelines import predict_future_events
from mma_functions.config import betfair_login_config, config, data_processing_config, model_hyper_params, test_fighters

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
    output_path = './results/temp'
    )
