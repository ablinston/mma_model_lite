# Load all the packages and functions needed
from ablinston_ml.validation import equals_to_dp

from mma_functions.pipelines import merge_and_process_all_clean_data
from mma_functions.config import config, data_processing_config, elo_config, model_hyper_params, odds_scrape_config, test_fighters

# This is a test on data that is before the event, and after the event, and tests that it is identical

def test_update_odds_on_merged_data():

    # First merge the current data with current config
    new_data = merge_and_process_all_clean_data(
            config,
            elo_config,
            data_processing_config,
            odds_scrape_config,
            model_hyper_params,
            test_fighter_list=test_fighters,
            skip_save_data=True # whether to skip the save data and just return merged data
            )

    # now replace clean data locations with the test data, which is data saved before 13/09/2025

    for filename in [
            "event_details_file_name", 
            "upcoming_event_details_file_name", 
            "upcoming_fight_details_file_name", 
            "fight_details_file_name",
            "fight_results_file_name",
            "fight_stats_file_name",
            "fighter_details_file_name",
            "fighter_tott_file_name",
            "fighter_odds_file_name",
            "fighter_names_file_name",
            "event_odds_file_name",
            "betfair_odds_file_name",
            "betfair_upcoming_odds_file_name"]:
        config[filename] = "../tests/data/test_merge_all_data/" + config[filename]

    # Now calculate data pre-event
    pre_event_data = merge_and_process_all_clean_data(
            config,
            elo_config,
            data_processing_config,
            odds_scrape_config,
            model_hyper_params,
            test_fighter_list=test_fighters,
            skip_save_data=True # whether to skip the save data and just return merged data
            )
    
    # Just filter the event upcoming
    pre_event_data = pre_event_data[pre_event_data["outcome"].isna()]
    
    new_data = new_data[(new_data["date"].isin(pre_event_data["date"].unique().tolist())) & (new_data["bout"].isin(pre_event_data["bout"].tolist()))]
    pre_event_data = pre_event_data[pre_event_data["bout"].isin(new_data["bout"].tolist())]

    # Drop any outcome columns
    cols_to_drop = ["outcome", "method", "time_format"] + [col for col in new_data.columns if "odd" in col or "margin" in col] + [col for col in new_data.columns if col.endswith("_fights")] # some new fighters have no _fights in data before
    pre_event_data.drop(columns=cols_to_drop + [col for col in pre_event_data if col not in new_data.columns and "notformodel_best_betfair_odds" in col], inplace=True)
    new_data.drop(columns=cols_to_drop, inplace=True)
    
    # Reset indexes
    new_data = new_data.sort_values(["event", "fighter1", "fighter2"]).reset_index(drop=True)
    pre_event_data = pre_event_data.sort_values(["event", "fighter1", "fighter2"]).reset_index(drop=True)

    # Had issue with stances of fighters changing (because fighter info is rescraped sometimes) so remove those rows from the data
    same_stance = (new_data["fighter1_stance"] == pre_event_data["fighter1_stance"]) & (new_data["fighter2_stance"] == pre_event_data["fighter2_stance"])

    pre_event_data = pre_event_data[same_stance]
    new_data = new_data[same_stance]
    
    assert equals_to_dp(pre_event_data, new_data, decimals=15) # Temp until can get new saves of data after fix to data cleaning



    
