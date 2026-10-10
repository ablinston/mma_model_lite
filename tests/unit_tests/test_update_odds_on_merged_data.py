# Load all the packages and functions needed
import fastparquet as fp
import pandas as pd

from mma_functions.odds import update_odds_on_merged_data
from mma_functions.config import config, data_processing_config

test_update_odds_on_merged_data_correct = fp.ParquetFile("./tests/data/test_update_odds_on_merged_data_correct.parquet").to_pandas()

def test_update_odds_on_merged_data():

    result = update_odds_on_merged_data(
        cleaned_fight_odds_data_path="./tests/data/test_update_odds_on_merged_data_fight_odds.parquet",
        cleaned_event_odds_data_path="./tests/data/test_update_odds_on_merged_data_event_odds.parquet",
        cleaned_betfair_odds_data_path="./tests/data/test_update_odds_on_merged_data_cleaned_bf.parquet",
        final_data_folder="./tests/data/test_update_odds_on_merged_data/",
        average_odds_model_path="./results/average_odds_model.joblib",
        config=config,
        data_processing_config=data_processing_config
    )

    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(
        test_update_odds_on_merged_data_correct,
        result,
        check_dtype=False
    )




# fight_odds = pd.DataFrame({
#     'open_odds': [-130.0, 200],
#     'closing_odds_low': [-155.0, 200],
#     'closing_odds_high': [-135.0, 300],
#     'event': ['UFC 1', 'UFC 1'],
#     'date': [pd.Timestamp('2025-04-13'), pd.Timestamp('2025-04-13')],
#     'fighter': ['A', 'B'],
#     'opponent': ['B', 'A'],
#     'notformodel_date_fighter_odd_scrape': [datetime(2025, 4, 7, 8, 40, 13, 538626), datetime(2025, 4, 7, 8, 40, 13, 538626)],
#     'bout': ['A vs B', 'A vs B']
# })

# event_odds = pd.DataFrame({
#     'fighter': ['A', 'B'],
#     'odds_co_1': [-135.0, 114.0],
#     'odds_co_2': [-145.0, 120.0],
#     'odds_co_3': [-155.0, 123.0],
#     'odds_co_4': [-148.0, 116.0],
#     'odds_co_5': [np.nan, np.nan],
#     'odds_co_6': [np.nan, np.nan],
#     'odds_co_7': [np.nan, np.nan],
#     'odds_co_8': [np.nan, np.nan],
#     'odds_co_9': [np.nan, np.nan],
#     'opponent': ['B', 'A'],
#     'date': [pd.Timestamp('2025-04-13')] * 2,
#     'event': ['UFC 1'] * 2,
#     'bout': ['A vs B'] * 2
# })



# merged_data = pd.DataFrame({
#     'event': ['UFC 1'],
#     'bout': ['A vs B'],
#     'fighter1': ['A'],
#     'fighter2': ['B'],
#     'date': [pd.Timestamp('2025-04-13')],
#     'fighter1_open_odds': [2],
#     'fighter1_mean_closing_odds': [2],
#     'fighter1_best_closing_odds': [2],
#     'fighter1_betfair_exch_odds': [2],
#     'fighter1_odds_change': [0],
#     'fighter1_notformodel_date_fighter_odd_scrape': [datetime(2023, 4, 7, 8, 40, 13, 538626)],
#     'fighter1_odds_change_gt_0': [-1],
#     'fighter2_open_odds': [float('nan')],
#     'fighter2_mean_closing_odds': [float('nan')],
#     'fighter2_best_closing_odds': [float('nan')],
#     'fighter2_betfair_exch_odds': [2],
#     'fighter2_odds_change': [float('nan')],
#     'fighter2_notformodel_date_fighter_odd_scrape': [pd.NaT],
#     'fighter2_odds_change_gt_0': [float('nan')],
#     'fighter1_backtesting_odds': [2],
#     'fighter2_backtesting_odds': [2],
#     'fighter1_notformodel_betfair_odds_datetime':[datetime(2023, 4, 7, 8, 40, 13, 538626)],
#     'fighter2_notformodel_betfair_odds_datetime':[datetime(2023, 4, 7, 8, 40, 13, 538626)],
#     'fighter1_notformodel_best_betfair_odds_after_cutoff': [np.nan],
#     'fighter2_notformodel_best_betfair_odds_after_cutoff': [np.nan]
# })
    
# merged_data = calculate_all_true_odds_columns(merged_data, impute_missing_betfair = data_processing_config['impute_missing_betfair'], odds_suffix_for_true_odds = data_processing_config['odds_suffix_for_true_odds'], impute_one_sided_missing_odds = data_processing_config['impute_one_sided_missing_odds'])

# merged_data = add_fighter_differentials(merged_data)




# # cleaned_bf = pd.DataFrame({'fighter':[ 'A', 'B'], 'bout': ['A vs B'] * 2, 'date': [pd.to_datetime('2025-04-13')] * 2, 'betfair_exch_odds':[1.5, 2.5], 'notformodel_best_betfair_odds_after_cutoff': [1.6, 2.6], 'opponent': ['B', 'A'], 'event': ['UFC 1'] * 2, 'notformodel_betfair_odds_datetime': [pd.to_datetime('2025-06-05')] * 2})

# # fp.write('./tests/data/test_update_odds_on_merged_data_fight_odds.parquet', fight_odds, compression = 'ZSTD')
# # fp.write('./tests/data/test_update_odds_on_merged_data_event_odds.parquet', event_odds, compression = 'ZSTD')
# fp.write('./tests/data/test_update_odds_on_merged_data/merged_data.parquet', merged_data, compression = 'ZSTD')
# # fp.write('./tests/data/test_update_odds_on_merged_data_cleaned_bf.parquet', cleaned_bf, compression = 'ZSTD')
# # 
# fp.write('./tests/data/test_update_odds_on_merged_data_correct.parquet', result, compression = 'ZSTD')
