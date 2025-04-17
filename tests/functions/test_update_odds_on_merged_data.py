# Load all the packages and functions needed
exec(open('global.py').read())

test_update_odds_on_merged_data_correct = fp.ParquetFile('./tests/data/test_update_odds_on_merged_data_correct.parquet').to_pandas()

def test_update_odds_on_merged_data():

    result = update_odds_on_merged_data(
        cleaned_fight_odds_data_path = './tests/data/test_update_odds_on_merged_data_fight_odds.parquet',
        cleaned_event_odds_data_path = './tests/data/test_update_odds_on_merged_data_event_odds.parquet',
        final_data_folder = './tests/data/test_update_odds_on_merged_data/',
        average_odds_model_path = './results/average_odds_model.joblib',
        config = config
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
#     'fighter1_odds_change': [0],
#     'fighter1_notformodel_date_fighter_odd_scrape': [datetime(2023, 4, 7, 8, 40, 13, 538626)],
#     'fighter1_odds_change_gt_0': [-1],
#     'fighter2_open_odds': [float('nan')],
#     'fighter2_mean_closing_odds': [float('nan')],
#     'fighter2_best_closing_odds': [float('nan')],
#     'fighter2_odds_change': [float('nan')],
#     'fighter2_notformodel_date_fighter_odd_scrape': [pd.NaT],
#     'fighter2_odds_change_gt_0': [float('nan')],
#     'fighter1_backtesting_odds': [2],
#     'fighter2_backtesting_odds': [2],
#     'diff_odds_change': [float('nan')],
#     'diff_odds_change_gt_0': [float('nan')],
#     'diff_ratio_best_closing_odds': [float('nan')],
#     'diff_ratio_mean_closing_odds': [float('nan')],
#     'diff_ratio_odds_change': [float('nan')],
#     'diff_ratio_odds_change_gt_0': [float('nan')],
#     'diff_ratio_open_odds': [float('nan')]
# })
    

# fp.write('./tests/data/test_update_odds_on_merged_data_fight_odds.parquet', fight_odds, compression = 'ZSTD')
# fp.write('./tests/data/test_update_odds_on_merged_data_event_odds.parquet', event_odds, compression = 'ZSTD')
# fp.write('./tests/data/test_update_odds_on_merged_data/merged_data.parquet', merged_data, compression = 'ZSTD')
# fp.write('./tests/data/test_update_odds_on_merged_data_correct.parquet', result, compression = 'ZSTD')
