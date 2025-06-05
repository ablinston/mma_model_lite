
# Load all the packages and functions needed
exec(open('global.py').read())

correct_result = fp.ParquetFile('./tests/data/test_calculate_all_true_odds_columns_correct.parquet').to_pandas()

def test_calculate_all_true_odds_columns(monkeypatch):

    # Create test data with various missing and non-missing values
    df = pd.DataFrame({
        'fighter1_betfair_exch_odds': [2.0, np.nan, 2.5, np.nan],
        'fighter2_betfair_exch_odds': [np.nan, 1.67, 3.0, np.nan],
        'fighter1_best_closing_odds': [2.05, 1.95, 2.55, 2.55],
        'fighter2_best_closing_odds': [1.95, 1.67, 3.05, 3.05],
        'fighter1_open_odds': [2.2, np.nan, 2.8, np.nan],
        'fighter2_open_odds': [1.8, 1.6, np.nan, 3.05],
        'fighter1_notformodel_betfair_odds_datetime': pd.to_datetime(['2024-01-01'] * 4),
        'fighter2_notformodel_betfair_odds_datetime': pd.to_datetime(['2024-01-01'] * 4),
        'fighter1_notformodel_date_fighter_odd_scrape': pd.to_datetime(['2024-01-02'] * 4),
        'fighter2_notformodel_date_fighter_odd_scrape': pd.to_datetime(['2024-01-02'] * 4),
    })

    result = calculate_all_true_odds_columns(
        merged_data=df.copy(),
        impute_missing_betfair=True,
        odds_suffix_for_true_odds='betfair_exch_odds',
        impute_one_sided_missing_odds=True
    )


    # Check true odds columns were created and match the dummy outputs
    assert 'fighter1_true_closing_odds' in result.columns
    assert result.equals(correct_result)

    # Check datetime logic
    # Row 0: fighter2 had missing original odds, should use scrape date
    assert result.loc[0, 'fighter2_notformodel_true_odds_asat_datetime'] == result.loc[0, 'fighter2_notformodel_betfair_odds_datetime']
    # Row 2: fighter2 had valid betfair odds, should use betfair datetime
    assert result.loc[2, 'fighter2_notformodel_true_odds_asat_datetime'] == result.loc[2, 'fighter2_notformodel_betfair_odds_datetime']

# fp.write('./tests/data/test_calculate_all_true_odds_columns_correct.parquet', result, compression = 'ZSTD')
