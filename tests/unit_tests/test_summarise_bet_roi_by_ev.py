# Load all the packages and functions needed
exec(open('global.py').read())

# This test looks at a specific URL for an historic event and checks that the results match what is expected
def test_summarise_bet_roi_by_ev():
    
   # Example: 4 fights, Fighter1 always wins with EV=0.1 (positive), odds=2.0
    data = {
        "Fighter1_win_EV": [0.1, 0.2, 0.1, 0.3],
        "fighter1_backtesting_odds": [2.0, 2.0, 2.0, 2.5],
        "Fighter2_win_EV": [-0.1, -0.2, -0.1, -0.3],
        "fighter2_backtesting_odds": [2.0, 2.0, 2.0, 1.5],
        "outcome": ["W/L", "W/L", "W/L", "W/L"],  # Fighter1 wins every time
    }
    df = pd.DataFrame(data)

    result = summarise_bet_roi_by_ev(df)

    # Ensure the positive EV band has ROI = 100%
    pos_band = result.loc[result.index.astype(str).str.contains("0.0, 0.1")]  # the EV bin around 0.1
    assert not pos_band.empty
    roi = pos_band["ROI_on_constant_bets"].iloc[0]
    assert np.isclose(roi, 100.0)
    
    assert np.isclose(result['ROI_on_constant_bets'].iloc[0],- 100.0)
    
    assert np.isclose(result['ROI_on_constant_bets'].iloc[-1], 150.0)
    


# fp.write('./tests/data/parse_fight_details_url_data_correct.parquet', result, compression = 'ZSTD')
