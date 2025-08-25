# Load all the packages and functions needed
exec(open('global.py').read())

# # Load what answers are expected and inputs to the function
# update_fighter_odds_data_input = fp.ParquetFile('./tests/data/update_fighter_odds_data_input.parquet').to_pandas()
# update_fighter_odds_data_correct1 = fp.ParquetFile('./tests/data/update_fighter_odds_data_data_correct.parquet').to_pandas().reset_index(drop = True)
# update_fighter_odds_data_correct2 = fp.ParquetFile('./tests/data/update_fighter_odds_data_data_correct2.parquet').to_pandas().reset_index(drop = True)
# update_fighter_odds_data_correct3 = fp.ParquetFile('./tests/data/update_fighter_odds_data_data_correct3.parquet').to_pandas().reset_index(drop = True)

# This test looks at a specific URL for an historic event and checks that the results match what is expected
def test_derive_start_elos():
    
    fight_results = pd.DataFrame({
        "event": ["UFC1", "UFC2", "UFC3", "UFC4"],
        "bout": [
            "Alicevs.Bob",       # Fight 1
            "Alicevs.Charlie",   # Fight 2
            "Bobvs.Charlie",     # Fight 3
            "Alicevs.Bob"
        ],
        "outcome": ["W/L", "L/W", "W/L", np.nan],   # Alice beats Bob, Alice loses to Charlie, Bob beats Charlie
        "method": ["KO/TKO", "Decision-Unanimous", "Submission", np.nan],
        "weightclass": ["LW", "LW", "LW", "LW"],
        "date": ["2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01"],
        "time format": ["3 Rnd (5-5-5)"] * 4,
    })

    # Fight stats: one row per fighter per fight (so 2x fight_results)
    fight_stats = pd.DataFrame([
        # Fight 1: Alice vs Bob
        {"event": "UFC1", "bout": "Alicevs.Bob", "fighter": "Alice",
         "allrounds_fight_minutes": 5, "allrounds_ctrl_minutes": 2,
         "allrounds_sig.str._landed": 20, "allrounds_kd": 1, "round1_fight_minutes": 5},
        {"event": "UFC1", "bout": "Alicevs.Bob", "fighter": "Bob",
         "allrounds_fight_minutes": 5, "allrounds_ctrl_minutes": 1,
         "allrounds_sig.str._landed": 10, "allrounds_kd": 0, "round1_fight_minutes": 5},

        # Fight 2: Alice vs Charlie
        {"event": "UFC2", "bout": "Alicevs.Charlie", "fighter": "Alice",
         "allrounds_fight_minutes": 15, "allrounds_ctrl_minutes": 7,
         "allrounds_sig.str._landed": 25, "allrounds_kd": 0, "round1_fight_minutes": 5},
        {"event": "UFC2", "bout": "Alicevs.Charlie", "fighter": "Charlie",
         "allrounds_fight_minutes": 15, "allrounds_ctrl_minutes": 8,
         "allrounds_sig.str._landed": 30, "allrounds_kd": 1, "round1_fight_minutes": 5},

        # Fight 3: Bob vs Charlie
        {"event": "UFC3", "bout": "Bobvs.Charlie", "fighter": "Bob",
         "allrounds_fight_minutes": 10, "allrounds_ctrl_minutes": 6,
         "allrounds_sig.str._landed": 22, "allrounds_kd": 1, "round1_fight_minutes": 5},
        {"event": "UFC3", "bout": "Bobvs.Charlie", "fighter": "Charlie",
         "allrounds_fight_minutes": 10, "allrounds_ctrl_minutes": 4,
         "allrounds_sig.str._landed": 15, "allrounds_kd": 0, "round1_fight_minutes": 5},
        
        # Fight 4: Alice vs Bob
        {"event": "UFC4", "bout": "Alicevs.Bob", "fighter": "Alice",
         "allrounds_fight_minutes": np.nan, "allrounds_ctrl_minutes": np.nan,
         "allrounds_sig.str._landed": np.nan, "allrounds_kd": np.nan, "round1_fight_minutes": np.nan},
        {"event": "UFC4", "bout": "Alicevs.Bob", "fighter": "Bob",
         "allrounds_fight_minutes": np.nan, "allrounds_ctrl_minutes": np.nan,
         "allrounds_sig.str._landed": np.nan, "allrounds_kd": np.nan, "round1_fight_minutes": np.nan},
    ])

     # Run prep function
    start_elo, merged_data = prep_stats_for_elo_calc(
        fight_results, fight_stats, config, remove_nas=False
    )

    start_elo, merged_data = derive_start_elos(start_elo, merged_data, elo_config)

    start_elo = start_elo.to_pandas()

    # Compare starting elos on second fight after first result
    assert start_elo[(start_elo['fighter'] == 'Alice') & (start_elo['fight_number'] == 2)]['start_elo'] > int(elo_config['starting_elo'])
    assert start_elo[(start_elo['fighter'] == 'Bob') & (start_elo['fight_number'] == 2)]['start_elo'] < int(elo_config['starting_elo'])
    assert start_elo[(start_elo['fighter'] == 'Charlie') & (start_elo['fight_number'] == 2)]['start_elo'] > int(elo_config['starting_elo'])
    assert all(start_elo[start_elo['fight_number'] == 1]['start_elo'] == int(elo_config['starting_elo']))
    # assert result3.astype(str).compare(update_fighter_odds_data_correct3.astype(str)) == True #  Not currently used in model but fix for errors if used.
    # Check we have the upcoming ELO
    assert start_elo[(start_elo['fighter'] == 'Alice') & (start_elo['fight_number'] == 3)]['start_elo'] < int(elo_config['starting_elo'])

# fp.write('./tests/data/update_fighter_odds_data_data_correct3.parquet', result3, compression = 'ZSTD')
