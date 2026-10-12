# Load all the packages and functions needed
import numpy as np
import pandas as pd

from mma_functions.elo import add_start_elo_to_fight_results
from mma_functions.config import config, elo_config

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
        "method": ["KO/TKO", "Other", "Submission", np.nan],
        "weightclass": ["LW", "LW", "LW", "LW"],
        "date": ["2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01"],
        "time format": ["3 Rnd (5-5-5)"] * 4,
    })

    # Fight stats: one row per fighter per fight (so 2x fight_results)
    fight_stats = pd.DataFrame([
        # Fight 1: Alice vs Bob
        {"date": "2021-01-01", "event": "UFC1", "bout": "Alicevs.Bob", "fighter": "Alice",
         "allrounds_fight_minutes": 5, "allrounds_ctrl_minutes": 2,
         "allrounds_sig.str._landed": 20, "allrounds_kd": 1, "round1_fight_minutes": 5},
        {"date": "2021-01-01", "event": "UFC1", "bout": "Alicevs.Bob", "fighter": "Bob",
         "allrounds_fight_minutes": 5, "allrounds_ctrl_minutes": 1,
         "allrounds_sig.str._landed": 10, "allrounds_kd": 0, "round1_fight_minutes": 5},

        # Fight 2: Alice vs Charlie
        {"date": "2021-02-01", "event": "UFC2", "bout": "Alicevs.Charlie", "fighter": "Alice",
         "allrounds_fight_minutes": 15, "allrounds_ctrl_minutes": 7,
         "allrounds_sig.str._landed": 25, "allrounds_kd": 0, "round1_fight_minutes": 5},
        {"date": "2021-02-01", "event": "UFC2", "bout": "Alicevs.Charlie", "fighter": "Charlie",
         "allrounds_fight_minutes": 15, "allrounds_ctrl_minutes": 8,
         "allrounds_sig.str._landed": 30, "allrounds_kd": 1, "round1_fight_minutes": 5},

        # Fight 3: Bob vs Charlie
        {"date": "2021-03-01", "event": "UFC3", "bout": "Bobvs.Charlie", "fighter": "Bob",
         "allrounds_fight_minutes": 10, "allrounds_ctrl_minutes": 6,
         "allrounds_sig.str._landed": 22, "allrounds_kd": 1, "round1_fight_minutes": 5},
        {"date": "2021-03-01", "event": "UFC3", "bout": "Bobvs.Charlie", "fighter": "Charlie",
         "allrounds_fight_minutes": 10, "allrounds_ctrl_minutes": 4,
         "allrounds_sig.str._landed": 15, "allrounds_kd": 0, "round1_fight_minutes": 5},
        
        # Fight 4: Alice vs Bob
        {"date": "2021-04-01", "event": "UFC4", "bout": "Alicevs.Bob", "fighter": "Alice",
         "allrounds_fight_minutes": np.nan, "allrounds_ctrl_minutes": np.nan,
         "allrounds_sig.str._landed": np.nan, "allrounds_kd": np.nan, "round1_fight_minutes": np.nan},
        {"date": "2021-04-01", "event": "UFC4", "bout": "Alicevs.Bob", "fighter": "Bob",
         "allrounds_fight_minutes": np.nan, "allrounds_ctrl_minutes": np.nan,
         "allrounds_sig.str._landed": np.nan, "allrounds_kd": np.nan, "round1_fight_minutes": np.nan},
    ])

    # Format dates
    fight_results["date"] = pd.to_datetime(fight_results["date"])
    fight_stats["date"] = pd.to_datetime(fight_stats["date"])
    
    # Add fighter
    fight_results[["fighter1", "fighter2"]] = fight_results["bout"].str.split("vs\\.", expand=True)
    
    # Run function to add ELOs
    start_elo = add_start_elo_to_fight_results(fight_results, fight_stats, config, elo_config)

    # Compare starting elos on second fight after first result
    assert all(start_elo[start_elo["event"] == "UFC3"]["fighter1_start_elo"] < int(elo_config["starting_elo"]))
    assert all(start_elo[start_elo["event"] == "UFC4"]["fighter1_start_elo"] > int(elo_config["starting_elo"]))
    assert all(start_elo[start_elo["event"] == "UFC4"]["fighter2_start_elo"] > int(elo_config["starting_elo"]))
    assert all(start_elo[start_elo["event"] == "UFC3"]["fighter2_start_elo"] == int(elo_config["starting_elo"]))
    
    # Check that the 'Other' method of victory (usually DQ or something) doesn't affect ELO. Alice's
    # rating going into UFC4 should be her rating going into UFC2 with nothing applied but the decay
    # for the two months between them, because the UFC2 result itself counts for nothing
    days_between = (pd.Timestamp("2021-04-01") - pd.Timestamp("2021-02-01")).days
    half_life = float(elo_config["shrinkage_half_life_days"])
    starting = float(elo_config["starting_elo"])

    ufc2_elo = start_elo[start_elo["event"] == "UFC2"]["fighter1_start_elo"].iloc[0]
    expected_at_ufc4 = starting + (ufc2_elo - starting) * 0.5 ** (days_between / half_life)

    assert np.isclose(start_elo[start_elo["event"] == "UFC4"]["fighter1_start_elo"].iloc[0], expected_at_ufc4)


# The whole single pass approach rests on a fighter's starting ELO for a fight being the ending ELO
# from their previous one, decayed towards the starting ELO for the time they were out

def test_derive_start_elos_carries_elo_forward():

    import fastparquet as fp
    from mma_functions.elo import derive_start_elos, prep_stats_for_elo_calc

    fight_results = fp.ParquetFile("./data/" + config["fight_results_file_name"].replace(".parquet", "_clean.parquet")).to_pandas()
    fight_stats = fp.ParquetFile("./data/" + config["fight_stats_file_name"].replace(".parquet", "_clean.parquet")).to_pandas()

    start_elo, merged_data = prep_stats_for_elo_calc(fight_results, fight_stats, config, start_date="2000-01-01", remove_nas=False)

    elos, fights = derive_start_elos(start_elo, merged_data, dict(elo_config))

    # Stack both corners so there is one row per fighter per fight
    fights = fights.to_pandas()

    corner1 = fights[["date", "fighter1", "fighter1_start_elo", "fighter1_end_elo"]].rename(
        columns={"fighter1": "fighter", "fighter1_start_elo": "start_elo", "fighter1_end_elo": "end_elo"})
    corner2 = fights[["date", "fighter2", "fighter2_start_elo", "fighter2_end_elo"]].rename(
        columns={"fighter2": "fighter", "fighter2_start_elo": "start_elo", "fighter2_end_elo": "end_elo"})

    careers = pd.concat([corner1, corner2]).sort_values(["fighter", "date"])

    careers["previous_end_elo"] = careers.groupby("fighter")["end_elo"].shift()
    careers["days_out"] = careers.groupby("fighter")["date"].diff().dt.days

    followed_on = careers[careers["previous_end_elo"].notna()].copy()

    assert len(followed_on) > 5000, f"Expected thousands of repeat fighters, got {len(followed_on)}"

    # The ELO a fighter starts on is the one they finished their last fight on, decayed by half for
    # every shrinkage_half_life_days they spent out of competition
    starting_elo = float(elo_config["starting_elo"])
    half_life = float(elo_config["shrinkage_half_life_days"])

    expected = starting_elo + (followed_on["previous_end_elo"] - starting_elo) * 0.5 ** (followed_on["days_out"] / half_life)

    biggest_gap = (followed_on["start_elo"] - expected).abs().max()

    assert biggest_gap < 1e-9, f"ELO did not carry forward with decay, biggest gap was {biggest_gap}"

    # And every fighter's first fight starts on the configured starting ELO, with nothing to decay
    first_fights = careers.groupby("fighter").first()

    assert np.allclose(first_fights["start_elo"], starting_elo)

# A fight with no winner, including an upcoming one with no outcome yet, must leave both ELOs alone

def test_derive_start_elos_ignores_fights_without_a_winner():

    import polars as pl
    from mma_functions.elo import calculate_end_elo

    constants = {k: float(v) for k, v in elo_config.items() if k != "model"}

    fights = pl.DataFrame({
        "outcome": ["W/L", "OTH", None],
        "fighter1_start_elo": [1600.0] * 3,
        "fighter2_start_elo": [1500.0] * 3,
        "fighter1_fight_number": [5] * 3,
        "fighter2_fight_number": [5] * 3,
        "control_proportion_diff": [0.0] * 3,
        "sig_strikes_landed_per_minute_diff": [0.0] * 3,
        "kds_diff": [0.0] * 3,
        "submissions_diff": [0.0] * 3,
        "knockouts_diff": [0.0] * 3,
        "unanimous_decs_diff": [0.0] * 3,
        "inverse_fight_length_proportion": [0.0] * 3,
    })

    result = calculate_end_elo(fights, constants).to_pandas()

    # The decided fight moves both ELOs
    assert result["fighter1_end_elo"][0] != result["fighter1_start_elo"][0]

    # The draw and the fight with no outcome yet leave them exactly where they were
    for row in [1, 2]:
        assert result["fighter1_end_elo"][row] == result["fighter1_start_elo"][row]
        assert result["fighter2_end_elo"][row] == result["fighter2_start_elo"][row]
