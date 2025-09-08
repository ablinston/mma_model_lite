# Load all the packages and functions needed
exec(open('global.py').read())

test_calculate_hist_opp_stance_stats_correct = fp.ParquetFile('./tests/data/test_calculate_hist_opp_stance_stats_correct.parquet').to_pandas()

def test_calculate_hist_opp_stance_stats():
    
    # Fighters with different stances
    fighters = pd.DataFrame({
        "fighter": ["Alice", "Bob", "Charlie"],
        "stance": ["Orthodox", "Southpaw", "Switch"]
    })

    # Build 20 bouts, each with 2 rows (one per fighter)
    bouts = []
    sig_values = list(range(1, 41))  # 40 rows of sig strikes
    idx = 0
    for bout_id in range(1, 21):
        date = pd.Timestamp("2022-01-01") + pd.offsets.MonthBegin(bout_id)
        if bout_id % 3 == 1:
            f1, f2 = "Alice", "Bob"
        elif bout_id % 3 == 2:
            f1, f2 = "Bob", "Charlie"
        else:
            f1, f2 = "Charlie", "Alice"

        bouts.append({
            "date": date,
            "event": f"Event{bout_id}",
            "bout": bout_id,
            "weightclass": "Lightweight",
            "fighter": f1,
            "opponent": f2,
            "sig_strikes": sig_values[idx] if sig_values[idx] % 7 != 0 else None
        })
        idx += 1
        bouts.append({
            "date": date,
            "event": f"Event{bout_id}",
            "bout": bout_id,
            "weightclass": "Lightweight",
            "fighter": f2,
            "opponent": f1,
            "sig_strikes": sig_values[idx] if sig_values[idx] % 7 != 0 else None
        })
        idx += 1

    fight_stats = pd.DataFrame(bouts)

    id_cols = ['event', 'bout', 'fighter']
    group_by = ['weightclass']
    fight_span = 2  # Using a larger span for robust testing

    # Priors (not used in mock, but passed through)
    fight_stats_bayes_priors = bayesian_priors(fight_stats, id_cols, group_by, fight_span, ignore_last_n_fights = 1)

    data_processing_config = {"bayes_alpha_plus_beta_rebase_per_minute": 10}

    # Run function
    result = calculate_hist_opp_stance_stats(
        fight_stats.drop(columns = 'weightclass'), fighters, fight_stats_bayes_priors, data_processing_config
    )

    # --- Assertions ---
    # Shape must match
    assert test_calculate_hist_opp_stance_stats_correct.sort_values(['event', 'fighter']).equals(result.sort_values(['event', 'fighter'])[test_calculate_hist_opp_stance_stats_correct.columns])


# fp.write('./tests/data/test_calculate_hist_opp_stance_stats_correct.parquet', result, compression = "ZSTD")
