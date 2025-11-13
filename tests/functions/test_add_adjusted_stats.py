# Load all the packages and functions needed
exec(open('global.py').read())

# # Load what answers are expected
test_add_adjusted_bayes_stats_correct = fp.ParquetFile('./tests/data/test_add_adjusted_bayes_stats_correct.parquet').to_pandas()

def test_add_adjusted_stats():
    # --- Build stats_df with 10 bouts (20 rows) ---
    fighters = ["AlwaysWins", "AlwaysLoses", "MixedA", "MixedB", "Random1", "Random2"]
    rows = []
    for i in range(10):
        if i % 3 == 0:
            f1, f2 = "AlwaysWins", "AlwaysLoses"
        elif i % 3 == 1:
            f1, f2 = "MixedA", "MixedB"
        else:
            f1, f2 = "Random1", "Random2"
        bout_name = f"{f1}vs.{f2}"
        rows.append({
            "event": f"event{i}",
            "bout": bout_name,
            "fighter": f1,
            "bayes_strikes_per_minute": 10 + i,
            "bayes_strikes_against_per_minute": 10 + i
        })
        rows.append({
            "event":  f"event{i}",
            "bout": bout_name,
            "fighter": f2,
            "bayes_strikes_per_minute": 5,
            "bayes_strikes_against_per_minute": 10 + i
        })

    stats_df = pd.DataFrame(rows)

    # Scramble the index to ensure robustness
    stats_df = stats_df.sample(frac=1).reset_index(drop=True).sort_values(['event', 'fighter'])

    # --- Build hist_df: same shape, event = bout ---
    hist_rows = []
    for _, row in stats_df.iterrows():
        f = row["fighter"]
        hist_rows.append({
            "event": row["event"],   # match bout
            "bout": row["bout"],    # match bout
            "fighter": f,
            "hist_bayes_strikes_per_minute": 0 if f == "AlwaysWins" else (100 if f == "AlwaysLoses" else 10),
            "hist_bayes_strikes_against_per_minute": 100 if f == "AlwaysWins" else (0 if f == "AlwaysLoses" else 20),
        })
    hist_df = pd.DataFrame(hist_rows)

    # --- Run function ---
    result = add_adjusted_stats(stats_df, hist_df, columns_with=["bayes_strikes"]).sort_values(['event', 'fighter'])

    # --- Assertions ---
    # Shape preserved
    assert test_add_adjusted_bayes_stats_correct.reset_index(drop = True).equals(result.reset_index(drop = True))

# fp.write('./tests/data/test_add_adjusted_bayes_stats_correct.parquet', result, compression = 'ZSTD')
