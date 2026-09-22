# Load all the packages and functions needed
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from mma_functions.features import calculate_win_loss_stats

def test_calculate_win_loss_statts():
    
    start_date = datetime(2020, 1, 1)

    # Build synthetic dataset directly
    fighters = []

    # Fighter A: always wins (KO/TKO)
    for i in range(10):
        fighters.append({
            "event": f"E{i}",
            "bout": f"B{i}_A",
            "fighter": "A",
            "date": start_date + timedelta(days=i),
            "result": "W",
            "method": "KO/TKO",
            "round": 1,
            "time format": "3 Rnd (5-5-5)",
            "rounds": 3,
            "weightclass": "LW",
        })

    # Fighter B: always loses (Submission)
    for i in range(10):
        fighters.append({
            "event": f"E{i}",
            "bout": f"B{i}_B",
            "fighter": "B",
            "date": start_date + timedelta(days=i),
            "result": "L",
            "method": "Submission",
            "round": 1,
            "time format": "3 Rnd (5-5-5)",
            "rounds": 3,
            "weightclass": "LW",
        })

    # Fighter C: alternating wins/losses (Decision)
    for i in range(10):
        fighters.append({
            "event": f"E{i}",
            "bout": f"B{i}_C",
            "fighter": "C",
            "date": start_date + timedelta(days=i),
            "result": "W" if i % 2 == 0 else "L",
            "method": "Decision",
            "round": 1,
            "time format": "3 Rnd (5-5-5)",
            "rounds": 3,
            "weightclass": "LW",
        })

    # Fighter D: mixed streaks
    results_D = ["W", "W", "L", "L", "W", "W", "L", "W", "L", "W"]
    methods_D = ["KO/TKO", "Decision", "Decision", "Submission", "Decision",
                 "KO/TKO", "KO/TKO", "Decision", "Submission", "KO/TKO"]
    for i, (res, method) in enumerate(zip(results_D, methods_D)):
        fighters.append({
            "event": f"E{i}",
            "bout": f"B{i}_D",
            "fighter": "D",
            "date": start_date + timedelta(days=i),
            "result": res,
            "method": method,
            "round": 1,
            "time format": "3 Rnd (5-5-5)",
            "rounds": 3,
            "weightclass": "LW",
        })

    df = pd.DataFrame(fighters)

    # Run your function
    out = calculate_win_loss_stats(df, last_x_fights=10)

    # -------------------
    # Assertions
    # -------------------

    # Fighter A: only wins
    A = out[out["fighter"] == "A"].sort_values("date")
    assert (A["hist_losses"] == 0).all()
    assert A["hist_wins"].tolist() == list(range(10))
    # Rolling win% should all be 1.0 after enough fights
    assert A["hist_win_percent_5_fights"].iloc[5:].notna().all()
    assert np.isclose(A["hist_win_percent_5_fights"].dropna(), 1.0).all()

    # Fighter B: only losses
    B = out[out["fighter"] == "B"].sort_values("date")
    assert (B["hist_wins"] == 0).all()
    assert B["hist_losses"].tolist() == list(range(10))
    # Rolling win% should all be 0
    assert np.isclose(B["hist_win_percent_5_fights"].dropna(), 0.0).all()

    # Fighter C: alternating
    C = out[out["fighter"] == "C"].sort_values("date")
    expected_wins = [0,1,1,2,2,3,3,4,4,5]
    expected_losses = [0,0,1,1,2,2,3,3,4,4]
    assert C["hist_wins"].tolist() == expected_wins
    assert C["hist_losses"].tolist() == expected_losses
    # Rolling win% of last 2 fights should alternate between 0.5 and 0.0/1.0
    assert set(np.round(C["hist_win_percent_2_fights"].dropna().unique(),2)) <= {0.0, 0.5, 1.0}

    # Fighter D: mixed
    D = out[out["fighter"] == "D"].sort_values("date")
    assert D["hist_wins"].iloc[0] == 0
    assert D["hist_losses"].iloc[0] == 0
    assert D["hist_wins"].iloc[-1] == results_D.count("W") - (1 if results_D[-1]=="W" else 0)
    assert D["hist_losses"].iloc[-1] == results_D.count("L") - (1 if results_D[-1]=="L" else 0)
    # By the end, rolling 10 fights win% should be total wins / 10
    final_win_percent = D["hist_win_percent_9_fights"].iloc[-1]
    assert np.isclose(final_win_percent, results_D[:-1].count("W")/9)

    
    # Fighter A: first fight should have NaN for rolling 5-fight win%
    assert pd.isna(A["hist_win_percent_5_fights"].iloc[0])

    # Fighter B: first fight should have NaN for rolling 3-fight win%
    assert pd.isna(B["hist_win_percent_1_fights"].iloc[0])

    # Fighter C: second fight should still be NaN for rolling 3-fight win%
    assert pd.isna(C["hist_win_percent_3_fights"].iloc[1])
