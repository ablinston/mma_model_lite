# Load all the packages and functions needed
from datetime import timedelta
import fastparquet as fp
import numpy as np
import pandas as pd

from mma_functions.features import aggregate_last_X_fight_data, calculate_adjusted_percentages

# # Load what answers are expected. This was manually calculated by me in Excel and is correct.
test_calculate_adjusted_percentage_correct = fp.ParquetFile('./tests/data/test_calculate_adjusted_percentage_correct.parquet').to_pandas()

def test_calculate_adjusted_percentages():
    
    np.random.seed(42)

    # Create dummy fight data
    fighters = ['Fighter_A', 'Fighter_B']
    bouts = [f"Bout_{i}" for i in range(1, 11)]

    fight_stats_records = []
    for i, bout in enumerate(bouts):
        for fighter in fighters:
            strikes_attempt = np.random.randint(20, 100)
            # ensure landed < attempt
            strikes_landed = np.random.randint(0, strikes_attempt)
            fight_stats_records.append({
                'event': 'event' + str(i),
                'bout': 'Fighter_Avs.Fighter_B',
                'date': pd.to_datetime('2020-01-01') + timedelta(days = i),
                'fighter': fighter,
                'strikes_attempt': strikes_attempt,
                'strikes_landed': strikes_landed,
            })

    fight_stats = pd.DataFrame(fight_stats_records)
    
    # Copy the fight stats but for against a fighter, then merge onto stats
    fight_stats_against = fight_stats.drop(columns = ['date'])
    fight_stats_against.rename(columns = lambda x: x + "_against" if "attempt" in x else x, inplace = True)
    fight_stats_against.rename(columns = lambda x: x + "_against" if "landed" in x else x, inplace = True)
    fight_stats_against.rename(columns = lambda x: x + "_against" if "ctrl_minutes" in x else x, inplace = True)
    fight_stats_against.rename(columns = lambda x: x + "_against" if "kd" in x else x, inplace = True)

    # Switch the two fighters around
    fight_stats_against['fighter'] = fight_stats_against['fighter'].replace('Fighter_A', 'Fighter_X').replace('Fighter_B', 'Fighter_A').replace('Fighter_X', 'Fighter_B')
    
    # Merge onto stats
    fight_stats = fight_stats.merge(fight_stats_against, on = ['event', 'bout', 'fighter'], how = 'left')
    
    # Create fight history for one fighter (10 fights)
    fight_stats_hist = aggregate_last_X_fight_data(
        fight_stats,
        id_cols = ['date', 'event', 'bout', 'fighter'],
        group_by = ['fighter'],
        prefix = "hist_",
        X = 100,
        per_minute_bayes = True,
        per_minute_calc = True, 
        bayes_priors_df = None,
        bayes_alpha_plus_beta_rebase = 100
    ).drop(columns = ['date'])

    data_processing_config = {
        'bayes_alpha_plus_beta_rebase_percent': 100,
        'add_adj_percentage_columns': True,
        'add_adj_percentage_bayes_columns': True
    }

    # Run the function
    result = calculate_adjusted_percentages(
        fight_stats,
        fight_stats_hist,
        None,
        data_processing_config
    )

    # Assertions
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(fight_stats)
    assert result.equals(test_calculate_adjusted_percentage_correct)


# fp.write('./tests/data/test_calculate_adjusted_percentage_correct.parquet', result, compression = 'ZSTD')
