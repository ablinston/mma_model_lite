# This function takes a data frame with fight stats and a start ELO for fighters and
# computes the ELOs at the end of the fight

# NB: This function's input must be a polars frame and it returns a new frame with the ELO columns added

# The standard ELO can be calculated by settings all constants except C and K to 0

def calculate_end_elo(pl_frame, constants):

    # pdb.set_trace()

    ## Debug Kwargs

    # C = 400  # Elo rating scaling constant
    # K = 30   # K-factor (rate at which the rating adjusts)

    # # Coefficients for fight metrics differences
    # constant_control_proportion_diff = 0.5
    # constant_sig_strikes_landed_per_minute_diff = 0.3
    # constant_kds_diff = 0.7
    # constant_submissions_diff = 0.6
    # constant_knockouts_diff = 0.4
    # constant_unanimous_decs_diff = 0.2
    # constant_split_decs_diff = 0.1

    # Calculate prerequisites for the start ELO formula
    pl_frame = pl_frame.with_columns(
        fighter1_S = pl.when(pl.col('outcome') == 'W/L').then(1).otherwise(0),
        fighter1_E = (10 ** (pl.col('fighter1_start_elo') / constants['C'])) /
                     ((10 ** (pl.col('fighter1_start_elo') / constants['C'])) +
                      (10 ** (pl.col('fighter2_start_elo') / constants['C'])))
    )

    # Calculate fighter1_D, a custom metric for the ELO
    pl_frame = pl_frame.with_columns(
        fighter1_D = (
            constants['constant_control_proportion_diff'] * pl.col('control_proportion_diff') +
            constants['constant_sig_strikes_landed_per_minute_diff'] * pl.col('sig_strikes_landed_per_minute_diff') +
            constants['constant_kds_diff'] * pl.col('kds_diff') +
            constants['constant_submissions_diff'] * pl.col('submissions_diff') +
            constants['constant_knockouts_diff'] * pl.col('knockouts_diff') +
            # constants['constant_split_decs_diff'] * pl.col('split_decs_diff') +
            constants['constant_unanimous_decs_diff'] * pl.col('unanimous_decs_diff') +
            constants['constant_inverse_fight_length_proportion'] * pl.col('inverse_fight_length_proportion')
        )
    )

    # Convert the D stat for the case when fighter 2 wins. If the D stat is positive it means fighter1 was dominant, and fighter 1 will get a boost to their positive points and fighter 2 will get an increase to the negative points by using fighter1_D as well. But if fighter 2 wins and dominates, fighter1_D will be negative and cause fighter 2 ELO to decline (will be a +ve * negative D). So it needs to be -fighter1_D if fighter 2 won.
    pl_frame = pl_frame.with_columns(
        dominance_stat = pl.when(pl.col('outcome') == 'L/W').then(-pl.col('fighter1_D')).otherwise(pl.col('fighter1_D')))

    # Calculate fighter1_end_elo and fighter2_end_elo
    # Shrinkage is applied within the calculation directly
    pl_frame = pl_frame.with_columns(
        fighter1_end_elo = pl.col('fighter1_start_elo') + (constants['starting_elo'] - pl.col('fighter1_start_elo')) * constants['constants_shrinkage_factor'] + (1 + (constants['K_factor'] / pl.col('fighter1_fight_number'))) * constants['K'] * (pl.col('fighter1_S') - pl.col('fighter1_E')) * (1 + pl.col('dominance_stat')),
        fighter2_end_elo = pl.col('fighter2_start_elo') + (constants['starting_elo'] - pl.col('fighter2_start_elo')) * constants['constants_shrinkage_factor'] + (1 + (constants['K_factor'] / pl.col('fighter2_fight_number'))) * constants['K'] *  ((1 - pl.col('fighter1_S')) - (1 - pl.col('fighter1_E'))) * (1 + pl.col('dominance_stat'))
    )

    # Overwrite where the outcome wasn't W/L or L/W so that the ELO remains unchanged
    pl_frame = pl_frame.with_columns(
        fighter1_end_elo = pl.when(pl.col('outcome') == 'OTH').then(pl.col('fighter1_start_elo')).otherwise(pl.col('fighter1_end_elo')),
        fighter2_end_elo = pl.when(pl.col('outcome') == 'OTH').then(pl.col('fighter2_start_elo')).otherwise(pl.col('fighter2_end_elo')))

    return pl_frame
