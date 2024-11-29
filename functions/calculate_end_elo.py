# This function takes a data frame with fight stats and a start ELO for fighters and 
# computes the ELOs at the end of the fight

# NB: This function's input must be a datatable object and it is a mutable operation i.e. it changes the underlying data table and has no return

# The standard ELO can be calculated by settings all constants except C and K to 0

def calculate_end_elo(dt_frame, constants):

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
    dt_frame[:, dt.update(
        fighter1_S=dt.ifelse(dt.f.outcome == 'W/L', 1, 0),
        fighter1_E=(10 ** (dt.f.fighter1_start_elo / constants['C'])) / 
                   ((10 ** (dt.f.fighter1_start_elo / constants['C'])) + 
                    (10 ** (dt.f.fighter2_start_elo / constants['C'])))
    )]
    
    # Calculate fighter1_D, a custom metric for the ELO
    dt_frame[:, dt.update(
        fighter1_D=(
            constants['constant_control_proportion_diff'] * dt.f.control_proportion_diff + 
            constants['constant_sig_strikes_landed_per_minute_diff'] * dt.f.sig_strikes_landed_per_minute_diff + 
            constants['constant_kds_diff'] * dt.f.kds_diff + 
            constants['constant_submissions_diff'] * dt.f.submissions_diff + 
            constants['constant_knockouts_diff'] * dt.f.knockouts_diff + 
            # constants['constant_split_decs_diff'] * dt.f.split_decs_diff +
            constants['constant_unanimous_decs_diff'] * dt.f.unanimous_decs_diff +
            constants['constant_inverse_fight_length_proportion'] * dt.f.inverse_fight_length_proportion
        )
    )]
    
    # Convert the D stat for the case when fighter 2 wins. If the D stat is positive it means fighter1 was dominant, and fighter 1 will get a boost to their positive points and fighter 2 will get an increase to the negative points by using fighter1_D as well. But if fighter 2 wins and dominates, fighter1_D will be negative and cause fighter 2 ELO to decline (will be a +ve * negative D). So it needs to be -fighter1_D if fighter 2 won.
    dt_frame[:, dt.update(dominance_stat = dt.ifelse(f.outcome == 'L/W', -f.fighter1_D, f.fighter1_D))]
        
    # Calculate fighter1_end_elo and fighter2_end_elo
    # Shrinkage is applied within the calculation directly
    dt_frame[:, dt.update(
        fighter1_end_elo=dt.f.fighter1_start_elo + (constants['starting_elo'] - dt.f.fighter1_start_elo) * constants['constants_shrinkage_factor'] + (1 + (constants['K_factor'] / dt.f.fighter1_fight_number)) * constants['K'] * (dt.f.fighter1_S - dt.f.fighter1_E) * (1 + dt.f.dominance_stat),
        fighter2_end_elo=dt.f.fighter2_start_elo + (constants['starting_elo'] - dt.f.fighter2_start_elo) * constants['constants_shrinkage_factor'] + (1 + (constants['K_factor'] / dt.f.fighter2_fight_number)) * constants['K'] *  ((1 - dt.f.fighter1_S) - (1 - dt.f.fighter1_E)) * (1 + dt.f.dominance_stat)
    )]
    
    # Overwrite where the outcome wasn't W/L or L/W so that the ELO remains unchanged
    dt_frame[:, dt.update(
        fighter1_end_elo = dt.ifelse(f.outcome == 'OTH', f.fighter1_start_elo, f.fighter1_end_elo),
        fighter2_end_elo = dt.ifelse(f.outcome == 'OTH', f.fighter2_start_elo, f.fighter2_end_elo))]

