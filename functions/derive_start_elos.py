# This function takes a data frame of fighters and dates along with a data frame of fight
# results and stats and calculates

# This function requires that all pre-requisite stats for the elos have been derived using the function
# prep_stats_for_elo_calc()

def derive_start_elos(elo_df, fight_stats_df, elo_constants):

    # pdb.set_trace()

    merged_data = fight_stats_df.clone()
    start_elo = elo_df.clone()

    # Do a check for NAs in the stats used for ELO
    na_count = merged_data['sig_strikes_landed_per_minute_diff'].is_null().sum()
    if na_count > 0:
        print("Error: NAs found in stats used for calculating ELO")
        return 1, 1

    # Convert the constants to numeric
    elo_constants = dict_to_numeric(elo_constants)

    # Calculate ELO for fighters first fights
    merged_data = calculate_end_elo(
        merged_data,
        constants = elo_constants)

    # # Do a check on the ELO for the first fights. All should be True
    # # Check 1: Max end elo for fighter1 where outcome is 'L/W' is less than start elo

    # # Check 1: Max end elo for fighter1 where outcome is 'L/W' is less than start elo
    # check1 = merged_data.filter(pl.col('outcome') == 'L/W')['fighter1_end_elo'].max() < config['start_elo']
    # # Check 2: Min end elo for fighter1 where outcome is 'W/L' is greater than start elo
    # check2 = merged_data.filter(pl.col('outcome') == 'W/L')['fighter1_end_elo'].min() > config['start_elo']
    # # Check 3: Max end elo for fighter2 where outcome is 'W/L' is less than start elo
    # check3 = merged_data.filter(pl.col('outcome') == 'W/L')['fighter2_end_elo'].max() < config['start_elo']
    # # Check 4: Min end elo for fighter2 where outcome is 'L/W' is greater than start elo
    # check4 = merged_data.filter(pl.col('outcome') == 'L/W')['fighter2_end_elo'].min() > config['start_elo']

    # if not check1 == check2 == check3 == check4 == True:
    #     print("Checks failed. Initial ELO for this configuration does not rise and fall as expected from fight 1")
    #     return 1, 1

    # Now loop through the rest of the fights in a career and fill in the ELO
    for fight in range(2, 500): #start_elo['fight_number'].max() + 1):

        # First, match on the end ELO from the results data onto the next fight as the starting ELO
        join_df = merged_data.select(['date', 'fighter1', 'fighter1_end_elo']).rename({'fighter1': 'fighter'})
        new_start_elo = start_elo.join(join_df, on = ['date', 'fighter'], how = 'left', maintain_order = 'left')

        # Repeat for fighter2
        join_df = merged_data.select(['date', 'fighter2', 'fighter2_end_elo']).rename({'fighter2': 'fighter'})
        new_start_elo = new_start_elo.join(join_df, on = ['date', 'fighter'], how = 'left', maintain_order = 'left')

        # Now combine the ELOs regardless of whether fighter 1 or 2
        new_start_elo = new_start_elo.with_columns(
            end_elo = pl.when(pl.col('fighter1_end_elo').is_null())
                        .then(pl.col('fighter2_end_elo'))
                        .otherwise(pl.col('fighter1_end_elo')))

        # Now shift the ending elo to the starting elo of the next fight. The frame is in
        # date order within a fighter, so a plain lag gives the previous fight
        new_start_elo = new_start_elo.with_columns(
            next_start_elo = pl.col('end_elo').shift().over('fighter'))

        # Update start_elo where the fighter has a previous fight to carry the ELO over from
        new_start_elo = new_start_elo.with_columns(
            start_elo = pl.when((pl.col('fight_number') > 1) & pl.col('next_start_elo').is_not_null())
                          .then(pl.col('next_start_elo'))
                          .otherwise(pl.col('start_elo')))

        # Drop columns no longer needed
        new_start_elo = new_start_elo.select(['date', 'fighter', 'fight_number', 'start_elo'])

        ## Now merge the start ELOs onto the merged data

        # Drop the fighter ELO columns if they exist on the data already
        try:
            new_merged_data = merged_data.drop(['fighter1_start_elo', 'fighter2_start_elo'])
        except:
            continue

        # Now join on the updated fighter start ELOs
        join_df = new_start_elo.select(['date', 'fighter', 'start_elo']).rename({'fighter': 'fighter1', 'start_elo': 'fighter1_start_elo'})
        new_merged_data = new_merged_data.join(join_df, on = ['date', 'fighter1'], how = 'left', maintain_order = 'left')

        # Repeat for fighter2
        join_df = new_start_elo.select(['date', 'fighter', 'start_elo']).rename({'fighter': 'fighter2', 'start_elo': 'fighter2_start_elo'})
        new_merged_data = new_merged_data.join(join_df, on = ['date', 'fighter2'], how = 'left', maintain_order = 'left')

        # Now calculate the end ELOs again
        new_merged_data = calculate_end_elo(
            new_merged_data,
            constants = elo_constants)

        # Now repeat the loop until all ELOs have been calculated
        # Clean up memory for efficiency
        start_elo = new_start_elo
        merged_data = new_merged_data
        del[new_start_elo, new_merged_data, join_df]
        gc.collect()

        # Check how many start ELOs are missing. If none, finish the loop
        na_count = start_elo['start_elo'].is_null().sum()
        if na_count > 0:
            pass
            # print(f"{na_count} ELOs missing. Continuing loop {fight}")
        else:
            return start_elo, merged_data


    print("ELOs failed to compute after 500 rounds of iteration")
    pdb.set_trace()
    return 1, 1
