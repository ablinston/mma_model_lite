# This function takes a data table of fighters and dates along with a data table of fight
# results and stats and calculates

# This function requires that all pre-requisite stats for the elos have been derived using the function
# prep_stats_for_elo_calc()

def derive_start_elos(elo_dt, fight_stats_dt, elo_constants):
    
    # pdb.set_trace()
    
    merged_data = fight_stats_dt.copy()
    start_elo = elo_dt.copy()
    
    # Do a check for NAs in the stats used for ELO
    na_count = merged_data[:, dt.isna(f.sig_strikes_landed_per_minute_diff)].sum()[0, 0]
    if na_count > 0:
        print("Error: NAs found in stats used for calculating ELO")
        return 1, 1
    
    # Convert the constants to numeric
    elo_constants = dict_to_numeric(elo_constants)
    
    # Calculate ELO for fighters first fights
    calculate_end_elo(
        merged_data,
        constants = elo_constants)

    # # Do a check on the ELO for the first fights. All should be True
    # # Check 1: Max end elo for fighter1 where outcome is 'L/W' is less than start elo

    # # Check 1: Max end elo for fighter1 where outcome is 'L/W' is less than start elo
    # check1 = merged_data[dt.f.outcome == 'L/W', dt.max(dt.f.fighter1_end_elo)].to_numpy()[0, 0] < config['start_elo']
    # # Check 2: Min end elo for fighter1 where outcome is 'W/L' is greater than start elo
    # check2 = merged_data[dt.f.outcome == 'W/L', dt.min(dt.f.fighter1_end_elo)].to_numpy()[0, 0] > config['start_elo']
    # # Check 3: Max end elo for fighter2 where outcome is 'W/L' is less than start elo
    # check3 = merged_data[dt.f.outcome == 'W/L', dt.max(dt.f.fighter2_end_elo)].to_numpy()[0, 0] < config['start_elo']
    # # Check 4: Min end elo for fighter2 where outcome is 'L/W' is greater than start elo
    # check4 = merged_data[dt.f.outcome == 'L/W', dt.min(dt.f.fighter2_end_elo)].to_numpy()[0, 0] > config['start_elo']

    # if not check1 == check2 == check3 == check4 == True:
    #     print("Checks failed. Initial ELO for this configuration does not rise and fall as expected from fight 1")
    #     return 1, 1
        
    dt.options.nthreads = 3  # Limit to a single thread if multi-threading overhead is an issue

    # Now loop through the rest of the fights in a career and fill in the ELO
    for fight in range(2, 500): #start_elo[:, f.fight_number].max()[0, 0] + 1):
        
        # First, match on the end ELO from the results data onto the next fight as the starting ELO
        # Need to get the keys the same for datatable join
        join_dt = merged_data[:, [f.fighter1, f.date, f.fighter1_end_elo]]
        join_dt.names = {'fighter1': 'fighter'}
        join_dt.key = ['date', 'fighter']
        new_start_elo = start_elo[:, :, dt.join(join_dt)]
        del(join_dt)
        
        # Repeat for fighter2
        join_dt = merged_data[:, [f.fighter2, f.date, f.fighter2_end_elo]]
        join_dt.names = {'fighter2': 'fighter'}
        join_dt.key = ['date', 'fighter']
        new_start_elo = new_start_elo[:, :, dt.join(join_dt)]
        del(join_dt)
                
        # Now combine the ELOs regardless of whether fighter 1 or 2
        new_start_elo[:, dt.update(end_elo = dt.ifelse(dt.isna(dt.f.fighter1_end_elo),
                                                   dt.f.fighter2_end_elo,
                                                   dt.f.fighter1_end_elo))]

        # Now shift the ending elo to the starting elo of the next fight
        new_start_elo = new_start_elo[:, f[:].extend({'next_start_elo': dt.shift(f.end_elo)}), dt.by('fighter')]

        # # Update start_elo based on the fight_number
        # new_start_elo[:, dt.update(start_elo = dt.ifelse(dt.f.fight_number > 1,
        #                                              dt.ifelse(dt.isna(f.next_start_elo),
        #                                                        dt.f.start_elo,
        #                                                        dt.f.next_start_elo),
        #                                              dt.f.start_elo))]
        
        # Update start_elo based on the fight_number
        # Create boolean masks as lists or numpy arrays
        mask_fight_number_gt_1 = new_start_elo[:, dt.f.fight_number > 1].to_list()[0]  # Fight number > 1
        mask_next_start_elo_na = new_start_elo[:, dt.isna(dt.f.next_start_elo)].to_list()[0]  # Next start_elo is NA
        
        # First, update rows where fight_number > 1 and next_start_elo is NA
        new_start_elo[[f and n for f, n in zip(mask_fight_number_gt_1, mask_next_start_elo_na)], 
                     dt.update(start_elo = dt.f.start_elo)]
        
        # Then, update rows where fight_number > 1 and next_start_elo is not NA
        new_start_elo[[f and not n for f, n in zip(mask_fight_number_gt_1, mask_next_start_elo_na)], 
                     dt.update(start_elo = dt.f.next_start_elo)]

        # Drop columns no longer needed
        new_start_elo = new_start_elo[:, [f.date, f.fighter, f.fight_number, f.start_elo]]
        
        ## Now merge the start ELOs onto the merged data
        
        # Drop the fighter ELO columns if they exist on the data already
        try:
            new_merged_data = merged_data[:, [col for col in merged_data.names if col not in ['fighter1_start_elo', 'fighter2_start_elo']]] 
        except:
            continue
        
        # Now join on the updated fighter start ELOs
        join_dt = new_start_elo[:, [f.date, f.fighter, f.start_elo]]
        join_dt.names = {'fighter': 'fighter1', 'start_elo': 'fighter1_start_elo'}
        join_dt.key = ['date', 'fighter1']
        new_merged_data.key = ['date', 'fighter1']
        new_merged_data = new_merged_data[:, :, dt.join(join_dt)]
        del(join_dt)
        
        # Repeat for fighter2
        join_dt = new_start_elo[:, [f.date, f.fighter, f.start_elo]]
        join_dt.names = {'fighter': 'fighter2', 'start_elo': 'fighter2_start_elo'}
        join_dt.key = ['date', 'fighter2']
        new_merged_data.key = ['date', 'fighter2']
        new_merged_data = new_merged_data[:, :, dt.join(join_dt)]
        del(join_dt)
        
        # Now calculate the end ELOs again
        calculate_end_elo(
            new_merged_data,
            constants = elo_constants)
        
        # Now repeat the loop until all ELOs have been calculated
        # Clean up memory for efficiency
        del[start_elo, merged_data]
        gc.collect()
        start_elo = new_start_elo.copy()
        merged_data = new_merged_data.copy()
        del[new_start_elo, new_merged_data]
        gc.collect()
        
        # Check how many start ELOs are missing. If none, finish the loop
        na_count = dt.isna(start_elo[:, dt.f.start_elo]).sum()[0, 0]
        if na_count > 0:
            pass
            # print(f"{na_count} ELOs missing. Continuing loop {fight}")
        else:
            return start_elo, merged_data


    print("ELOs failed to compute after 500 rounds of iteration")
    pdb.set_trace()
    return 1, 1
