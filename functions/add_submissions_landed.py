# This function adds the stats for submissions landed to the stats data using the results data

def add_submissions_landed(fight_stats, fight_results):
        
    # pdb.set_trace()
    
    # Get submission success
    fighter1_submissions = fight_results[['event', 'bout', 'fighter1', 'outcome', 'method', 'round']].rename(columns = {'fighter1': 'fighter'})
    fighter2_submissions = fight_results[['event', 'bout', 'fighter2', 'outcome', 'method', 'round']].rename(columns = {'fighter2': 'fighter'})

    # Create a columne which shows the round of the submission
    fighter1_submissions['submission'] = np.where(
        (fighter1_submissions['outcome'] == 'W/L') & (fighter1_submissions['method'] == 'Submission'), 
        fighter1_submissions['round'], 
        0
    )
    
    fighter2_submissions['submission'] = np.where(
        (fighter2_submissions['outcome'] == 'L/W') & (fighter2_submissions['method'] == 'Submission'), 
        fighter2_submissions['round'], 
        0
    )
    
    # Concat fighter1 and fighter2
    submissions = pd.concat([
        fighter1_submissions,
        fighter2_submissions
        ],
        axis = 0)
    
    # Merge onto the stats data
    new_stats = fight_stats.merge(
        submissions[['event', 'bout', 'fighter', 'submission']],
        on = ['event', 'bout', 'fighter'], 
        how = 'left')
    
    # Check there are no NAs after 2010. Settled for 10 because some tiny misalignments are too cumbersome to fix
    stop_if_not(new_stats[new_stats['date'] > '2010-01-01']['submission'].isna().sum() < 10)
        
    # Create round columns
    for round in range(1, 6):
        new_stats['round' + str(round) + '_sub_landed'] = np.where(new_stats['submission'] == round, 1, 0)
        
        # CHeck that landed isn't > attempts and if it is, replace it. (This only happens in a handful of cases, so not a systemic issue)
        new_stats['round' + str(round) + '_sub_attempt'] = np.where(
            new_stats['round' + str(round) + '_sub_landed'] > new_stats['round' + str(round) + '_sub_attempt'],
            new_stats['round' + str(round) + '_sub_landed'],
            new_stats['round' + str(round) + '_sub_attempt']
        )
    
    new_stats['allrounds_sub_landed'] = new_stats[[col for col in new_stats.columns if 'sub_landed' in col]].sum(axis = 1)
    
    # CHeck that landed isn't > attempts and if it is, replace it
    new_stats['allrounds_sub_attempt'] = np.where(
        new_stats['allrounds_sub_landed'] > new_stats['allrounds_sub_attempt'],
        new_stats['allrounds_sub_landed'],
        new_stats['allrounds_sub_attempt']
    )
    del([fighter1_submissions, fighter2_submissions])
    
    # Cleanup
    new_stats.drop(columns = ['submission'], inplace = True)
    
    # Final checks
    stop_if_not(new_stats['allrounds_sub_landed'].max() == 1, "Max submissions landed check")
    stop_if_not(len(new_stats) == len(fight_stats), "Data length subs landed")
    
    return new_stats
    
    