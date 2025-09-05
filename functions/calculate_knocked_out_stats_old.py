def calculate_knocked_out_stats_old(fighter_results):
   pdb.set_trace()
    ###########################################################################################
   # Calculate the date was last knocked out
   dates_knocked_out = fighter_results.copy()
   dates_knocked_out['knocked_out'] = np.where((dates_knocked_out['method'] == 'KO/TKO') & (dates_knocked_out['result'] == 'L'), 1, 0)
   dates_knocked_out['date_knocked_out'] = np.where(dates_knocked_out['knocked_out'] == 1, dates_knocked_out['date'], np.datetime64('NaT'))

   dates_knocked_out.sort_values(by = ['fighter', 'date'], inplace = True)

   # Calculate date last knocked out
   dates_knocked_out['date_last_knocked_out'] = dates_knocked_out.groupby(['fighter'])['date_knocked_out'].shift(1)
   dates_knocked_out['date_last_knocked_out'] = dates_knocked_out.groupby(['fighter'])['date_last_knocked_out'].ffill()

   # Calculate the number times knocked out previously
   dates_knocked_out['hist_times_knocked_out'] = dates_knocked_out.groupby(['fighter'])['knocked_out'].cumsum() - dates_knocked_out['knocked_out']

   row_check = len(dates_knocked_out)

   # Calculate how many times each round they've been knocked out
   ## IT LOOKS LIKE THE HISTORIC STATS ARE CORRECT, BUT THEY ARE ONLY APPEARING WHEN A FIGHTER GETS KOd IN ROUND 1 IN THE FIGHT WE WANT TO PREDICT
   for round in [1, 2, 3, 4, 5]:
       round_knocked_out = dates_knocked_out[(dates_knocked_out['round'] == round) & dates_knocked_out['knocked_out'] == 1].copy()
       round_knocked_out['hist_times_knocked_out_round_' + str(round)] = round_knocked_out.groupby(['fighter'])['knocked_out'].cumsum() - 1
       dates_knocked_out = dates_knocked_out.merge(round_knocked_out[['event', 'bout', 'fighter', 'hist_times_knocked_out_round_' + str(round)]], 
                                                   on = ['event', 'bout', 'fighter'], 
                                                   how = 'left')
       del(round_knocked_out)

   dates_knocked_out = dates_knocked_out[['event',
                                          'bout',
                                          'fighter', 
                                          'date_last_knocked_out', 
                                          'hist_times_knocked_out',
                                          'hist_times_knocked_out_round_1',
                                          'hist_times_knocked_out_round_2',
                                          'hist_times_knocked_out_round_3',
                                          'hist_times_knocked_out_round_4',
                                          'hist_times_knocked_out_round_5']]

   # Fill NAs with 0
   dates_knocked_out.loc[:, ['hist_times_knocked_out', 'hist_times_knocked_out_round_1', 'hist_times_knocked_out_round_2', 'hist_times_knocked_out_round_3', 'hist_times_knocked_out_round_4', 'hist_times_knocked_out_round_5']] = dates_knocked_out[['hist_times_knocked_out', 'hist_times_knocked_out_round_1', 'hist_times_knocked_out_round_2', 'hist_times_knocked_out_round_3', 'hist_times_knocked_out_round_4', 'hist_times_knocked_out_round_5']].fillna(0) 

   # Fill NAs in dates with arbitrary early date
   dates_knocked_out['date_last_knocked_out'] = pd.to_datetime(dates_knocked_out['date_last_knocked_out'].fillna('1900-01-01'))
    
   return dates_knocked_out
