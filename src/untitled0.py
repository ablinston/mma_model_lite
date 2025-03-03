

def sample_fights(df, n_rows):
    # Get the unique fighters
    unique_fighters = pd.concat([df['notformodel_fighter1'], df['notformodel_fighter2']]).unique()
    
    # Initialize the list for the resulting rows
    sampled_data = []
    
    # Keep track of the number of rows added
    while len(sampled_data) < n_rows:
        # Sample fighters with replacement
        sampled_fighters = np.random.choice(unique_fighters, size=n_rows, replace=True)
        
        # For each sampled fighter, add their corresponding fight data
        for fighter in sampled_fighters:
            fighter_fights = df[(df['notformodel_fighter1'] == fighter) | (df['notformodel_fighter2'] == fighter)]
            sampled_data.append(fighter_fights)
            
            # If we've reached the desired number of rows, break early
            if len(sampled_data) >= n_rows:
                break
    
    # Concatenate all the sampled data and slice it to the correct size (n_rows)
    sampled_df = pd.concat(sampled_data).head(n_rows).reset_index(drop=True)
    return sampled_df


def sample_fights(df, n_rows):
    
    # Create a series of all the fighters
    fighters = pd.concat([df['notformodel_fighter1'], df['notformodel_fighter2']])

    # Count how many fights they have and shuffle
    fighters_counts = pd.DataFrame(fighters.value_counts().sample(frac = 1)).reset_index()
    
    # Now do cumulative sums until I get the required rows
    fighters_counts['cumsums'] = fighters_counts['count'].cumsum()
    
    # Create a flag for which fighters will be in the selected data
    fighters_counts['select_flag'] = np.where(fighters_counts['cumsums'] < n_rows, 1, 0)

    # Get the selected fighters
    selected_fighters = fighters_counts[fighters_counts['select_flag'] == 1]['index'].tolist()

    # Now return the selected data
    return df[df['notformodel_fighter1'].isin(selected_fighters) | df['notformodel_fighter1'].isin(selected_fighters)]
    
    # Create an array of corresponding indices for each fight
    fight_indices = np.concatenate([df.index.values, df.index.values])
    
    
    # Group by fighter to get all fights for each fighter
    fighter_groups = df.groupby([df['notformodel_fighter1'], df['notformodel_fighter2']])
    
    # Get the fighter groups as a list of DataFrames
    fighter_group_list = [group for _, group in fighter_groups]
    
    # Randomly sample fighter indices with replacement
    sampled_indices = np.random.choice(fight_indices, size=n_rows, replace=True)
    
    # Use the sampled indices to get the corresponding rows from the original DataFrame
    sampled_df = df.loc[sampled_indices].reset_index(drop=True)
    
    return sampled_df

te = sample_fights(X, int(len(X) * subsample))


df = X

