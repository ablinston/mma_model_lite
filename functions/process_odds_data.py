# This function processes the cleaned odds data so that it is ready for merging with the other datasets

def process_odds_data(df):
    
   # Convert opening odds to decimal
   df['open_odds'] = df['open_odds'].apply(american_odds_to_decimal)

   # Record the best odds as the highest (as that is what will give the biggest return)
   df['best_closing_odds'] = df['closing_odds_high'].apply(american_odds_to_decimal)

   # Calculate the odds change
   df['odds_change'] = df['best_closing_odds'] - df['open_odds']

   # Keep only the columns we need, and remove duplicates (these duplicates are where the event and bout are empty)
   df = df[['event', 'bout', 'fighter', 'open_odds', 'best_closing_odds', 'odds_change']].drop_duplicates(subset = ['event', 'bout', 'fighter'])
    
   return df
