# This script takes the latest upcoming predictions and checks the bookmaker odds on oddschecker for
# the fights the model says to bet on, then says where to place each bet on the best odds after
# exchange commission.
#
# oddschecker is behind Cloudflare, so the fast scrape path gets served the challenge page. Pages come
# through a real browser instead, the same way the bestfightodds scrape does.

import glob
import os
import pandas as pd

from mma_functions.odds_scraping import find_fight_url, get_fight_urls, get_fighter_odds, get_page_source, get_subevent_markets, report_missing_bookies
from mma_functions.config import bet_config

# The listing page holding the upcoming fights
oddschecker_mma_url = "https://www.oddschecker.com/ufc-mma"

# How many of the best priced bookies to report per bet
bookies_to_show = 5


##########################################################################################
### Read the latest predictions ###
##########################################################################################

# Find the most recent predictions file written to the temp folder
latest_file = max(glob.glob("./results/temp/upcoming_predictions_*.csv"), key=os.path.getmtime)

print("Reading predictions from " + latest_file)

predictions = pd.read_csv(latest_file)

# Keep the fights the model wants a bet on. Bet_trigger holds the fighter to back, or None
bets = predictions[predictions["Bet_trigger"].notna() & (predictions["Bet_trigger"] != "None")].copy()

print(str(len(bets)) + " fights to check")


##########################################################################################
### Check the odds on oddschecker ###
##########################################################################################

# Get the fight urls once, they cover the whole card
fight_urls = get_fight_urls(get_page_source(oddschecker_mma_url))

print(str(len(fight_urls)) + " fights found on " + oddschecker_mma_url)

if len(fight_urls) == 0:
    raise RuntimeError("No fight urls on " + oddschecker_mma_url + ", so either the card is empty or the page layout has changed")

recommendations = []
bookies_reported = False

for row in bets.itertuples():

    fight_url = find_fight_url(fight_urls, row.Bet_trigger)

    if fight_url == "":
        print("No fight page found for " + row.Bet_trigger)
        continue

    markets = get_subevent_markets(get_page_source(fight_url))

    # Say once which configured bookmakers aren't on offer
    if not bookies_reported:
        report_missing_bookies(markets)
        bookies_reported = True

    odds_data = get_fighter_odds(markets, row.Bet_trigger)

    if len(odds_data) == 0:
        print("None of the configured bookies priced " + row.Bet_trigger)
        continue

    # The model's probability for the fighter being backed
    if row.Bet_trigger == row.fighter1:
        probability = row.Fighter1_win_prob / 100
    else:
        probability = row.Fighter2_win_prob / 100

    # The odds needed to still hit the EV threshold. These are compared against the odds after
    # commission, so the commission is not in the threshold as well, which would count it twice
    required_odds = (1 + bet_config["min_ev_to_bet"]) / probability

    # Take the best priced bookies in order, best first
    top_bookies = odds_data.sort_values("effective_odds", ascending=False).head(bookies_to_show)

    for place, bookie in enumerate(top_bookies.itertuples(), start=1):

        recommendations.append({
            "fighter": row.Bet_trigger,
            "place": place,
            "bookie": bookie.bookie,
            "odds": bookie.odds,
            "effective_odds": bookie.effective_odds,
            "model_odds": row.bet_assumed_odds,
            "required_odds": round(required_odds, 3),
            "ev_at_odds": round(probability * bookie.effective_odds - 1, 3),
            "stake": row.Final_bet,
            "fight_url": fight_url
            })

recommendations = pd.DataFrame(recommendations)


##########################################################################################
### Output ###
##########################################################################################

if len(recommendations) == 0:
    print("No bets could be priced")
else:
    # Whether each price still clears the EV threshold the bet was triggered on
    recommendations["still_worth_it"] = recommendations["ev_at_odds"] >= bet_config["min_ev_to_bet"]

    print(recommendations.to_string(index=False))

    recommendations.to_csv("./results/temp/bookie_odds_check.csv", index=False)
