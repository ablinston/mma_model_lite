# This script takes the latest upcoming predictions and checks the bookmaker odds on oddschecker for
# the fights the model says to bet on, then says where to place each bet on the best odds after
# exchange commission.
#
# oddschecker is behind Cloudflare, so the fast scrape path gets served the challenge page. Pages have
# to come through a real browser, the same way the bestfightodds scrape does.

import glob
import json
import os
import pandas as pd
import re

from ablinston_ml.scraping import clear_captcha, start_browser

from mma_functions.odds_scraping import clean_name_for_odds_search
from mma_functions.config import bet_config, config

# Replace the chrome location if we're not on windows
if not os.name == "nt":
    config['chrome_binary_location'] = config['chrome_binary_location_linux']

# The bookies to consider, matched against oddschecker's own names, case insensitive. This is a dummy
# until the list goes in the config file. Note oddschecker lists Betfair the exchange and Betfair
# Sportsbook separately, so 'Betfair' here means the exchange only
bookies_to_use = ['Bet365', 'Betfair']

# The listing page holding the upcoming fights
oddschecker_mma_url = 'https://www.oddschecker.com/ufc-mma'

# The market to read odds from, as oddschecker names it
market_to_use = 'Win Market'


##########################################################################################
### Functions ###
##########################################################################################

def get_page_source(url):

    """
    Opens a page in a real browser, clears the Cloudflare check and returns the html. A browser per
    page is slow but it is the only path that gets through.
    """

    browser = start_browser(
        url,
        chrome_binary = config['chrome_binary_location'],
        headless = False)

    clear_captcha(browser)

    page_source = browser.get_page_source()

    browser.stop()

    return page_source


def get_fight_urls(page_source):

    """
    Pulls the fight page urls off an oddschecker mma listing page. They look like
    /ufc-mma/ufc/ernesta-kareckaite-v-melissa-gatto/winner, so the fighter names are in the url itself.
    Returns a dict of the two cleaned fighter names to the full url.
    """

    urls = {}

    for path in set(re.findall(r'"(/ufc-mma/[a-z0-9-]+/[a-z0-9-]+-v-[a-z0-9-]+/winner)"', page_source)):

        slug = path.split('/')[-2]
        names = tuple(clean_name_for_odds_search(name) for name in slug.split('-v-'))

        urls[names] = 'https://www.oddschecker.com' + path

    return urls


def find_fight_url(fight_urls, fighter_name):

    """
    Finds the fight url where one of the two fighters in the url matches the fighter being backed.
    Returns an empty string if the fighter isn't on the card.
    """

    cleaned_name = clean_name_for_odds_search(fighter_name)

    for names, url in fight_urls.items():
        if cleaned_name in names:
            return url

    return ''


def get_subevent_markets(page_source):

    """
    Pulls the odds data out of an oddschecker fight page. The page is server rendered React, so the
    whole thing sits in a json payload in a script tag rather than in the html itself.
    """

    payload = re.search(
        r'<script type="application/json" data-hypernova-key="subeventmarkets"[^>]*>(.*?)</script>',
        page_source,
        re.S).group(1).strip()

    # Hypernova wraps its payloads in an html comment
    payload = re.sub(r'-->$', '', re.sub(r'^<!--', '', payload)).strip()

    return json.loads(payload)['bestOdds']


def get_fighter_odds(markets, fighter_name):

    """
    Returns a data frame of the odds for one fighter in the win market, one row per bookmaker, with
    the odds after commission. Commission only comes off exchange winnings, which matches the
    commission used in calculate_target_odds.
    """

    # Find the win market rather than method of victory, round betting and the rest
    win_market_ids = [market['ocMarketId'] for market in markets['markets']['entities'].values()
                      if market['marketTypeName'] == market_to_use]

    # Find the bet for this fighter in that market
    cleaned_name = clean_name_for_odds_search(fighter_name)

    bet_ids = [bet['ocBetId'] for bet in markets['bets']['entities'].values()
               if bet['marketId'] in win_market_ids and clean_name_for_odds_search(bet['betName']) == cleaned_name]

    wanted_bookies = [bookie.lower() for bookie in bookies_to_use]

    odds_rows = []

    for bet_id in bet_ids:
        for bookie_code, price in markets['odds'].get(str(bet_id), {}).items():

            bookie = markets['bookmakers']['entities'][bookie_code]

            if bookie['bookmakerName'].lower() not in wanted_bookies:
                continue

            if price['status'] != 'ACTIVE' or price['oddsDecimal'] is None:
                continue

            # Take the commission off the winnings for an exchange, leaving the stake alone
            if bookie['bookmakerType'] == 'exchange':
                effective_odds = 1 + (price['oddsDecimal'] - 1) * (1 - bet_config['exchange_commission'])
            else:
                effective_odds = price['oddsDecimal']

            odds_rows.append({
                'bookie': bookie['bookmakerName'],
                'is_exchange': bookie['bookmakerType'] == 'exchange',
                'odds': price['oddsDecimal'],
                'effective_odds': round(effective_odds, 3)
                })

    return pd.DataFrame(odds_rows)


##########################################################################################
### Read the latest predictions ###
##########################################################################################

# Find the most recent predictions file written to the temp folder
latest_file = max(glob.glob('./results/temp/upcoming_predictions_*.csv'), key = os.path.getmtime)

print('Reading predictions from ' + latest_file)

predictions = pd.read_csv(latest_file)

# Keep the fights the model wants a bet on. Bet_trigger holds the fighter to back, or None
bets = predictions[predictions['Bet_trigger'].notna() & (predictions['Bet_trigger'] != 'None')].copy()

print(str(len(bets)) + ' fights to check')


##########################################################################################
### Check the odds on oddschecker ###
##########################################################################################

# Get the fight urls once, they cover the whole card
fight_urls = get_fight_urls(get_page_source(oddschecker_mma_url))

print(str(len(fight_urls)) + ' fights found on ' + oddschecker_mma_url)

recommendations = []

for row in bets.itertuples():

    fight_url = find_fight_url(fight_urls, row.Bet_trigger)

    if fight_url == '':
        print('No fight page found for ' + row.Bet_trigger)
        continue

    odds_data = get_fighter_odds(get_subevent_markets(get_page_source(fight_url)), row.Bet_trigger)

    if len(odds_data) == 0:
        print('None of ' + str(bookies_to_use) + ' priced ' + row.Bet_trigger)
        continue

    best = odds_data.loc[odds_data['effective_odds'].idxmax()]

    recommendations.append({
        'fighter': row.Bet_trigger,
        'model_odds': row.bet_assumed_odds,
        'bookie': best['bookie'],
        'odds': best['odds'],
        'effective_odds': best['effective_odds'],
        'stake': row.Final_bet,
        'fight_url': fight_url
        })

recommendations = pd.DataFrame(recommendations)


##########################################################################################
### Output ###
##########################################################################################

if len(recommendations) == 0:
    print('No bets could be priced')
else:
    # Flag anything where the best price available now is worse than the odds the bet was triggered on
    recommendations['still_worth_it'] = recommendations['effective_odds'] >= recommendations['model_odds']

    print(recommendations.to_string(index = False))

    recommendations.to_csv('./results/temp/bookie_odds_check.csv', index = False)
