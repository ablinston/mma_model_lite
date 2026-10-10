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
import time

from ablinston_ml.scraping import clear_captcha, start_browser

from mma_functions.odds_scraping import clean_name_for_odds_search
from mma_functions.config import bet_config, config

# The listing page holding the upcoming fights
oddschecker_mma_url = 'https://www.oddschecker.com/ufc-mma'

# The market to read odds from, as oddschecker names it
market_to_use = 'Win Market'

# How many of the best priced bookies to report per bet
bookies_to_show = 3

# Page titles that mean the site has blocked us rather than served the page. Checked against the title
# only, because the word cloudflare on its own appears in the real pages too
blocked_page_titles = ['just a moment', 'attention required', 'verifying you are human',
                       'checking your browser', 'access denied']

# The challenge page loads this, and the real pages do not
challenge_script = 'challenges.cloudflare.com'

# The real pages run to several hundred thousand characters and a challenge page to a few thousand, so
# anything short is a block I haven't seen the wording of
min_page_characters = 50000

# Wait between pages so a card's worth of fights isn't hammered out in a few seconds
seconds_between_pages = 5


##########################################################################################
### Functions ###
##########################################################################################

def clean_bookie_name(name):

    return name.lower().replace(' ', '')


def check_page_source(url, page_source):

    """
    Stops with a message saying what came back if the page is a block or an error rather than the real
    thing. Without this a challenge page just turns into nothing found much later on.
    """

    title_match = re.search(r'<title>(.*?)</title>', page_source, re.S)
    title = title_match.group(1).strip() if title_match else ''

    print('  ' + str(len(page_source)) + ' characters, title: ' + title)

    for marker in blocked_page_titles:
        if marker in title.lower():
            raise RuntimeError('Blocked by oddschecker on ' + url + ', it returned "' + title
                               + '" instead of the page. Leave it a while before running again.')

    if challenge_script in page_source[:5000]:
        raise RuntimeError('Got a Cloudflare challenge on ' + url
                           + '. Leave it a while before running again.')

    if len(page_source) < min_page_characters:
        raise RuntimeError('Only got ' + str(len(page_source)) + ' characters from ' + url
                           + ', titled "' + title + '". Too short to be the real page.')

    if '404' in title:
        raise RuntimeError('Got a 404 page for ' + url + ', so the url has probably moved')


def get_page_source(url):

    """
    Opens a page in a real browser, clears the Cloudflare check and returns the html. A browser per
    page is slow but it is the only path that gets through.
    """

    print('Fetching ' + url)

    browser = start_browser(
        url,
        chrome_binary = config['chrome_binary_location'],
        headless = False)

    clear_captcha(browser)

    page_source = browser.get_page_source()

    browser.stop()

    check_page_source(url, page_source)

    time.sleep(seconds_between_pages)

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
        re.S)

    if payload is None:
        raise RuntimeError('No odds data in the page. The layout has changed, or the page is not a fight page')

    payload = payload.group(1).strip()

    # Hypernova wraps its payloads in an html comment
    payload = re.sub(r'-->$', '', re.sub(r'^<!--', '', payload)).strip()

    return json.loads(payload)['bestOdds']


def get_fighter_odds(markets, fighter_name):

    """
    Returns a data frame of the odds for one fighter in the win market, one row per bookmaker, with
    the odds after commission. Commission only comes off exchange winnings, and which bookmakers are
    exchanges comes from oddschecker rather than from a list I keep.
    """

    # Find the win market rather than method of victory, round betting and the rest
    win_market_ids = [market['ocMarketId'] for market in markets['markets']['entities'].values()
                      if market['marketTypeName'] == market_to_use]

    # Find the bet for this fighter in that market
    cleaned_name = clean_name_for_odds_search(fighter_name)

    bet_ids = [bet['ocBetId'] for bet in markets['bets']['entities'].values()
               if bet['marketId'] in win_market_ids and clean_name_for_odds_search(bet['betName']) == cleaned_name]

    wanted_bookies = [clean_bookie_name(bookie) for bookie in bet_config['bookies_to_use']]

    odds_rows = []

    for bet_id in bet_ids:
        for bookie_code, price in markets['odds'].get(str(bet_id), {}).items():

            bookie = markets['bookmakers']['entities'][bookie_code]

            if clean_bookie_name(bookie['bookmakerName']) not in wanted_bookies:
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


def report_missing_bookies(markets):

    """
    Says which of the configured bookmakers oddschecker isn't offering, so a name that never matches
    doesn't just quietly produce no odds.
    """

    available = [clean_bookie_name(bookie['bookmakerName']) for bookie in markets['bookmakers']['entities'].values()]

    missing = [bookie for bookie in bet_config['bookies_to_use'] if clean_bookie_name(bookie) not in available]

    if len(missing) > 0:
        print('Not offered on this page: ' + ', '.join(missing))


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

if len(fight_urls) == 0:
    raise RuntimeError('No fight urls on ' + oddschecker_mma_url + ', so either the card is empty or the page layout has changed')

recommendations = []
bookies_reported = False

for row in bets.itertuples():

    fight_url = find_fight_url(fight_urls, row.Bet_trigger)

    if fight_url == '':
        print('No fight page found for ' + row.Bet_trigger)
        continue

    markets = get_subevent_markets(get_page_source(fight_url))

    # Say once which configured bookmakers aren't on offer
    if not bookies_reported:
        report_missing_bookies(markets)
        bookies_reported = True

    odds_data = get_fighter_odds(markets, row.Bet_trigger)

    if len(odds_data) == 0:
        print('None of the configured bookies priced ' + row.Bet_trigger)
        continue

    # The model's probability for the fighter being backed
    if row.Bet_trigger == row.fighter1:
        probability = row.Fighter1_win_prob / 100
    else:
        probability = row.Fighter2_win_prob / 100

    # The odds needed to still hit the EV threshold. These are compared against the odds after
    # commission, so the commission is not in the threshold as well, which would count it twice
    required_odds = (1 + bet_config['min_ev_to_bet']) / probability

    # Take the best priced bookies in order, best first
    top_bookies = odds_data.sort_values('effective_odds', ascending = False).head(bookies_to_show)

    for place, bookie in enumerate(top_bookies.itertuples(), start = 1):

        recommendations.append({
            'fighter': row.Bet_trigger,
            'place': place,
            'bookie': bookie.bookie,
            'odds': bookie.odds,
            'effective_odds': bookie.effective_odds,
            'model_odds': row.bet_assumed_odds,
            'required_odds': round(required_odds, 3),
            'ev_at_odds': round(probability * bookie.effective_odds - 1, 3),
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
    # Whether each price still clears the EV threshold the bet was triggered on
    recommendations['still_worth_it'] = recommendations['ev_at_odds'] >= bet_config['min_ev_to_bet']

    print(recommendations.to_string(index = False))

    recommendations.to_csv('./results/temp/bookie_odds_check.csv', index = False)
