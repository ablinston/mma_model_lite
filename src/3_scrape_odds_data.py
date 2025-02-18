# This script scrapes the historic betting and odds data from the UFC events

# imports
from io import StringIO
import requests
# from bs4 import BeautifulSoup
# import pandas as pd
# import numpy as np
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
import urllib.parse
# import lxml

exec(open('global.py').read())

# Get a full list of fighters
fighters = fp.ParquetFile('./data/ufc_fighter_details.parquet').to_pandas()

# Get a unique list of fighters that we want to scrape data for
fighter_list = fighters['FIRST'] + ' ' + fighters['LAST']
fighter_list = list(set(fighter_list))

# Read the fighter odds urls
fighter_odds_urls = pd.read_csv(config['fighter_odds_urls'])

# Run scraping tasks in parallel
fighter_odds = parallel_scrape_odds_fast(
    fighter_list, 
    num_threads = config['num_threads'], 
    firefox_location = config['firefox_location'], 
    gecko_driver = './data/firefox_driver_32bit/geckodriver.exe',
    odds_urls_overide_df = fighter_odds_urls,
    event_urls_only = False)

# Do it a second time to catch all those not successful, in case it was a temp issue
missing_fighters = [fighter for fighter in fighter_list if fighter not in fighter_odds['fighter'].tolist()]

fighter_odds = update_fighter_odds_data(
    fighter_odds, 
    missing_fighters, 
    pd.date_range(start = '1900-01-01', end = '2100-01-01'), 
    config, 
    num_threads = config['num_threads'])


# Do it a final time to catch all those not successful, in case it was a temp issue
missing_fighters = [fighter for fighter in fighter_list if fighter not in fighter_odds['fighter'].tolist()]

fighter_odds = update_fighter_odds_data(
    fighter_odds, 
    missing_fighters, 
    pd.date_range(start = '1900-01-01', end = '2100-01-01'), 
    config, 
    num_threads = config['num_threads'])



#########################################################################
### GET EVENTS ODDS

# Get all the event urls for full odds
event_urls_odds = parallel_scrape_odds_fast(
    fighter_list, 
    num_threads = config['num_threads'], 
    firefox_location = config['firefox_location'], 
    gecko_driver = './data/firefox_driver_32bit/geckodriver.exe',
    odds_urls_overide_df = fighter_odds_urls,
    playwright = False,
    event_urls_only = True,
    fast = False)

# Get rid of duplicate events and keep the most common date for the event
event_urls_odds_agg = event_urls_odds.groupby('event_url', as_index=False)['event_date'].apply(lambda x: x.mode().iloc[0])

# Convert the date
event_urls_odds_agg['event_date'] = pd.to_datetime(event_urls_odds_agg['event_date'].str.replace(r'(\d+)(st|nd|rd|th)', r'\1', regex=True), format = '%b %d %Y')

# Scrape the odds of each event after 2017 (odds only really start to show after then)
event_odds = parallel_scrape_event_odds_fast(
    event_urls_odds_agg[event_urls_odds_agg['event_date'] > '01-01-2018'], 
    num_threads = config['num_threads'], 
    firefox_location = config['firefox_location'], 
    gecko_driver = './data/firefox_driver_32bit/geckodriver0_35.exe',
    fast = False)

# Try to scrape again any missing events
missing_event_odds = parallel_scrape_event_odds_fast(
    event_urls_odds_agg[(event_urls_odds_agg['event_date'] > '01-01-2018') & ~(event_urls_odds_agg['event_date'].isin(event_odds['date'].tolist()))], 
    num_threads = config['num_threads'], 
    firefox_location = config['firefox_location'], 
    gecko_driver = './data/firefox_driver_32bit/geckodriver0_35.exe',
    fast = False)


#########################################################################
# Save the final files
fp.write('./data/fight_odds_data.parquet',
         fighter_odds,
         compression = 'ZSTD')

fp.write('./data/event_odds_urls.parquet', 
         event_urls_odds_agg,
         compression = 'ZSTD')

fp.write('./data/' + config['event_odds_file_name'], 
         event_odds,
         compression = 'ZSTD')