# This script scrapes the historic betting and odds data from the UFC events

# imports
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
    odds_urls_overide_df = fighter_odds_urls)


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


# Save the final file
fp.write('./data/fight_odds_data.parquet',
         fighter_odds,
         compression = 'GZIP')