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

fp.write('./data/fight_odds_data.parquet',
         fighter_odds,
         compression = 'GZIP')

# # Close the browser
# driver.quit()
#    Zoila Frausto Gurgel
# # Put the urls into the data frame
# events['fight_odds_url'] =  fightodds_url

# events.sort_values('Date', inplace = True)
# events.tail(20)

# # get coverage of how many ufc events in Wikipedia have bestfightodds urls in each year
# events['fight_odds_url'].groupby(events['Date'].apply(lambda x: x.year)).apply(lambda x: (x != '').sum())/events['fight_odds_url'].groupby(events['Date'].apply(lambda x: x.year)).size()
