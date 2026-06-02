# Import functions from my scraper library
import sys
import os

if os.name == "nt":
    sys.path.append('X:/Repos/custom_tools')
else:
    sys.path.append('/home/andy/Repos/custom_tools')

from model_tools.k_fold_id_by_runner import k_fold_id_by_runner
from model_tools.TemperatureScaler import TemperatureScaler
from scraping_tools.get_website_soup import get_website_soup
# from scraping_tools.parallel_scrape_and_process_urls import parallel_scrape_and_process_urls

# from scraping_tools.parallel_scrape_and_process_urls import parallel_scrape_and_process_urls

import betfairlightweight
from betfairlightweight import StreamListener
from betfairlightweight.filters import market_filter, price_projection, streaming_market_filter, streaming_market_data_filter
import bz2
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import copy
import datatable as dt
from datatable import f
from datetime import datetime, date, timedelta, timezone
# from fake_useragent import UserAgent
import fastparquet as fp
from functools import partial
import gc
# import importlib
import inspect
import io
import itertools
import joblib
import mlflow
# import mlflow.pyfunc
# from numba import njit
import numpy as np
from os import listdir, makedirs
from os import path
import pandas as pd
from pathlib import Path
import pdb
# from playwright.sync_api import sync_playwright
# from playwright.async_api import async_playwright
import polars as pl
import pyarrow.parquet as pq
# from pycaret.classification import *
import queue
from rapidfuzz import fuzz, process
import random
import re
from re import sub
import requests
from scipy.special import logit
from scipy.stats import hmean
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
import shutil
# from sklearn.utils import resample
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, label_binarize, OrdinalEncoder
import string
# import sys
import threading
import time
import logging
import undetected_chromedriver as uc
logging.getLogger('undetected_chromedriver').setLevel(logging.ERROR)
import xgboost as xgb
from yaml import safe_load, dump
import warnings
import zstandard as zstd

np.warnings = warnings # to avoid ATPE error with hyperopt

# Load the config files

# Directory containing the YAML files
yaml_folder = './configs'

# Loop through all files in the folder
for filename in os.listdir(yaml_folder):
    if filename.endswith('.yaml') or filename.endswith('.yml'):  # Check for YAML file extensions
        file_path = os.path.join(yaml_folder, filename)
        
        # Create a valid variable name from the filename
        variable_name = os.path.splitext(filename)[0]
        variable_name = variable_name.replace('-', '_').replace(' ', '_')
        
        # Load the YAML file
        with open(file_path, 'r') as yaml_file:
            globals()[variable_name] = safe_load(yaml_file)
            

# Go through each function file and run it
for file in listdir('./functions/'):
    if file.endswith('.py'):
        exec(open('./functions/' + file).read())


# Settings so that data frames always display in full in terminal
pd.set_option('display.max_columns', None)

# Change configs for Raspberry Pi
if not os.name == "nt":
    config['firefox_location'] = config['firefox_location_pi']
    config['gecko_driver_location'] = config['gecko_driver_location_pi']
    config['chrome_driver_location'] = config['chrome_driver_location_pi']
    config['num_threads'] = 2