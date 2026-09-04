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
import subprocess
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

# Auto-detect the installed Chrome/Chromium version so undetected_chromedriver
# can be told exactly which chromedriver build to fetch, instead of relying on
# its own detection (which mismatches when running Chrome Beta, since it
# updates ahead of published stable drivers)
def _detect_chrome_version(binary_path):
    try:
        if os.name == "nt":
            # On Windows, running chrome.exe with --version can attach to (or
            # launch) a full browser window instead of printing and exiting, so
            # read the version straight off the file instead of executing it
            ps_command = f"(Get-Item -LiteralPath '{binary_path}').VersionInfo.ProductVersion"
            version_output = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", ps_command],
                stderr=subprocess.STDOUT, timeout=15
            ).decode('utf-8', errors='ignore')
        else:
            # On Linux/Chromium (e.g. the Raspberry Pi), --version is safe: it
            # prints to stdout and exits rather than opening a window
            version_output = subprocess.check_output(
                [binary_path, '--version'], stderr=subprocess.STDOUT, timeout=15
            ).decode('utf-8', errors='ignore')

        version_match = re.search(r'(\d+)\.\d+\.\d+\.\d+', version_output)
        if version_match:
            return int(version_match.group(1))
    except Exception as e:
        logging.warning(f"Could not auto-detect Chrome version from {binary_path}: {e}")
    return None

if os.name == "nt":
    config['chrome_version_main'] = _detect_chrome_version(config['chrome_binary_location'])
    uc_cache_path = os.path.join(os.path.expanduser('~'), 'appdata', 'roaming', 'undetected_chromedriver')
else:
    config['chrome_version_main'] = _detect_chrome_version(config.get('chrome_binary_location_pi', '/usr/bin/chromium-browser'))
    uc_cache_path = os.path.join(os.path.expanduser('~'), '.local', 'share', 'undetected_chromedriver')

# Clear any cached chromedriver binary so a stale driver left over from a
# previous Chrome/Chromium install or update is never reused
if os.path.isdir(uc_cache_path):
    shutil.rmtree(uc_cache_path, ignore_errors=True)