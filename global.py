from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import copy
# import csv
from datetime import datetime, date, timedelta
# from fake_useragent import UserAgent
import fastparquet as fp
from functools import partial
import gc
# from hyperopt.early_stop import no_progress_loss
# import importlib
import inspect
# import itertools
import joblib
# import matplotlib.pyplot as plt
import mlflow.pyfunc
import numpy as np
import os
from os import listdir, makedirs
from os import path
import pandas as pd
import pdb
# from playwright.sync_api import sync_playwright
# from playwright.async_api import async_playwright
import pyarrow.parquet as pq
# from pycaret.classification import *
import random
import re
from re import sub
import requests
# from scipy.stats import hmean
# import seaborn as sns
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
# from sklearn.utils import resample
# from sklearn.metrics import log_loss
import string
# import sys
# import time
import xgboost as xgb
from yaml import safe_load, dump

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
    config['num_threads'] = 2