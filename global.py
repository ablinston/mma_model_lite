from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import copy
import csv
from datetime import datetime, date, timedelta
import datatable as dt
from datatable import f
import fastparquet as fp
import gc
import importlib
import inspect
import itertools
import joblib
import numpy as np
import os
from os import listdir, makedirs
from os import path
import pandas as pd
import pdb
import pyarrow.parquet as pq
# from pycaret.classification import *
import re
from re import sub
import requests
from scipy.stats import hmean
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from sklearn.utils import resample
from sklearn.metrics import log_loss
import string
import sys
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

