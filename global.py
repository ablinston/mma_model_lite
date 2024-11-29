from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
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
# from pycaret.classification import *
import re
from re import sub
import requests
from scipy.stats import hmean
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from sklearn.utils import resample
import string
import sys
from yaml import safe_load, dump

# Load the config files
config = safe_load(open('./configs/config.yaml'))
elo_config = safe_load(open('./configs/elo_params.yaml'))
final_model_features = safe_load(open('./configs/final_model_features.yaml'))

# Go through each function file and run it
for file in listdir('./functions/'):
    if file.endswith('.py'):
        exec(open('./functions/' + file).read())


# Settings so that data frames always display in full in terminal
pd.set_option('display.max_columns', None)

