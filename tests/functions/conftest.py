# This script will run at the beginning of the test process
# Change the working directory because it will be run within the tests/functions/folder
import os
os.chdir('../..')

def pytest_sessionstart(session):
    pass

