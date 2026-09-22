# This script will run at the beginning of the test process
# Change the working directory because it will be run within the tests/functions/folder
from pathlib import Path
import os
os.chdir(Path(__file__).parent.parent.parent) # Change the working directory to the root of the project from this files

def pytest_sessionstart(session):
    pass

