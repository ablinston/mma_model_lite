# This file is just to change the working directory of the pytest environment back up to the root directory

import os
# Absolute or relative to the current file
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

