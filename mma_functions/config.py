from pathlib import Path
import os
from yaml import safe_load

# Define where the configs are
CONFIG_DIR = Path(__file__).parent.parent / 'configs'

# Function to load the configs
def load_configs(configs_path = CONFIG_DIR):
    """
    Loads the configuration from YAML files and returns the configuration dictionaries.
    """

    # Initiate a dictioary for the config files
    configs = {}

    # Get a list of the config files in the directory
    config_yamls = os.listdir(configs_path)

    # Add each to the dictionary
    for config_file in config_yamls:
        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
            config_name = os.path.splitext(config_file)[0]
            with open(os.path.join(configs_path, config_file), 'r') as f:
                configs[config_name] = safe_load(f)

    return configs

# Load the configs into global space
globals().update(load_configs())