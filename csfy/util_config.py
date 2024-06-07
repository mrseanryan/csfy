import os
from cornsnake import util_toml

from . import config

def populate_config():
    """
    Populates config.py from config.ini
    """
    util_toml.read_config_ini_file("./config.ini", config)

def path_to_checkpoint():
    return path_to_generated_model()

def path_to_generated_model():
    return os.path.join(config.OUTPUT_DIR, 'trained.model')

def path_to_results():
    return os.path.join(config.OUTPUT_DIR)
