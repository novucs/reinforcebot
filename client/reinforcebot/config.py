import json
import os
import sys
from pathlib import Path

BASE_PATH = os.path.join(str(Path.home()), 'ReinforceBot')
SESSION_FILE = os.path.join(BASE_PATH, 'session.json')
AGENTS_PATH = os.path.join(BASE_PATH, 'agents')
CACHE_PATH = os.path.join(BASE_PATH, 'cache')
CONFIG_PATH = os.path.join(BASE_PATH, 'config.json')

os.makedirs(BASE_PATH, exist_ok=True)
os.makedirs(AGENTS_PATH, exist_ok=True)
os.makedirs(CACHE_PATH, exist_ok=True)

if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'w') as config_file:
        base_url = 'https://reinforcebot.novucs.net/'
        json.dump({
            'BASE_URL': base_url,
            'API_URL': base_url + 'api/',
            'FRAME_DISPLAY_SIZE': (256, 256),
            'STEP_SECONDS': 0.1,
            'SEGMENT_SIZE': int(2 / 0.1),
            'EXPERIENCE_BUFFER_SIZE': int(2.5e5),
            'EXPERIENCE_BATCH_SIZE': 128,
            'REWARD_BUFFER_SIZE': 16,
            'REWARD_BATCH_SIZE': 8,
            'UPDATE_TARGET_PARAMETERS_STEPS': 32,
        }, config_file, indent=2)

with open(CONFIG_PATH, 'r') as config_file:
    user_config = json.load(config_file)
    BASE_URL = user_config['BASE_URL']
    API_URL = user_config['API_URL']
    FRAME_DISPLAY_SIZE = user_config['FRAME_DISPLAY_SIZE']
    STEP_SECONDS = user_config['STEP_SECONDS']
    SEGMENT_SIZE = user_config['SEGMENT_SIZE']
    EXPERIENCE_BUFFER_SIZE = user_config['EXPERIENCE_BUFFER_SIZE']
    EXPERIENCE_BATCH_SIZE = user_config['EXPERIENCE_BATCH_SIZE']
    REWARD_BUFFER_SIZE = user_config['REWARD_BUFFER_SIZE']
    REWARD_BATCH_SIZE = user_config['REWARD_BATCH_SIZE']
    UPDATE_TARGET_PARAMETERS_STEPS = user_config['UPDATE_TARGET_PARAMETERS_STEPS']

# Not to be modified.
FRAME_SIZE = (80, 80)
OBSERVATION_SPACE = (2, *FRAME_SIZE)
ENSEMBLE_SIZE = 5

CONFIG = {k: v for k, v in sys.modules[__name__].__dict__.items() if k.isupper()}
