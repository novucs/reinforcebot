import os
from pathlib import Path

FRAME_DISPLAY_SIZE = (256, 256)
FRAME_SIZE = (80, 80)
OBSERVATION_SPACE = (2, *FRAME_SIZE)
ENSEMBLE_SIZE = 5
STEP_SECONDS = 0.1
SEGMENT_SIZE = int(2 / STEP_SECONDS)
UPDATE_TARGET_PARAMETERS_STEPS = 32
BASE_URL = 'https://reinforcebot.novucs.net/'
API_URL = BASE_URL + 'api/'
BASE_PATH = os.path.join(str(Path.home()), 'ReinforceBot')
SESSION_FILE = os.path.join(BASE_PATH, 'session.json')
AGENTS_PATH = os.path.join(BASE_PATH, 'agents')
CACHE_PATH = os.path.join(BASE_PATH, 'cache')

os.makedirs(BASE_PATH, exist_ok=True)
os.makedirs(AGENTS_PATH, exist_ok=True)
os.makedirs(CACHE_PATH, exist_ok=True)
