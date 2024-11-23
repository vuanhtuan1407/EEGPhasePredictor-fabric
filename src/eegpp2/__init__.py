import os

EEGPP_DIR = os.path.dirname(os.path.abspath(__file__))

CACHE_DIR = os.path.join(EEGPP_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
