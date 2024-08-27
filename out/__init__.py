import os
from pathlib import Path

OUT_DIR = os.path.abspath(os.path.dirname(__file__))

os.makedirs(str(Path(OUT_DIR, 'checkpoints')), exist_ok=True)
os.makedirs(str(Path(OUT_DIR, 'logs')), exist_ok=True)
os.makedirs(str(Path(OUT_DIR, 'figures')), exist_ok=True)
