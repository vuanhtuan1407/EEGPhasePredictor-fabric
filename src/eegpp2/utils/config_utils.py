from pathlib import Path

import yaml

from ..configs import CONFIG_DIR


def load_yaml_config(yaml_file):
    yml_config_path = str(Path(CONFIG_DIR, yaml_file))
    with open(yml_config_path, 'r') as f:
        return yaml.safe_load(f)
