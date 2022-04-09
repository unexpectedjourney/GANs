from pprint import pprint

import yaml


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.full_load(file)
    pprint(f"Config: {config}")
    return config
