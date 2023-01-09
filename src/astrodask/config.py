import os

import yaml


def get_config():
    prefix = "ASTRODASK_"
    envconf = {
        k.replace(prefix, "").lower(): v
        for k, v in os.environ.items()
        if k.startswith(prefix)
    }
    path = envconf.pop("config_path", None)
    if path is None:
        path = os.path.join(os.path.expanduser("~"), ".astrodask.yaml")
    config = {}
    if os.path.isfile(path):
        with open(path, "r") as file:
            config = yaml.safe_load(file)
    config.update(**envconf)
    return config


_config = get_config()
