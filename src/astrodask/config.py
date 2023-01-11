import os

import yaml

_conf = dict()


def get_config(reload=False):
    global _conf
    prefix = "ASTRODASK_"
    envconf = {
        k.replace(prefix, "").lower(): v
        for k, v in os.environ.items()
        if k.startswith(prefix)
    }
    path = envconf.pop("config_path", None)
    if path is None:
        path = os.path.join(os.path.expanduser("~"), ".astrodask.yaml")
    if not reload and len(_conf) > 0:
        return _conf
    config = {}
    if os.path.isfile(path):
        with open(path, "r") as file:
            config = yaml.safe_load(file)
    config.update(**envconf)
    _conf = config
    return config


_config = get_config()
