import configparser

def get_config():
    import os
    prefix="ASTRODASK_"
    envconf = {k.replace(prefix, '').lower(): v for k, v in os.environ.items() if k.startswith(prefix)}
    cp = configparser.ConfigParser()
    cp.read(os.path.join(os.path.expanduser('~'), ".astrodask"))
    config = cp["DEFAULT"]
    config.update(**envconf)
    return config


_config = get_config()


