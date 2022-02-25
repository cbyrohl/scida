import os
import configparser

config = configparser.ConfigParser()
config.read(os.path.join(os.path.expanduser('~'), ".astrodask"))
_config = config["DEFAULT"]
