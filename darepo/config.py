import os
import configparser

config = configparser.ConfigParser()
config.read(os.path.join(os.path.expanduser('~'), ".darepo"))
_config = config["DEFAULT"]
