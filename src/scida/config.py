import importlib.resources
import os
import pathlib
from typing import Dict, List, Optional

import yaml

_conf = dict()


def _access_confdir() -> str:
    """
    Get the path to the configuration directory.
    Create it and copy the default configuration if it does not exist yet.
    Returns
    -------
    str
        The path to the configuration directory.
    """
    path_user = os.path.expanduser("~")
    path_confdir = os.path.join(path_user, ".config/scida")
    path_conf = os.path.join(path_confdir, "config.yaml")
    if not os.path.exists(path_conf):
        copy_defaultconfig(overwrite=False)
    return path_confdir


def get_config(reload: bool = False, update_global=True) -> dict:
    """
    Load the configuration from the default path.
    Parameters
    ----------
    reload: bool
        Reload the configuration, even if it has already been loaded.
    update_global: bool
        Update the global configuration dictionary.

    Returns
    -------
    dict
        The configuration dictionary.

    """
    global _conf
    prefix = "SCIDA_"
    envconf = {
        k.replace(prefix, "").lower(): v
        for k, v in os.environ.items()
        if k.startswith(prefix)
    }

    # in any case, we make sure that there is some config in the default path.
    path_confdir = _access_confdir()
    path_conf = os.path.join(path_confdir, "config.yaml")

    # next, we load the config from the default path, unless explicitly overridden.
    path = envconf.pop("config_path", None)
    if path is None:
        path = path_conf
    if not reload and len(_conf) > 0:
        return _conf
    config = get_config_fromfile(path)
    if config.get("copied_default", False):
        print(
            "Warning! Using default configuration. Please adjust/replace in '%s'."
            % path
        )

    config.update(**envconf)
    if update_global:
        _conf = config
    return config


def combine_configs(configs: List[Dict], mode="overwrite_keys") -> Dict:
    # this mode will merge dictionaries recursively and replace node entries in a with b
    if mode == "overwrite_values":

        def mergefunc_values(a, b):
            if b is None:
                return a
            return b  # just overwrite by latter entry

        mergefunc_keys = None

    elif mode == "overwrite_keys":

        def mergefunc_keys(a, b):
            if b is None:
                return a
            return b

        mergefunc_values = None

    else:
        raise ValueError("Unknown mode '%s'" % mode)
    conf = configs[0]
    for c in configs[1:]:
        merge_dicts_recursively(
            conf, c, mergefunc_keys=mergefunc_keys, mergefunc_values=mergefunc_values
        )
    return conf


def _get_simulationconfig_user():
    conf_sims_user = None
    path_confdir = _access_confdir()
    p = pathlib.Path(path_confdir) / "simulations.yaml"
    if p.exists():
        p = str(p)
        conf_sims_user = get_config_fromfile(p)
    return conf_sims_user


def get_simulationconfig():
    conf_user = get_config()
    conf_sims = get_config_fromfile("simulations.yaml")
    conf_sims_user = _get_simulationconfig_user()

    confs_base = [conf_sims, conf_user]
    if conf_sims_user is not None:
        confs_base.append(conf_sims_user)
    # we only want to overwrite keys within "data", otherwise no merging of simkeys would take place
    confs = []
    for c in confs_base:
        if "data" in c:
            confs.append(c["data"])

    conf_sims = combine_configs(confs, mode="overwrite_keys")
    conf_sims = {"data": conf_sims}

    return conf_sims


def copy_defaultconfig(overwrite=False) -> None:
    """
    Copy the configuration example to the user's home directory.
    Parameters
    ----------
    overwrite: bool
        Overwrite existing configuration file.

    Returns
    -------

    """

    path_user = os.path.expanduser("~")
    path_confdir = os.path.join(path_user, ".config/scida")
    if not os.path.exists(path_confdir):
        os.makedirs(path_confdir, exist_ok=True)
    path_conf = os.path.join(path_confdir, "config.yaml")
    if os.path.exists(path_conf) and not overwrite:
        raise ValueError("Configuration file already exists at '%s'" % path_conf)
    with importlib.resources.path("scida.configfiles", "config.yaml") as fp:
        with open(fp, "r") as file:
            content = file.read()
            with open(path_conf, "w") as newfile:
                newfile.write(content)


def get_config_fromfile(resource: str) -> Dict:
    """
    Load config from a YAML file.
    Parameters
    ----------
    resource
        The name of the resource or file path.

    Returns
    -------

    """
    if resource == "":
        raise ValueError("Config name cannot be empty.")
    # order (in descending order of priority):
    # 1. absolute path?
    path = os.path.expanduser(resource)
    if os.path.isabs(path):
        with open(path, "r") as file:
            conf = yaml.safe_load(file)
        return conf
    bpath = os.path.expanduser("~/.config/scida")
    path = os.path.join(bpath, resource)
    # 2. non-absolute path?
    # 2.1. check ~/.config/scida/units/
    if os.path.isfile(path):
        with open(path, "r") as file:
            conf = yaml.safe_load(file)
        return conf
    # 2.2 check scida package resource units/
    resource_path = "scida.configfiles"
    resource_elements = resource.split("/")
    rname = resource_elements[-1]
    if len(resource_elements) > 1:
        resource_path += "." + ".".join(resource_elements[:-1])
    with importlib.resources.path(resource_path, rname) as fp:
        with open(fp, "r") as file:
            conf = yaml.safe_load(file)
    return conf


def merge_dicts_recursively(
    dict_a: Dict,
    dict_b: Dict,
    path: Optional[List] = None,
    mergefunc_keys: Optional[callable] = None,
    mergefunc_values: Optional[callable] = None,
) -> Dict:
    """
    Merge two dictionaries recursively.
    Parameters
    ----------
    dict_a
        The first dictionary.
    dict_b
        The second dictionary.
    path
        The path to the current node.
    mergefunc_keys: callable
        The function to use for merging. If None, we recursively enter the dictionary.
    mergefunc_values: callable
        The function to use for merging. If None, collisions will raise an exception.

    Returns
    -------
    dict
    """
    if path is None:
        path = []
    for key in dict_b:
        if key in dict_a:
            if mergefunc_keys is not None:
                dict_a[key] = mergefunc_keys(dict_a[key], dict_b[key])
            elif isinstance(dict_a[key], dict) and isinstance(dict_b[key], dict):
                merge_dicts_recursively(
                    dict_a[key],
                    dict_b[key],
                    path + [str(key)],
                    mergefunc_keys=mergefunc_keys,
                    mergefunc_values=mergefunc_values,
                )
            elif dict_a[key] == dict_b[key]:
                pass  # same leaf value
            else:
                if mergefunc_values is not None:
                    dict_a[key] = mergefunc_values(dict_a[key], dict_b[key])
                else:
                    raise Exception("Conflict at %s" % ".".join(path + [str(key)]))
        else:
            dict_a[key] = dict_b[key]
    return dict_a


def get_config_fromfiles(paths: List[str], subconf_keys: Optional[List[str]] = None):
    """
    Load and merge multiple YAML config files
    Parameters
    ----------
    paths
        Paths to the config files.
    subconf_keys
        The keys to the correct sub configuration within each config.

    Returns
    -------

    """
    confs = []
    for path in paths:
        confs.append(get_config_fromfile(path))
    conf = {}
    for confdict in confs:
        conf = merge_dicts_recursively(conf, confdict)
    return conf


_config = get_config()
