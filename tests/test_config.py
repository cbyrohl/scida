import pytest

from scida.config import get_config, get_config_fromfiles, get_simulationconfig


@pytest.mark.unit
def test_load_defaultconf():
    # We use some available resource to test against
    conf = get_config()
    assert isinstance(conf, dict)
    assert "cache_path" in conf


@pytest.mark.unit
def test_load_multiple_confs():
    conf = get_config_fromfiles(["config.yaml", "units/gadget_cosmological.yaml"])
    assert conf is not None
    assert "units" in conf
    assert "cache_path" in conf


@pytest.mark.unit
def test_usersimconf(mocker):
    usersimconf = {"data": {"NewSim": dict()}}
    mocker.patch("scida.config._get_simulationconfig_user", return_value=usersimconf)
    data = get_simulationconfig()["data"]
    assert "TNG50" in data  # already existing in package conf
    assert "NewSim" in data  # added by user

    usersimconf = {"data": {"TNG50": {"test": "test"}}}
    mocker.patch("scida.config._get_simulationconfig_user", return_value=usersimconf)
    data = get_simulationconfig()["data"]
    assert "TNG50" in data
    assert "test" in data["TNG50"]
    assert len(data["TNG50"]) == 1  # only the key we just added should be there
