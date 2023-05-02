from astrodask.config import get_config, get_config_fromfiles


def test_load_defaultconf():
    # We use some available resource to test against
    conf = get_config()
    assert isinstance(conf, dict)
    assert "cache_path" in conf


def test_load_multiple_confs():
    conf = get_config_fromfiles(["config.yaml", "units/illustris.yaml"])
    assert conf is not None
    assert "units" in conf
    assert "cache_path" in conf
