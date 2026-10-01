from pathlib import Path

import pytest

from llmanifold.config import ConfigError, load, parse

ROOT = Path(__file__).resolve().parent.parent


def test_example_config_loads():
    cfg = load(ROOT / "config.example.yaml")
    m = cfg.resolve("default")
    assert m.name == "local-large"
    assert "429" in m.fallback_on          # bare 429 in YAML is an int; must still count
    assert m.context == 262144


def _base(**model):
    return {"endpoints": {"a": {"url": "http://x/v1"}, "f": {"url": "http://y", "fallback": True}},
            "models": {"m": {"pool": ["a"], **model}}}


def test_url_v1_is_stripped():
    assert parse(_base()).endpoints["a"].url == "http://x"


@pytest.mark.parametrize("model, msg", [
    ({"pool": ["f"]}, "fallback-only"),
    ({"pool": ["nope"]}, "unknown endpoint"),
    ({"fallback_on": ["sometimes"]}, "unknown fallback trigger"),
    ({"colour": "red"}, "unknown keys"),
    ({"auth": "maybe"}, "auth must be"),
])
def test_bad_models_are_rejected(model, msg):
    with pytest.raises(ConfigError, match=msg):
        parse(_base(**model))


def test_duplicate_alias_rejected():
    raw = _base(aliases=["x"])
    raw["models"]["n"] = {"pool": ["a"], "aliases": ["x"]}
    with pytest.raises(ConfigError, match="alias"):
        parse(raw)
