"""Providers: one API account (URL plus key or ChatGPT sign-in) serving several models."""
import json

import pytest
import yaml
from pathlib import Path

from llmanifold.config import ConfigError, parse
from test_codex import FakeOpenAIAuth, chat


def _cfg(providers, endpoints):
    return parse({"providers": providers, "endpoints": endpoints,
                  "models": {"m": {"pool": [next(iter(endpoints))]}}})


def test_endpoint_takes_its_connection_from_the_provider():
    cfg = _cfg({"ds": {"url": "https://api.deepseek.com/v1", "key_env": "DS_KEY", "max_concurrency": 4}},
               {"flash": {"provider": "ds", "model": "deepseek-flash", "max_concurrency": 8},
                "pro": {"provider": "ds", "model": "deepseek-pro", "metered": False}})
    flash, pro = cfg.endpoints["flash"], cfg.endpoints["pro"]
    assert flash.url == pro.url == "https://api.deepseek.com" and flash.key_env == "DS_KEY"
    assert flash.metered and not pro.metered            # shared defaults can be overridden
    assert flash.probe == "none" and flash.account == pro.account == "ds"
    assert flash.is_deepseek and flash.emulates_json_schema


def test_connection_fields_belong_to_the_provider():
    with pytest.raises(ConfigError, match="come from provider"):
        _cfg({"ds": {"url": "https://x"}}, {"a": {"provider": "ds", "url": "https://y"}})
    with pytest.raises(ConfigError, match="unknown provider"):
        _cfg({}, {"a": {"provider": "nope"}})
    with pytest.raises(ConfigError, match="needs dialect: responses"):
        _cfg({"c": {"url": "https://x", "login": "chatgpt"}}, {"a": {"provider": "c"}})


def _codex(base):
    return {"codex": {"url": base + "/backend-api/codex", "dialect": "responses", "login": "chatgpt",
                      "max_concurrency": 1}}


async def test_one_chatgpt_sign_in_serves_every_model(stack, aiohttp_server):
    fake = FakeOpenAIAuth()
    base = str((await aiohttp_server(fake.app())).make_url("")).rstrip("/")
    s = await stack({"luna": {"kind": "raw", "provider": "codex", "model": "gpt-6-luna", "max_concurrency": 4},
                     "sol": {"kind": "raw", "provider": "codex", "model": "gpt-6-sol", "max_concurrency": 4}},
                    {"qwen": {"pool": ["luna"]}, "big": {"pool": ["sol"]}}, providers=_codex(base))
    s.core.chatgpt.issuer = base
    st = await (await s.admin.post("/api/providers/codex/chatgpt/import",
                                   json={"auth_json": json.dumps({"tokens": fake.tokens(1)})})).json()
    assert st["signed_in"]
    # the models see the provider's sign-in, whichever way they're asked
    assert (await (await s.admin.get("/api/endpoints/sol/chatgpt")).json())["signed_in"]
    assert (await s.api.post("/v1/chat/completions", json=chat())).status == 200
    assert (await s.api.post("/v1/chat/completions", json={**chat(), "model": "big"})).status == 200
    assert [b["model"] for b in fake.bodies] == ["gpt-6-luna", "gpt-6-sol"]
    before = len(fake.headers)
    await s.core._balance_once()                         # one quota check for the account, shown on both
    assert len(fake.headers) - before == 1
    eps = {e["name"]: e for e in (await (await s.admin.get("/api/status")).json())["endpoints"]}
    assert eps["luna"]["quota"] == eps["sol"]["quota"] and eps["luna"]["quota"]
    cfg = await (await s.admin.get("/api/config")).json()
    assert cfg["providers"]["codex"]["endpoints"] == ["luna", "sol"]
    assert cfg["providers"]["codex"]["chatgpt"]["signed_in"] and cfg["endpoints"]["sol"]["provider"] == "codex"
    models = await (await s.admin.get("/api/providers/codex/models")).json()
    assert models["ok"] and "gpt-5.5" in models["models"]


async def test_provider_limit_is_shared(stack, aiohttp_server):
    fake = FakeOpenAIAuth()
    base = str((await aiohttp_server(fake.app())).make_url("")).rstrip("/")
    s = await stack({"luna": {"kind": "raw", "provider": "codex", "model": "a", "max_concurrency": 4},
                     "sol": {"kind": "raw", "provider": "codex", "model": "b", "max_concurrency": 4}},
                    {"qwen": {"pool": ["luna"]}}, providers=_codex(base))
    s.core.chatgpt.import_auth_json("codex", json.dumps({"tokens": fake.tokens(1)}))
    s.core._signed_in("codex")
    r = s.core.router
    luna, sol = r.states["luna"], r.states["sol"]
    assert r.has_room(luna) and r.has_room(sol)
    held = await r.try_endpoint("luna", background=False, rid="x")
    assert held is luna
    assert not r.has_room(sol)                           # the provider allows one at a time across both
    assert await r.try_endpoint("sol", background=False, rid="y") is None
    await r.release(luna, False, "x")
    assert r.has_room(sol)


async def test_add_provider_and_models_from_the_admin_site(stack, tmp_path):
    s = await stack({"a": {"kind": "openai"}}, {"qwen": {"pool": ["a"]}})
    r = await s.admin.post("/api/providers", json={"name": "ds", "url": "https://api.deepseek.com",
                                                   "dialect": "openai", "max_concurrency": 6, "key": "sk-test"})
    assert r.status == 200, await r.text()
    for name, model in (("flash", "deepseek-flash"), ("pro", "deepseek-pro")):
        r = await s.admin.post("/api/endpoints", json={"name": name, "provider": "ds", "model": model,
                                                       "url": "ignored", "max_concurrency": 3})
        assert r.status == 200, await r.text()
    disk = yaml.safe_load(Path(s.core.cfg.path).read_text())
    assert list(disk) .index("providers") < list(disk).index("endpoints")
    assert "sk-test" not in Path(s.core.cfg.path).read_text()
    assert disk["endpoints"]["flash"] == {"provider": "ds", "model": "deepseek-flash", "max_concurrency": 3}
    assert s.core.cfg.endpoints["pro"].api_key() == "sk-test"
    cfg = await (await s.admin.get("/api/config")).json()
    assert cfg["providers"]["ds"]["key_set"] and cfg["providers"]["ds"]["endpoints"] == ["flash", "pro"]

    r = await s.admin.delete("/api/providers/ds")
    assert r.status == 400 and "flash, pro come from ds" in (await r.json())["error"]
    for name in ("flash", "pro"):
        assert (await s.admin.delete(f"/api/endpoints/{name}")).status == 200
    assert (await s.admin.delete("/api/providers/ds")).status == 200
    assert "providers" not in yaml.safe_load(Path(s.core.cfg.path).read_text())
