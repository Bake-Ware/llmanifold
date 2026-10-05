"""Per-flow limits for a flow's models, requests-per-minute limits, and requests per caller."""
from pathlib import Path

import pytest
import yaml

from llmanifold.config import ConfigError, parse

POOL2 = {"a": {"kind": "openai", "max_concurrency": 4}, "b": {"kind": "openai"}}


def chat(model="qwen"):
    return {"model": model, "messages": [{"role": "user", "content": "hi"}]}


def test_limits_must_name_a_member():
    base = {"endpoints": {"a": {"url": "http://x"}, "b": {"url": "http://y"}}}
    with pytest.raises(ConfigError, match="isn't in its pool"):
        parse({**base, "models": {"m": {"pool": ["a"], "limits": {"b": {"max_concurrency": 1}}}}})
    with pytest.raises(ConfigError, match="can only set"):
        parse({**base, "models": {"m": {"pool": ["a"], "limits": {"a": {"context": 1}}}}})
    cfg = parse({**base, "models": {"m": {"pool": ["a"], "limits": {"a": {"rate_limit": 30}}}}})
    assert cfg.models["m"].limit("a", "rate_limit") == 30 and cfg.models["m"].limit("a", "overflow_at") is None


async def test_flow_caps_its_share_of_a_model(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a"], "limits": {"a": {"max_concurrency": 1}}},
                            "other": {"pool": ["a"]}})
    r, qwen, other = s.core.router, s.core.cfg.models["qwen"], s.core.cfg.models["other"]
    a = r.states["a"]
    assert await r.try_endpoint("a", background=False, rid="1", flow=qwen) is a
    assert not r.has_room(a, flow=qwen)                 # qwen may hold one of a's four lanes
    assert r.has_room(a, flow=other)                    # other flows still get the rest
    assert a.flow_inflight == {"qwen": 1}
    await r.release(a, False, "1")
    assert r.has_room(a, flow=qwen) and a.flow_inflight == {"qwen": 0}


async def test_rate_limits_per_model_and_per_flow(stack):
    s = await stack({"a": {"kind": "openai", "max_concurrency": 8, "rate_limit": 3}},
                    {"qwen": {"pool": ["a"], "limits": {"a": {"rate_limit": 1}}}, "other": {"pool": ["a"]}})
    r, qwen, other = s.core.router, s.core.cfg.models["qwen"], s.core.cfg.models["other"]
    a = r.states["a"]
    await r.try_endpoint("a", background=False, rid="1", flow=qwen)
    await r.release(a, False, "1")
    assert not r.has_room(a, flow=qwen)                 # one a minute from qwen, even though it's idle
    for rid in ("2", "3"):
        assert await r.try_endpoint("a", background=False, rid=rid, flow=other) is a
        await r.release(a, False, rid)
    assert not r.has_room(a, flow=other)                # three a minute in total
    a.starts[0] = (a.starts[0][0] - 61, a.starts[0][1])  # a minute later the oldest start no longer counts
    assert r.has_room(a, flow=other) and r.has_room(a, flow=qwen)


async def test_rate_limited_pool_falls_back(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a"], "fallback": ["b"], "queue_timeout": 1,
                                     "limits": {"a": {"rate_limit": 1}}}})
    assert (await s.api.post("/v1/chat/completions", json=chat())).headers.get("X-LLManifold-Endpoint") == "a"
    r = await s.api.post("/v1/chat/completions", json=chat())
    assert r.status == 200 and r.headers.get("X-LLManifold-Endpoint") == "b"


async def test_edit_limits_from_the_admin_site(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a", "b"]}})
    r = await s.admin.put("/api/flows/qwen/members/a/limits", json={"max_concurrency": 2, "rate_limit": "30",
                                                                    "first_token_timeout": None})
    assert r.status == 200, await r.text()
    disk = yaml.safe_load(Path(s.core.cfg.path).read_text())
    assert disk["models"]["qwen"]["limits"] == {"a": {"max_concurrency": 2, "rate_limit": 30}}
    models = {m["name"]: m for m in (await (await s.admin.get("/api/status")).json())["models"]}
    assert models["qwen"]["limits"]["a"]["rate_limit"] == 30
    r = await s.admin.put("/api/flows/qwen/members/b/limits", json={"colour": 1})
    assert r.status == 400
    assert (await s.admin.delete("/api/flows/qwen/members/a")).status == 200    # limits leave with the model
    assert "limits" not in yaml.safe_load(Path(s.core.cfg.path).read_text())["models"]["qwen"]


async def test_requests_per_caller(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a"]}, "other": {"pool": ["b"]}})
    tok = (await (await s.admin.post("/api/tokens", json={"label": "voice"})).json())["token"]
    h = {"Authorization": f"Bearer {tok}"}
    for _ in range(3):
        await s.api.post("/v1/chat/completions", json=chat(), headers=h)
    await s.api.post("/v1/chat/completions", json=chat("other"), headers=h)
    await s.api.post("/v1/chat/completions", json=chat())
    callers = (await (await s.admin.get("/api/history?range=1h")).json())["callers"]
    by = {c["client"]: c for c in callers}
    assert by["voice"]["requests"] == 4 and by["voice"]["token"] and by["voice"]["flows"] == ["qwen", "other"]
    anon = [c for c in callers if c["client"] != "voice"]
    assert len(anon) == 1 and anon[0]["requests"] == 1 and not anon[0]["token"]
