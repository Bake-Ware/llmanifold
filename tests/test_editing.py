"""Editing models (endpoints) and alias flows from the admin site."""
import os
from pathlib import Path

import yaml

POOL2 = {"a": {"kind": "openai"}, "b": {"kind": "openai"}}
HUMAN = {"admin": {"local_humans": True}}


def chat(model="qwen"):
    return {"model": model, "messages": [{"role": "user", "content": "hi"}]}


def on_disk(s) -> dict:
    return yaml.safe_load(Path(s.core.cfg.path).read_text())


async def test_edits_need_a_person(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a"]}})
    r = await s.admin.post("/api/flows/qwen/members", json={"endpoint": "b"})
    assert r.status == 403
    assert (await s.admin.post("/api/endpoints/a/pause")).status == 200      # pausing is fine for anyone


async def test_add_endpoint_with_key_then_route_to_it(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a"]}}, **HUMAN)
    url = s.core.cfg.endpoints["b"].url
    r = await s.admin.post("/api/endpoints", json={
        "name": "cloud", "url": url + "/v1", "model": "remote-name", "key": "sk-secret",
        "metered": True, "fallback": True, "max_concurrency": 4, "probe": "none"})
    assert r.status == 200, await r.text()
    raw = on_disk(s)
    ep = raw["endpoints"]["cloud"]
    assert ep["url"] == url + "/v1" and ep["metered"] is True and "key" not in ep
    key_file = Path(ep["key_file"])
    assert key_file.read_text().strip() == "sk-secret"
    assert oct(os.stat(key_file).st_mode & 0o777) == "0o600"
    assert "sk-secret" not in Path(s.core.cfg.path).read_text()
    assert Path(s.core.cfg.path).read_text().startswith("# test config")   # comments survive

    r = await s.admin.post("/api/flows/qwen/members", json={"endpoint": "cloud", "role": "fallback"})
    assert r.status == 200
    assert s.core.cfg.models["qwen"].fallback == ["cloud"]                   # applied live
    s.fakes["a"].mode = "500"
    r = await s.api.post("/v1/chat/completions", json=chat())
    assert r.status == 502                                    # metered fallback refused for anonymous callers
    r = await s.admin.put("/api/flows/qwen", json={"allow_metered_unauthenticated": True})
    assert r.status == 200
    r = await s.api.post("/v1/chat/completions", json=chat())
    assert r.status == 200
    assert s.fakes["b"].bodies[-1]["model"] == "remote-name"
    assert s.fakes["b"].headers[-1]["Authorization"] == "Bearer sk-secret"

    cfg = await (await s.admin.get("/api/config")).json()
    assert cfg["endpoints"]["cloud"]["key"] == "file" and cfg["endpoints"]["cloud"]["flows"] == ["qwen"]
    assert "sk-secret" not in str(cfg)


async def test_fallback_only_endpoint_cant_join_a_pool(stack):
    s = await stack({**POOL2, "c": {"kind": "openai", "fallback": True}}, {"qwen": {"pool": ["a"]}}, **HUMAN)
    r = await s.admin.post("/api/flows/qwen/members", json={"endpoint": "c", "role": "pool"})
    assert r.status == 400 and "fallback-only" in (await r.json())["error"]
    assert on_disk(s)["models"]["qwen"]["pool"] == ["a"]                     # nothing written


async def test_new_flow_remove_member_and_delete(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a"]}}, **HUMAN)
    r = await s.admin.post("/api/flows", json={"name": "fast", "aliases": "quick, speedy", "pool": ["a", "b"]})
    assert r.status == 200, await r.text()
    assert (await s.api.post("/v1/chat/completions", json=chat("speedy"))).status == 200
    assert (await s.admin.delete("/api/flows/fast/members/a")).status == 200
    assert s.core.cfg.models["fast"].pool == ["b"]
    r = await s.admin.delete("/api/flows/fast/members/b")
    assert r.status == 400 and "last model" in (await r.json())["error"]
    assert (await s.admin.delete("/api/flows/fast")).status == 200
    assert (await s.api.post("/v1/chat/completions", json=chat("speedy"))).status == 404


async def test_new_flow_needs_a_model(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a"]}}, **HUMAN)
    r = await s.admin.post("/api/flows", json={"name": "empty"})
    assert r.status == 400 and "needs a pool or a fallback" in (await r.json())["error"]


async def test_remove_endpoint_in_use(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a", "b"]}, "solo": {"pool": ["b"]}}, **HUMAN)
    r = await s.admin.delete("/api/endpoints/a")
    assert r.status == 400 and "qwen" in (await r.json())["error"]
    assert (await s.admin.delete("/api/endpoints/a?force=1")).status == 200
    assert s.core.cfg.models["qwen"].pool == ["b"] and "a" not in s.core.cfg.endpoints
    r = await s.admin.delete("/api/endpoints/b?force=1")
    assert r.status == 400 and "nothing to route to" in (await r.json())["error"]


async def test_default_flow_cant_be_deleted(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a"]}, "other": {"pool": ["b"]}}, default_model="qwen", **HUMAN)
    r = await s.admin.delete("/api/flows/qwen")
    assert r.status == 400 and "default" in (await r.json())["error"]


async def test_bad_names_rejected(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a"]}}, **HUMAN)
    r = await s.admin.post("/api/endpoints", json={"name": "has space", "url": "http://x"})
    assert r.status == 400


async def test_edit_endpoint_keeps_key_unless_replaced(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a"]}}, **HUMAN)
    await s.admin.put("/api/endpoints/b", json={"key": "k1"})
    await s.admin.put("/api/endpoints/b", json={"max_concurrency": 3})
    assert s.core.cfg.endpoints["b"].api_key() == "k1" and s.core.cfg.endpoints["b"].max_concurrency == 3
    await s.admin.put("/api/endpoints/b", json={"clear_key": True})
    assert s.core.cfg.endpoints["b"].api_key() is None


async def test_pause_survives_restart_and_reload(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a", "b"]}})
    assert (await s.admin.post("/api/endpoints/a/pause")).status == 200
    for _ in range(4):
        assert (await s.api.post("/v1/chat/completions", json=chat())).status == 200
    assert not s.fakes["a"].bodies
    assert s.core.store.paused().keys() == {"a"}
    from llmanifold.config import load
    s.core.apply_config(load(s.core.cfg.path))                               # config reload keeps it paused
    assert s.core.router.states["a"].draining
    from llmanifold.router import Router
    assert Router(s.core.cfg, set(s.core.store.paused())).states["a"].draining   # and a restart
    await s.admin.post("/api/endpoints/a/resume")
    assert not s.core.store.paused()


async def test_test_endpoint_lists_models(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a"]}}, **HUMAN)
    r = await s.admin.post("/api/test-endpoint", json={"url": s.core.cfg.endpoints["a"].url})
    j = await r.json()
    assert j["ok"] and j["models"] == ["fake"]
    j = await (await s.admin.post("/api/test-endpoint", json={"url": "http://127.0.0.1:9"})).json()
    assert not j["ok"] and j["error"]


async def test_history_buckets(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a", "b"]}})
    for _ in range(3):
        await s.api.post("/v1/chat/completions", json=chat())
    s.fakes["a"].mode = s.fakes["b"].mode = "500"
    await s.api.post("/v1/chat/completions", json=chat())
    import asyncio
    await asyncio.sleep(0.2)                                                # history writes are async
    h = await (await s.admin.get("/api/history?range=1h")).json()
    assert h["totals"]["requests"] == 4 and h["totals"]["errors"] == 1
    assert sum(sum(b["requests"].values()) for b in h["buckets"]) == 3
    assert h["endpoints"][:2] == ["a", "b"] and len(h["problems"]) == 1


async def test_page_and_assets_are_cache_safe(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a"]}})
    r = await s.admin.get("/")
    html = await r.text()
    assert r.headers["Cache-Control"] == "no-store"
    import re
    js = re.search(r'/static/app\.js\?v=([0-9a-f]{12})', html)
    assert js and re.search(r'/static/style\.css\?v=' + js.group(1), html)
    r = await s.admin.get(f"/static/app.js?v={js.group(1)}")
    assert r.status == 200 and "immutable" in r.headers["Cache-Control"]
    assert f'/static/art.js?v={js.group(1)}' in html
    assert (await s.admin.get(f"/static/art.js?v={js.group(1)}")).status == 200
    assert (await s.admin.get("/api/status")).headers["Cache-Control"] == "no-store"


async def test_edits_keep_section_gaps_and_comments(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a"]}}, **HUMAN)
    path = Path(s.core.cfg.path)
    text = path.read_text().replace("models:", "\nmodels:", 1).replace("pool:", "pool: # lanes\n   ", 0)
    path.write_text(text)
    await s.admin.put("/api/endpoints/b", json={"overflow_at": 3})
    await s.admin.post("/api/endpoints", json={"name": "c", "url": "http://127.0.0.1:9"})
    out = path.read_text()
    assert "\n\nmodels:" in out                        # the blank line before the next section survives
    assert "overflow_at: 3\n" in out
