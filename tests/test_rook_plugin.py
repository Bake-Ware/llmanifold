"""The Rook plugin talks to a real admin listener."""
import asyncio

import pytest

from llmanifold_rook.plugin import PLUGIN

POOL2 = {"a": {"kind": "openai"}, "b": {"kind": "openai"}}


@pytest.fixture
async def plugin(stack, monkeypatch):
    s = await stack(POOL2, {"qwen": {"pool": ["a", "b"], "aliases": ["q"]}})
    monkeypatch.setenv("LLMANIFOLD_ADMIN", str(s.admin.make_url("")).rstrip("/"))
    return s, PLUGIN()


async def test_caps_are_declared(plugin):
    _, p = plugin
    names = {getattr(getattr(p, n), "_rook_cap_suffix", None) for n in dir(p)}
    assert {"status", "endpoints", "queue", "requests", "drain", "undrain", "reload"} <= names


async def test_status_and_drain(plugin):
    s, p = plugin
    assert await asyncio.to_thread(p.available)
    st = await p.status()
    assert [m["name"] for m in st["models"]] == ["qwen"]
    assert {e["name"]: e["state"] for e in st["endpoints"]} == {"a": "idle", "b": "idle"}
    assert "url" not in st["endpoints"][0]

    assert (await p.drain("a"))["draining"] is True
    assert {e["name"]: e["state"] for e in await p.endpoints()}["a"] == "draining"
    await p.undrain("a")
    assert {e["name"]: e["state"] for e in await p.endpoints()}["a"] == "idle"


async def test_requests_and_errors(plugin):
    s, p = plugin
    r = await s.api.post("/v1/chat/completions", json={"model": "q", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status == 200
    rows = await p.requests(limit=5)
    assert rows and rows[0]["model"] == "qwen"
    assert await p.requests(limit=5, errors_only=True) == []
    with pytest.raises(RuntimeError, match="unknown endpoint"):
        await p.drain("nope")


async def test_unreachable_admin_is_unavailable(monkeypatch):
    monkeypatch.setenv("LLMANIFOLD_ADMIN", "http://127.0.0.1:9")
    p = PLUGIN()
    assert await asyncio.to_thread(p.available) is False
    with pytest.raises(RuntimeError, match="can't reach"):
        await p.status()
