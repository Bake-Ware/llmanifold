import asyncio
import json

POOL2 = {"a": {"kind": "openai", "context": 262144}, "b": {"kind": "openai", "context": 262144}}


def chat(text="hi", stream=False, model="qwen", system="sys", **extra):
    return {"model": model, "stream": stream,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}], **extra}


async def read_sse(resp):
    raw = (await resp.read()).decode()
    return [block for block in raw.split("\n\n") if block.strip()]


# ---------------------------------------------------------------- pools, race, affinity

async def test_simultaneous_identical_requests_use_both_lanes(stack):
    s = await stack(
        {"a": {"kind": "openai", "fake": {"delay": 0.3}}, "b": {"kind": "openai", "fake": {"delay": 0.3}}},
        {"qwen": {"pool": ["a", "b"]}})
    for _ in range(5):
        before = (len(s.fakes["a"].bodies), len(s.fakes["b"].bodies))
        r1, r2 = await asyncio.gather(s.api.post("/v1/chat/completions", json=chat()),
                                      s.api.post("/v1/chat/completions", json=chat()))
        assert r1.status == r2.status == 200
        got = (len(s.fakes["a"].bodies) - before[0], len(s.fakes["b"].bodies) - before[1])
        assert got == (1, 1), f"collision: {got}"
    assert s.fakes["a"].max_active == 1 and s.fakes["b"].max_active == 1


async def test_affinity_reuses_the_lane_for_a_conversation(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a", "b"]}})
    first = await s.api.post("/v1/chat/completions", json=chat(system="conversation-1"))
    lane = first.headers["X-LLManifold-Endpoint"]
    for _ in range(4):
        r = await s.api.post("/v1/chat/completions", json=chat(system="conversation-1"))
        assert r.headers["X-LLManifold-Endpoint"] == lane


async def test_model_alias_and_upstream_model_name(stack):
    s = await stack({"a": {"kind": "openai", "model": "engine-name"}},
                    {"qwen": {"pool": ["a"], "aliases": ["legacy-name"]}})
    r = await s.api.post("/v1/chat/completions", json=chat(model="legacy-name"))
    assert r.status == 200
    assert (await r.json())["model"] == "legacy-name"            # the client sees the name it asked for
    assert s.fakes["a"].bodies[-1]["model"] == "engine-name"  # the engine gets its own name
    models = await (await s.api.get("/v1/models")).json()
    ids = {m["id"]: m for m in models["data"]}
    assert ids["legacy-name"]["alias_of"] == "qwen" and "meta" not in ids["qwen"]   # no context known: none advertised


async def test_unknown_model_uses_default_when_configured(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a", "b"], "input_modalities": ["text", "image"]}})
    assert (await s.api.post("/v1/chat/completions", json=chat(model="mystery"))).status == 404
    s = await stack(POOL2, {"qwen": {"pool": ["a", "b"]}}, default_model="qwen")
    for name in ("mystery", ""):
        r = await s.api.post("/v1/chat/completions", json=chat(model=name))
        assert r.status == 200
    models = await (await s.api.get("/v1/models")).json()
    assert "architecture" not in models["data"][0]


async def test_input_modalities_advertised(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a", "b"], "input_modalities": ["text", "image"]}})
    m = (await (await s.api.get("/v1/models")).json())["data"][0]
    assert m["architecture"]["input_modalities"] == ["text", "image"]


# ---------------------------------------------------------------- priorities

async def test_background_limited_to_one_lane_and_interactive_jumps_queue(stack):
    s = await stack(
        {"a": {"kind": "openai", "fake": {"delay": 0.4}}, "b": {"kind": "openai", "fake": {"delay": 0.4}}},
        {"qwen": {"pool": ["a", "b"], "background_max_lanes": 1}})
    bg = {"X-LLManifold-Priority": "background"}
    order = []

    async def go(name, headers, text):
        r = await s.api.post("/v1/chat/completions", json=chat(text=text, system=name), headers=headers)
        order.append(name)
        return r.status

    t_bg1 = asyncio.create_task(go("bg1", bg, "1"))
    await asyncio.sleep(0.05)
    t_bg2 = asyncio.create_task(go("bg2", bg, "2"))      # must wait: background already holds its one lane
    await asyncio.sleep(0.05)
    t_int = asyncio.create_task(go("int", {}, "3"))       # takes the free lane immediately
    await asyncio.gather(t_bg1, t_bg2, t_int)
    assert order.index("int") < order.index("bg2")
    total_active = s.fakes["a"].max_active + s.fakes["b"].max_active
    assert total_active == 2


async def test_interactive_waiter_served_before_background_waiter(stack):
    s = await stack({"a": {"kind": "openai", "fake": {"delay": 0.3}}}, {"qwen": {"pool": ["a"]}})
    served = []

    async def go(name, headers):
        await s.api.post("/v1/chat/completions", json=chat(system=name), headers=headers)
        served.append(name)

    first = asyncio.create_task(go("first", {}))
    await asyncio.sleep(0.05)
    bg = asyncio.create_task(go("bg", {"X-LLManifold-Priority": "background"}))
    await asyncio.sleep(0.05)
    inter = asyncio.create_task(go("inter", {}))
    await asyncio.gather(first, bg, inter)
    assert served == ["first", "inter", "bg"]


# ---------------------------------------------------------------- fallback rules

async def test_fallback_on_5xx(stack):
    s = await stack({"a": {"kind": "openai", "fake": {"mode": "500"}}, "remote": {"kind": "openai", "fallback": True}},
                    {"qwen": {"pool": ["a"], "fallback": ["remote"]}})
    r = await s.api.post("/v1/chat/completions", json=chat())
    assert r.status == 200 and r.headers["X-LLManifold-Endpoint"] == "remote"
    row = s.core.recent[0]
    assert row["fallback"] == 1 and [a.get("trigger") for a in row["attempts"]] == ["5xx", None]


async def test_fallback_on_dead_endpoint(stack):
    s = await stack({"a": {"kind": "dead"}, "remote": {"kind": "openai", "fallback": True}},
                    {"qwen": {"pool": ["a"], "fallback": ["remote"]}})
    r = await s.api.post("/v1/chat/completions", json=chat())
    assert r.status == 200 and r.headers["X-LLManifold-Endpoint"] == "remote"
    assert s.core.router.states["a"].healthy is False


async def test_dead_pool_member_retried_on_the_other_lane(stack):
    s = await stack({"a": {"kind": "dead"}, "b": {"kind": "openai"}}, {"qwen": {"pool": ["a", "b"]}})
    statuses = [(await s.api.post("/v1/chat/completions", json=chat(system=str(i)))).status for i in range(4)]
    assert statuses == [200] * 4


async def test_fallback_on_empty_stream_before_any_bytes(stack):
    s = await stack({"a": {"kind": "openai", "fake": {"mode": "empty"}}, "remote": {"kind": "openai", "fallback": True}},
                    {"qwen": {"pool": ["a"], "fallback": ["remote"]}})
    r = await s.api.post("/v1/chat/completions", json=chat(stream=True))
    blocks = await read_sse(r)
    text = "".join(json.loads(b[6:])["choices"][0]["delta"].get("content", "")
                   for b in blocks if b.startswith("data: {") and json.loads(b[6:]).get("choices"))
    assert text == "hello from fake" and r.headers["X-LLManifold-Endpoint"] == "remote"


async def test_fallback_on_slow_first_token(stack):
    s = await stack({"a": {"kind": "openai", "fake": {"delay": 0.0, "chunk_delay": 2.0}},
                     "remote": {"kind": "openai", "fallback": True}},
                    {"qwen": {"pool": ["a"], "fallback": ["remote"], "first_token_timeout": 0.3,
                              "fallback_on": ["slow", "5xx"]}})
    r = await s.api.post("/v1/chat/completions", json=chat(stream=True))
    assert r.headers["X-LLManifold-Endpoint"] == "remote"
    assert "hello" in (await r.read()).decode()


async def test_client_errors_are_not_retried(stack):
    s = await stack({"a": {"kind": "openai", "fake": {"mode": "badreq"}}, "remote": {"kind": "openai", "fallback": True}},
                    {"qwen": {"pool": ["a"], "fallback": ["remote"]}})
    r = await s.api.post("/v1/chat/completions", json=chat())
    assert r.status == 400
    assert "temperature" in (await r.json())["error"]["message"]
    assert not s.fakes["remote"].bodies


async def test_context_error_falls_back(stack):
    s = await stack({"a": {"kind": "openai", "fake": {"mode": "context"}}, "big": {"kind": "openai", "fallback": True}},
                    {"qwen": {"pool": ["a"], "fallback": ["big"]}})
    r = await s.api.post("/v1/chat/completions", json=chat())
    assert r.status == 200 and r.headers["X-LLManifold-Endpoint"] == "big"


async def test_mid_stream_failure_is_not_swapped(stack):
    s = await stack({"a": {"kind": "openai", "fake": {"mode": "midfail", "text": "one two three four"}},
                     "remote": {"kind": "openai", "fallback": True}},
                    {"qwen": {"pool": ["a"], "fallback": ["remote"]}})
    r = await s.api.post("/v1/chat/completions", json=chat(stream=True))
    body = (await r.read()).decode()
    assert "one" in body and "llmanifold" in body          # partial content, then an error event
    assert not s.fakes["remote"].bodies


async def test_queue_timeout_falls_back(stack):
    s = await stack({"a": {"kind": "openai", "fake": {"delay": 1.0}}, "remote": {"kind": "openai", "fallback": True}},
                    {"qwen": {"pool": ["a"], "fallback": ["remote"], "queue_timeout": 0.2}})
    busy = asyncio.create_task(s.api.post("/v1/chat/completions", json=chat(system="x")))
    await asyncio.sleep(0.1)
    r = await s.api.post("/v1/chat/completions", json=chat(system="y"))
    assert r.headers["X-LLManifold-Endpoint"] == "remote"
    await busy


# ---------------------------------------------------------------- metered endpoints, webhook, auth

async def test_metered_fallback_needs_a_token(stack):
    s = await stack({"a": {"kind": "openai", "fake": {"mode": "500"}},
                     "deepseek": {"kind": "openai", "fallback": True, "metered": True}},
                    {"qwen": {"pool": ["a"], "fallback": ["deepseek"]}}, webhook=True)
    r = await s.api.post("/v1/chat/completions", json=chat())
    assert r.status == 502
    assert not s.fakes["deepseek"].bodies
    tok = s.core.store.create_token("home-agent", ["*"], False, "test")
    r = await s.api.post("/v1/chat/completions", json=chat(), headers={"Authorization": f"Bearer {tok['token']}"})
    assert r.status == 200 and r.headers["X-LLManifold-Endpoint"] == "deepseek"
    for _ in range(50):
        if s.hooks:
            break
        await asyncio.sleep(0.02)
    assert s.hooks and s.hooks[0]["event"] == "metered" and s.hooks[0]["endpoint"] == "deepseek"
    assert s.hooks[0]["client"] == "home-agent" and "metered endpoint" in s.hooks[0]["text"]


async def test_metered_fallback_allowed_for_open_model_when_configured(stack):
    s = await stack({"a": {"kind": "openai", "fake": {"mode": "500"}},
                     "deepseek": {"kind": "openai", "fallback": True, "metered": True}},
                    {"qwen": {"pool": ["a"], "fallback": ["deepseek"], "allow_metered_unauthenticated": True}})
    r = await s.api.post("/v1/chat/completions", json=chat())
    assert r.status == 200 and r.headers["X-LLManifold-Endpoint"] == "deepseek"
    st = await (await s.admin.get("/api/status")).json()
    assert st["models"][0]["warning"]


async def test_per_model_tokens(stack):
    s = await stack({"a": {"kind": "openai"}, "b": {"kind": "openai"}},
                    {"private": {"pool": ["a"], "auth": "token"}, "public": {"pool": ["b"]}})
    assert (await s.api.post("/v1/chat/completions", json=chat(model="public"))).status == 200
    assert (await s.api.post("/v1/chat/completions", json=chat(model="private"))).status == 401
    tok = s.core.store.create_token("x", ["public"], False, "test")
    h = {"Authorization": f"Bearer {tok['token']}"}
    assert (await s.api.post("/v1/chat/completions", json=chat(model="private"), headers=h)).status == 401
    tok2 = s.core.store.create_token("y", ["private"], False, "test")
    r = await s.api.post("/v1/chat/completions", json=chat(model="private"),
                         headers={"x-api-key": tok2["token"]})
    assert r.status == 200
    s.core.store.set_auth("private", "open", "test")       # a human opened it in the admin UI
    assert (await s.api.post("/v1/chat/completions", json=chat(model="private"))).status == 200


async def test_background_token(stack):
    s = await stack({"a": {"kind": "openai"}}, {"qwen": {"pool": ["a"]}})
    tok = s.core.store.create_token("cron", ["*"], True, "test")
    await s.api.post("/v1/chat/completions", json=chat(), headers={"Authorization": f"Bearer {tok['token']}"})
    assert s.core.recent[0]["priority"] == "background" and s.core.recent[0]["client"] == "cron"


# ---------------------------------------------------------------- dialect translation end to end

async def test_anthropic_client_on_openai_engine_streaming_with_tools(stack):
    s = await stack({"a": {"kind": "openai", "fake": {"tool": True}}}, {"qwen": {"pool": ["a"]}})
    body = {"model": "qwen", "max_tokens": 64, "stream": True, "system": "be brief",
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}]}
    r = await s.api.post("/v1/messages", json=body, headers={"anthropic-version": "2023-06-01"})
    assert r.status == 200
    events = [b.split("\n")[0][7:] for b in await read_sse(r)]
    assert events[0] == "message_start" and events[-1] == "message_stop"
    assert "content_block_delta" in events
    up = s.fakes["a"].bodies[-1]
    assert up["messages"][0] == {"role": "system", "content": "be brief"}
    assert up["tools"][0]["function"]["name"] == "get_weather"
    assert s.core.recent[0]["tokens_out"] > 0


async def test_openai_client_on_anthropic_endpoint(stack):
    s = await stack({"claude": {"kind": "anthropic", "key": "sk-test", "fake": {"tool": True}}},
                    {"smart": {"pool": ["claude"]}})
    r = await s.api.post("/v1/chat/completions", json=chat(model="smart", max_tokens=32))
    j = await r.json()
    assert j["object"] == "chat.completion" and j["choices"][0]["finish_reason"] == "tool_calls"
    assert j["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    h = s.fakes["claude"].headers[-1]
    assert h["x-api-key"] == "sk-test" and h["anthropic-version"]
    up = s.fakes["claude"].bodies[-1]
    assert up["system"] == "sys" and up["max_tokens"] == 32
    r = await s.api.post("/v1/chat/completions", json=chat(model="smart", stream=True))
    blocks = await read_sse(r)
    assert blocks[-1] == "data: [DONE]"


async def test_count_tokens_and_compat_endpoints(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a", "b"]}})
    r = await s.api.post("/v1/messages/count_tokens", json={"messages": [{"role": "user", "content": "x" * 350}]})
    assert (await r.json())["input_tokens"] > 50
    slots = await (await s.api.get("/slots")).json()
    assert len(slots) == 2 and "url" not in json.dumps(slots)
    props = await (await s.api.get("/props")).json()
    assert props["n_ctx"] == 262144
    r = await s.api.post("/v1/completions", json={"model": "qwen", "prompt": "x"})
    assert (await r.json())["choices"][0]["text"] == "legacy"


# ---------------------------------------------------------------- admin: drain, access, tokens, reload

async def test_drain_moves_traffic(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a", "b"]}})
    r = await s.admin.post("/api/endpoints/a/drain")
    assert (await r.json())["draining"] is True
    for i in range(4):
        r = await s.api.post("/v1/chat/completions", json=chat(system=str(i)))
        assert r.headers["X-LLManifold-Endpoint"] == "b"
    await s.admin.post("/api/endpoints/a/undrain")
    assert s.core.router.states["a"].draining is False


async def test_admin_refuses_unlisted_networks(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a", "b"]}}, admin={"allow_from": ["10.99.0.0/16"]})
    assert (await s.admin.get("/api/status")).status == 403


async def test_token_admin_needs_a_human(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a", "b"]}})
    r = await s.admin.post("/api/tokens", json={"label": "x"})
    assert r.status == 403                      # a local agent can drain, not mint tokens
    assert (await s.admin.post("/api/endpoints/a/drain")).status == 200


async def test_human_through_trusted_proxy(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a", "b"], "auth": "token"}},
                    admin={"trusted_proxies": ["127.0.0.1/32"], "allowed_emails": ["me@example.com"]})
    assert (await s.admin.get("/api/status")).status == 403          # proxy without an email: refused
    h = {"Cf-Access-Authenticated-User-Email": "Me@Example.com"}
    r = await s.admin.post("/api/tokens", json={"label": "home-agent", "models": "qwen"}, headers=h)
    tok = (await r.json())["token"]
    assert tok.startswith("llm_")
    r = await s.api.post("/v1/chat/completions", json=chat(), headers={"Authorization": f"Bearer {tok}"})
    assert r.status == 200
    lst = await (await s.admin.get("/api/tokens", headers=h)).json()
    assert lst[0]["label"] == "home-agent" and "token" not in lst[0] and lst[0]["uses"] >= 0
    r = await s.admin.post("/api/models/qwen/auth", json={"mode": "open"}, headers=h)
    assert (await r.json())["auth"] == "open"
    bad = {"Cf-Access-Authenticated-User-Email": "intruder@example.com"}
    assert (await s.admin.get("/api/status", headers=bad)).status == 403


async def test_reload_and_metrics(stack, tmp_path):
    s = await stack(POOL2, {"qwen": {"pool": ["a", "b"]}})
    await s.api.post("/v1/chat/completions", json=chat())
    m = await (await s.admin.get("/metrics")).text()
    assert "llmanifold_requests_total 1" in m and 'llmanifold_endpoint_inflight{endpoint="a"}' in m
    st = await (await s.admin.get("/api/status")).json()
    assert {e["name"] for e in st["endpoints"]} == {"a", "b"}
    reqs = await (await s.admin.get("/api/requests")).json()
    assert reqs[0]["model"] == "qwen"


async def test_keepalive_while_queued(stack):
    s = await stack({"a": {"kind": "openai", "fake": {"delay": 0.6}}},
                    {"qwen": {"pool": ["a"]}}, keepalive_after=0.1)
    busy = asyncio.create_task(s.api.post("/v1/chat/completions", json=chat(system="x", stream=True)))
    await asyncio.sleep(0.05)
    r = await s.api.post("/v1/chat/completions", json=chat(system="y", stream=True))
    body = (await r.read()).decode()
    assert body.startswith(": keepalive") and "hello" in body
    await (await busy).read()


async def test_stream_token_counts_without_client_usage(stack):
    s = await stack({"a": {"kind": "openai", "fake": {"text": "one two three"}}}, {"qwen": {"pool": ["a"]}})
    r = await s.api.post("/v1/chat/completions", json=chat(stream=True))
    blocks = await read_sse(r)
    assert not any('"usage"' in b for b in blocks)                    # the client didn't ask: stripped
    assert s.fakes["a"].bodies[-1]["stream_options"] == {"include_usage": True}
    assert s.core.recent[0]["tokens_out"] == 3 and s.core.recent[0]["tokens_in"] == 7
    r = await s.api.post("/v1/chat/completions", json=chat(stream=True, stream_options={"include_usage": True}))
    assert any('"usage"' in b for b in await read_sse(r))              # the client asked: kept


async def test_release_clears_a_stale_engine_busy_flag(stack):
    s = await stack({"a": {"kind": "openai"}}, {"qwen": {"pool": ["a"]}})
    st = s.core.router.states["a"]
    await s.core.router.acquire(s.core.cfg.models["qwen"], key=None, background=False, timeout=1)
    st.probe_busy, st.probe_at = 1, __import__("time").monotonic()   # probe taken while our request ran
    await s.core.router.release(st, False)
    assert st.load() == 0


async def test_failed_request_records_a_readable_trail(stack):
    s = await stack({"a": {"kind": "openai", "fake": {"mode": "500"}},
                     "deepseek": {"kind": "openai", "fallback": True, "metered": True}},
                    {"qwen": {"pool": ["a"], "fallback": ["deepseek"]}})
    await s.api.post("/v1/chat/completions", json=chat())
    err = s.core.recent[0]["error"]
    assert err.startswith("boom") and "deepseek skipped: metered endpoint needs a token" in err


async def test_reasoning_only_answer_is_not_empty(stack):
    """A thinking model that hits max_tokens mid-thought returns content=null; that's an answer."""
    s = await stack({"a": {"kind": "openai", "fake": {"mode": "think"}}, "fb": {"kind": "openai"}},
                    {"qwen": {"pool": ["a"], "fallback": ["fb"]}})
    r = await s.api.post("/v1/chat/completions", json=chat())
    assert r.status == 200
    assert (await r.json())["choices"][0]["message"]["reasoning_content"]
    r = await s.api.post("/v1/chat/completions", json=chat(stream=True))
    assert r.status == 200 and "reasoning_content" in "".join(await read_sse(r))
    assert not s.fakes["fb"].bodies                      # no fallback happened
    r = await s.api.post("/v1/messages", json={"model": "qwen", "max_tokens": 5,
                                                 "messages": [{"role": "user", "content": "hi"}]})
    assert (await r.json())["content"][0]["type"] == "thinking"


async def test_overflow_fallback_clears_a_backlog(stack):
    """overflow_at=2: one request may wait for the pool; the 2nd and later in the queue go to the fallback."""
    s = await stack({"a": {"kind": "openai", "fake": {"delay": 0.4}},
                     "f": {"kind": "openai", "fallback": True, "overflow_at": 2, "max_concurrency": 4}},
                    {"qwen": {"pool": ["a"], "fallback": ["f"]}})
    rs = await asyncio.gather(*[s.api.post("/v1/chat/completions", json=chat(text=f"q{i}")) for i in range(4)])
    assert all(r.status == 200 for r in rs)
    assert len(s.fakes["a"].bodies) == 2 and len(s.fakes["f"].bodies) == 2
    await asyncio.sleep(0.1)
    over = [r for r in s.core.recent if any(a.get("trigger") == "overflow" for a in r["attempts"])]
    assert len(over) == 2 and all(r["fallback"] for r in over)


async def test_overflow_respects_metered_rule(stack):
    s = await stack({"a": {"kind": "openai", "fake": {"delay": 0.3}},
                     "f": {"kind": "openai", "fallback": True, "metered": True, "overflow_at": 1, "max_concurrency": 4}},
                    {"qwen": {"pool": ["a"], "fallback": ["f"]}})
    rs = await asyncio.gather(*[s.api.post("/v1/chat/completions", json=chat(text=f"q{i}")) for i in range(3)])
    assert all(r.status == 200 for r in rs)
    assert not s.fakes["f"].bodies                      # anonymous callers wait for the pool instead


async def test_stalled_api_is_abandoned_and_fallen_back_from(stack):
    """A provider that accepts requests but only sends keep-alives (seen with DeepSeek under load)
    is given up on after its first_token_timeout, for streaming and non-streaming clients alike."""
    s = await stack({"a": {"kind": "openai", "fake": {"mode": "stall"}, "first_token_timeout": 0.5},
                     "b": {"kind": "openai", "fallback": True}},
                    {"qwen": {"pool": ["a"], "fallback": ["b"]}})
    import time
    for stream in (False, True):
        t = time.monotonic()
        r = await s.api.post("/v1/chat/completions", json=chat(stream=stream))
        assert r.status == 200
        body = await r.text()
        assert "hello" in body
        assert time.monotonic() - t < 5
    assert len(s.fakes["b"].bodies) == 2
    assert s.fakes["a"].bodies[-1]["stream"] is True          # upstream is always streamed


async def test_non_streaming_client_still_gets_plain_json(stack):
    s = await stack(POOL2, {"qwen": {"pool": ["a", "b"]}})
    r = await s.api.post("/v1/chat/completions", json=chat())
    j = await r.json()
    assert j["object"] == "chat.completion" and j["choices"][0]["message"]["content"] == "hello from fake"
    assert j["usage"]["completion_tokens"] == 3
    r = await s.api.post("/v1/messages", json={"model": "qwen", "max_tokens": 9, "messages": [{"role": "user", "content": "hi"}]})
    j = await r.json()
    assert j["type"] == "message" and j["content"][0]["text"] == "hello from fake"
