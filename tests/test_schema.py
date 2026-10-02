"""Structured output on endpoints that only have JSON mode (DeepSeek)."""
import json

from llmanifold import schema as S
from llmanifold.config import parse

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["action", "count"],
          "properties": {"action": {"type": "string", "enum": ["mine", "build"]},
                         "count": {"type": "integer", "minimum": 1},
                         "notes": {"type": "array", "items": {"$ref": "#/$defs/note"}}},
          "$defs": {"note": {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}}}}
GOOD = '{"action": "mine", "count": 3}'


def body(stream=False, **extra):
    return {"model": "qwen", "stream": stream, "messages": [{"role": "system", "content": "sys"},
                                                           {"role": "user", "content": "what next?"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "step", "schema": SCHEMA}}, **extra}


def test_validator():
    assert S.check(GOOD, SCHEMA) == []
    assert S.check('{"action": "mine", "count": 3, "notes": [{"text": "x"}]}', SCHEMA) == []
    assert "not valid JSON" in S.check("sure! here you go", SCHEMA)[0]
    assert S.check("", SCHEMA) == ["the response was empty"]
    errs = S.check('{"action": "dance", "count": 0, "extra": 1, "notes": [{}]}', SCHEMA)
    assert any("$.action: must be one of" in e for e in errs) and any("$.count: must be >= 1" in e for e in errs)
    assert any("unexpected property 'extra'" in e for e in errs)
    assert any("$.notes[0]: missing required property 'text'" in e for e in errs)
    assert S.check('{"action": "mine"}', SCHEMA) == ["$: missing required property 'count'"]
    assert S.check('{"action": "mine", "count": true}', SCHEMA) == ["$.count: expected integer, got boolean"]
    assert S.validate("a", {"anyOf": [{"type": "string"}, {"type": "null"}]}) == []
    assert S.validate(1, {"anyOf": [{"type": "string"}, {"type": "null"}]})


def test_deepseek_urls_emulate_by_default():
    eps = parse({"endpoints": {"ds": {"url": "https://api.deepseek.com"}, "local": {"url": "http://127.0.0.1:1"},
                               "forced": {"url": "http://127.0.0.1:2", "json_schema": "emulate"},
                               "off": {"url": "https://api.deepseek.com/beta", "json_schema": "native"}}}).endpoints
    assert [eps[n].emulates_json_schema for n in ("ds", "local", "forced", "off")] == [True, False, True, False]


async def test_native_endpoints_get_the_schema_untouched(stack):
    s = await stack({"a": {"kind": "openai", "fake": {"text": GOOD}}}, {"qwen": {"pool": ["a"]}})
    assert (await s.api.post("/v1/chat/completions", json=body())).status == 200
    assert s.fakes["a"].bodies[-1]["response_format"]["type"] == "json_schema"


async def test_bad_reply_is_sent_back_with_the_format(stack):
    s = await stack({"a": {"kind": "openai", "json_schema": "emulate",
                           "fake": {"texts": ['{"action": "dance", "count": 3}', GOOD]}}}, {"qwen": {"pool": ["a"]}})
    r = await s.api.post("/v1/chat/completions", json=body())
    out = await r.json()
    assert r.status == 200 and json.loads(out["choices"][0]["message"]["content"]) == {"action": "mine", "count": 3}
    first, second = s.fakes["a"].bodies
    assert first["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in first["messages"]] == ["system", "system", "user"]
    assert "json" in first["messages"][1]["content"].lower() and '"enum": ["mine", "build"]' in first["messages"][1]["content"]
    assert [m["role"] for m in second["messages"][-2:]] == ["assistant", "user"]
    assert "$.action: must be one of" in second["messages"][-1]["content"]
    assert "Please respond in the format" in second["messages"][-1]["content"]
    assert out["usage"]["completion_tokens"] == 8          # both rounds are counted


async def test_streaming_client_gets_only_the_reply_that_fits(stack):
    s = await stack({"a": {"kind": "openai", "json_schema": "emulate", "fake": {"texts": ["not json at all", GOOD]}}},
                    {"qwen": {"pool": ["a"]}})
    r = await s.api.post("/v1/chat/completions", json=body(stream=True))
    raw = (await r.read()).decode()
    assert r.status == 200 and "not json" not in raw and raw.rstrip().endswith("data: [DONE]")
    text = "".join(json.loads(b[6:])["choices"][0]["delta"].get("content") or ""
                   for b in raw.split("\n\n") if b.startswith("data: {") and json.loads(b[6:])["choices"])
    assert json.loads(text) == {"action": "mine", "count": 3}


async def test_never_fitting_reply_falls_back(stack):
    s = await stack({"ds": {"kind": "openai", "json_schema": "emulate", "fake": {"text": '{"action": "dance"}'}},
                     "next": {"kind": "openai", "fallback": True, "fake": {"text": GOOD}}},
                    {"qwen": {"pool": ["ds"], "fallback": ["next"]}})
    r = await s.api.post("/v1/chat/completions", json=body())
    assert r.status == 200 and r.headers["X-LLManifold-Endpoint"] == "next"
    assert len(s.fakes["ds"].bodies) == S.MAX_ROUNDS
    assert s.fakes["next"].bodies[-1]["response_format"]["type"] == "json_schema"
    assert s.core.recent[0]["attempts"][0]["trigger"] == "schema"


async def test_tool_calls_are_not_checked(stack):
    s = await stack({"a": {"kind": "openai", "json_schema": "emulate", "fake": {"text": "", "tool": True}}},
                    {"qwen": {"pool": ["a"]}})
    r = await s.api.post("/v1/chat/completions", json=body())
    out = await r.json()
    assert r.status == 200 and out["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert len(s.fakes["a"].bodies) == 1


async def test_reply_cut_off_by_max_tokens_is_not_asked_again(stack):
    s = await stack({"ds": {"kind": "openai", "json_schema": "emulate", "fake": {"mode": "think"}},
                     "next": {"kind": "openai", "fallback": True, "fake": {"text": GOOD}}},
                    {"qwen": {"pool": ["ds"], "fallback": ["next"]}})
    r = await s.api.post("/v1/chat/completions", json=body())
    assert r.status == 200 and r.headers["X-LLManifold-Endpoint"] == "next"
    assert len(s.fakes["ds"].bodies) == 1


async def test_deepseek_balance_in_status(stack, aiohttp_server, monkeypatch):
    from aiohttp import web
    seen = []

    async def bal(req):
        seen.append(req.headers.get("Authorization"))
        return web.json_response({"is_available": True, "balance_infos": [
            {"currency": "USD", "total_balance": "12.34", "granted_balance": "0.00", "topped_up_balance": "12.34"}]})
    app = web.Application()
    app.router.add_get("/user/balance", bal)
    srv = await aiohttp_server(app)
    from llmanifold.config import Endpoint
    monkeypatch.setattr(Endpoint, "is_deepseek", property(lambda self: self.name == "ds"))
    s = await stack({"ds": {"kind": "dead", "key": "sk-test"}, "a": {"kind": "openai"}}, {"qwen": {"pool": ["a"]}})
    st = s.core.router.states["ds"]
    st.cfg.url = str(srv.make_url("")).rstrip("/")
    await s.core.balance(st)
    await s.core.balance(s.core.router.states["a"])
    eps = {e["name"]: e for e in (await (await s.admin.get("/api/status")).json())["endpoints"]}
    assert eps["ds"]["balance"] == [{"amount": "12.34", "currency": "USD"}] and eps["a"]["balance"] is None
    assert seen == ["Bearer sk-test"]
