"""OpenAI Responses API upstreams, and ChatGPT (Codex) sign-in."""
import asyncio
import base64
import json
import time

from aiohttp import web

import llmanifold.dialects as D
from llmanifold.chatgpt import ChatGPTLogins


def _jwt(claims: dict) -> str:
    b = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")
    return f"{b({'alg': 'none'})}.{b(claims)}.sig"


def _sse(obj):
    return f"event: {obj['type']}\ndata: {json.dumps(obj)}\n\n".encode()


class FakeOpenAIAuth:
    """auth.openai.com device flow + token endpoint, and the Codex backend's /responses."""

    def __init__(self):
        self.polls = 0
        self.refreshes = 0
        self.bodies = []
        self.headers = []
        self.mode = "text"       # text | tool | fail | unauthorized_once
        self.access_exp = time.time() + 3600

    def tokens(self, n):
        return {"access_token": _jwt({"exp": self.access_exp, "n": n}), "refresh_token": f"rt{n}",
                "id_token": _jwt({"email": "bake@example.com", "https://api.openai.com/auth": {
                    "chatgpt_account_id": "acct-1", "chatgpt_plan_type": "pro"}})}

    def app(self):
        async def usercode(req):
            assert (await req.json())["client_id"]
            return web.json_response({"device_auth_id": "dev1", "user_code": "ABCD-1234", "interval": "0"})

        async def devtoken(req):
            self.polls += 1
            if self.polls < 2:
                return web.json_response({"error": "pending"}, status=403)
            return web.json_response({"authorization_code": "code1", "code_verifier": "ver1", "code_challenge": "ch"})

        async def token(req):
            if req.content_type == "application/x-www-form-urlencoded":
                f = await req.post()
                assert f["grant_type"] == "authorization_code" and f["code_verifier"] == "ver1"
                assert f["redirect_uri"].endswith("/deviceauth/callback")
                return web.json_response(self.tokens(1))
            j = await req.json()
            assert j["grant_type"] == "refresh_token"
            self.refreshes += 1
            self.access_exp = time.time() + 3600
            return web.json_response(self.tokens(1 + self.refreshes))

        async def responses(req):
            body = await req.json()
            self.bodies.append(body)
            self.headers.append(dict(req.headers))
            if self.mode == "unauthorized_once":
                self.mode = "text"
                return web.json_response({"detail": "expired"}, status=401)
            resp = web.StreamResponse()
            resp.content_type = "text/event-stream"
            await resp.prepare(req)
            await resp.write(_sse({"type": "response.created", "response": {"id": "r1"}}))
            if self.mode == "fail":
                await resp.write(_sse({"type": "response.failed", "response": {"error": {"message": "usage limit reached"}}}))
                return resp
            await resp.write(_sse({"type": "response.reasoning_summary_text.delta", "delta": "thinking"}))
            if self.mode == "tool":
                await resp.write(_sse({"type": "response.output_item.added", "item": {
                    "type": "function_call", "id": "fc1", "call_id": "call_9", "name": "get_weather", "arguments": ""}}))
                await resp.write(_sse({"type": "response.function_call_arguments.delta", "item_id": "fc1", "delta": '{"city":'}))
                await resp.write(_sse({"type": "response.function_call_arguments.delta", "item_id": "fc1", "delta": ' "Paris"}'}))
            else:
                for w in ("hello", " from", " codex"):
                    await resp.write(_sse({"type": "response.output_text.delta", "item_id": "m1", "delta": w}))
            await resp.write(_sse({"type": "response.completed", "response": {
                "usage": {"input_tokens": 11, "output_tokens": 3}}}))
            return resp

        async def models(req):
            return web.json_response({"models": [{"slug": "gpt-5.5"}, {"slug": "gpt-5.6-terra"}]})

        app = web.Application()
        app.router.add_post("/api/accounts/deviceauth/usercode", usercode)
        app.router.add_post("/api/accounts/deviceauth/token", devtoken)
        app.router.add_post("/oauth/token", token)
        app.router.add_post("/backend-api/codex/responses", responses)
        app.router.add_get("/backend-api/codex/models", models)

        async def usage(req):
            self.headers.append(dict(req.headers))
            return web.json_response({"plan_type": "pro", "rate_limit": {"allowed": True, "primary_window": {
                "used_percent": 17, "limit_window_seconds": 604800, "reset_after_seconds": 90000,
                "reset_at": 1791056930}, "secondary_window": None}})
        app.router.add_get("/backend-api/wham/usage", usage)
        return app


def chat(stream=False, **kw):
    return {"model": "qwen", "stream": stream, "messages": [
        {"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}], **kw}


async def _setup(stack, aiohttp_server):
    fake = FakeOpenAIAuth()
    srv = await aiohttp_server(fake.app())
    base = str(srv.make_url("")).rstrip("/")
    s = await stack({"a": {"kind": "dead"},
                     "codex": {"kind": "dead", "url": base + "/backend-api/codex", "dialect": "responses",
                               "login": "chatgpt", "model": "gpt-5.5", "fallback": True}},
                    {"qwen": {"pool": ["a"], "fallback": ["codex"]}}, admin={"local_humans": True})
    s.core.chatgpt.issuer = base
    return s, fake, base


async def test_device_sign_in_then_requests(stack, aiohttp_server):
    s, fake, base = await _setup(stack, aiohttp_server)
    r = await s.api.post("/v1/chat/completions", json=chat())
    assert r.status in (502, 503)                         # not signed in yet: the flow can't use it
    await asyncio.sleep(2.3)                              # the probe loop marks it down until someone signs in
    assert not s.core.router.states["codex"].healthy
    assert "not signed in" in s.core.router.states["codex"].last_error
    j = await (await s.admin.post("/api/endpoints/codex/chatgpt/start")).json()
    assert j["user_code"] == "ABCD-1234" and j["url"].endswith("/codex/device")
    for _ in range(50):
        st = await (await s.admin.get("/api/endpoints/codex/chatgpt")).json()
        if st["signed_in"]:
            break
        await asyncio.sleep(0.05)
    assert st["signed_in"] and st["email"] == "bake@example.com" and st["plan"] == "pro"
    assert "token" not in json.dumps(st)                  # never shown back
    await s.core.probe(s.core.router.states["codex"])     # sign-in is its health
    assert s.core.router.states["codex"].healthy
    await s.core.balance(s.core.router.states["codex"])   # the plan's allowance shows on the dashboard
    eps = {e["name"]: e for e in (await (await s.admin.get("/api/status")).json())["endpoints"]}
    assert eps["codex"]["quota"] == [{"used_percent": 17, "window_seconds": 604800, "reset_at": 1791056930}]
    assert fake.headers[-1]["Authorization"].startswith("Bearer ") and eps["a"]["quota"] is None

    r = await s.api.post("/v1/chat/completions", json=chat())
    out = await r.json()
    assert r.status == 200, out
    msg = out["choices"][0]["message"]
    assert msg["content"] == "hello from codex" and msg["reasoning_content"] == "thinking"
    assert out["usage"]["prompt_tokens"] == 11
    h = fake.headers[-1]
    assert h["Authorization"].startswith("Bearer ") and h["ChatGPT-Account-ID"] == "acct-1"
    b = fake.bodies[-1]
    assert b["model"] == "gpt-5.5" and b["instructions"] == "be brief" and b["store"] is False and b["stream"] is True
    assert b["input"] == [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}]


async def test_streaming_anthropic_and_tools(stack, aiohttp_server):
    s, fake, base = await _setup(stack, aiohttp_server)
    s.core.chatgpt.import_auth_json("codex", json.dumps({"tokens": fake.tokens(1)}))
    r = await s.api.post("/v1/chat/completions", json=chat(stream=True))
    body = (await r.read()).decode()
    assert '"content":"hello"' in body.replace(" ", "") and "[DONE]" in body

    fake.mode = "tool"
    r = await s.api.post("/v1/chat/completions", json=chat(tools=[{"type": "function", "function": {
        "name": "get_weather", "parameters": {"type": "object"}}}]))
    msg = (await r.json())["choices"][0]
    assert msg["finish_reason"] == "tool_calls"
    assert msg["message"]["tool_calls"][0]["function"] == {"name": "get_weather", "arguments": '{"city": "Paris"}'}
    assert fake.bodies[-1]["tools"][0]["name"] == "get_weather"

    fake.mode = "text"
    r = await s.api.post("/v1/messages", json={"model": "qwen", "max_tokens": 50, "stream": True,
                                               "messages": [{"role": "user", "content": "hi"}]})
    ev = (await r.read()).decode()
    assert "thinking_delta" in ev and "text_delta" in ev and "message_stop" in ev


async def test_expired_access_token_refreshes(stack, aiohttp_server):
    s, fake, base = await _setup(stack, aiohttp_server)
    fake.access_exp = time.time() + 10                     # inside the refresh margin
    s.core.chatgpt.import_auth_json("codex", json.dumps(fake.tokens(1)))
    assert (await s.api.post("/v1/chat/completions", json=chat())).status == 200
    assert fake.refreshes == 1
    fake.mode = "unauthorized_once"                        # server-side revocation: refresh and retry once
    assert (await s.api.post("/v1/chat/completions", json=chat())).status == 200
    assert fake.refreshes == 2


async def test_failed_response_is_a_failure(stack, aiohttp_server):
    s, fake, base = await _setup(stack, aiohttp_server)
    s.core.chatgpt.import_auth_json("codex", json.dumps(fake.tokens(1)))
    fake.mode = "fail"
    r = await s.api.post("/v1/chat/completions", json=chat())
    assert r.status == 502 and "usage limit" in (await r.text())


async def test_models_listing_via_sign_in(stack, aiohttp_server):
    s, fake, base = await _setup(stack, aiohttp_server)
    s.core.chatgpt.import_auth_json("codex", json.dumps(fake.tokens(1)))
    j = await (await s.admin.post("/api/test-endpoint", json={
        "url": base + "/backend-api/codex", "dialect": "responses", "login": "chatgpt", "name": "codex"})).json()
    assert j["ok"] and j["models"] == ["gpt-5.5", "gpt-5.6-terra"]


def test_request_translation_round_trip():
    body = {"model": "x", "messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "text", "text": "look"},
                                     {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "42"}],
        "reasoning_effort": "high", "tool_choice": {"type": "function", "function": {"name": "f"}}}
    r = D.translate_request(body, "openai", "responses")
    assert r["instructions"] == "sys" and r["reasoning"] == {"summary": "auto", "effort": "high"}
    assert [i["type"] for i in r["input"]] == ["message", "function_call", "function_call_output"]
    assert r["input"][0]["content"][1] == {"type": "input_image", "image_url": "data:image/png;base64,AA"}
    assert r["tool_choice"] == {"type": "function", "name": "f"}
    a = D.translate_request({"model": "x", "max_tokens": 5, "system": "s",
                             "messages": [{"role": "user", "content": "hi"}]}, "anthropic", "responses")
    assert a["instructions"] == "s" and a["input"][0]["role"] == "user"


def test_import_rejects_junk(tmp_path):
    lg = ChatGPTLogins(tmp_path)
    for bad in ("nope", "{}", '{"tokens": {"access_token": "a"}}'):
        try:
            lg.import_auth_json("x", bad)
            raise AssertionError("accepted junk")
        except Exception as e:
            assert "JSON" in str(e) or "refresh_token" in str(e)
