"""Fake upstreams and a stack builder for llmanifold tests."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import pytest
from aiohttp import web

import yaml

from llmanifold.apps import build_admin_app, build_api_app
from llmanifold.config import load, parse
from llmanifold.proxy import Core
from llmanifold.store import Store


@dataclass
class FakeState:
    mode: str = "ok"            # ok | 500 | 429 | context | badreq | empty | stream_error | midfail | think | stall
    text: str = "hello from fake"
    texts: list = field(default_factory=list)   # replies for successive requests, before falling back to `text`
    tool: bool = False
    delay: float = 0.0          # before the first byte
    chunk_delay: float = 0.0
    active: int = 0
    max_active: int = 0
    bodies: list = field(default_factory=list)
    headers: list = field(default_factory=list)


def _sse(obj, event=None):
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(obj)}\n\n".encode()


def fake_openai(state: FakeState) -> web.Application:
    async def chat(req: web.Request):
        body = await req.json()
        state.bodies.append(body)
        state.headers.append(dict(req.headers))
        state.active += 1
        state.max_active = max(state.max_active, state.active)
        try:
            await asyncio.sleep(state.delay)
            m = state.mode
            if m == "500":
                return web.json_response({"error": {"message": "boom"}}, status=500)
            if m == "429":
                return web.json_response({"error": {"message": "slow down"}}, status=429)
            if m == "context":
                return web.json_response({"error": {"message": "prompt (300000 tokens) exceeds the context (262144)"}},
                                         status=400)
            if m == "badreq":
                return web.json_response({"error": {"message": "temperature must be <= 2"}}, status=400)
            text = "" if m in ("empty", "stream_error", "think") else (state.texts.pop(0) if state.texts else state.text)
            if m == "think":   # a reasoning model that spent its whole budget thinking
                thought = "let me think about this"
                if not body.get("stream"):
                    return web.json_response({"id": "x", "object": "chat.completion", "model": body.get("model"),
                                              "choices": [{"index": 0, "finish_reason": "length", "message": {
                                                  "role": "assistant", "content": None,
                                                  "reasoning_content": thought}}],
                                              "usage": {"prompt_tokens": 7, "completion_tokens": 5}})
                resp = web.StreamResponse()
                resp.content_type = "text/event-stream"
                await resp.prepare(req)
                base = {"id": "c", "object": "chat.completion.chunk", "model": body.get("model")}
                for w in thought.split(" "):
                    await resp.write(_sse({**base, "choices": [{"index": 0, "delta": {"reasoning_content": w + " "}}]}))
                await resp.write(_sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}))
                await resp.write(b"data: [DONE]\n\n")
                return resp
            words = text.split(" ") if text else []
            if not body.get("stream"):
                msg = {"role": "assistant", "content": text}
                if state.tool and m == "ok":
                    msg["tool_calls"] = [{"id": "call_1", "type": "function",
                                          "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]
                return web.json_response({"id": "x", "object": "chat.completion", "model": body.get("model"),
                                          "choices": [{"index": 0, "message": msg,
                                                       "finish_reason": "tool_calls" if state.tool else "stop"}],
                                          "usage": {"prompt_tokens": 7, "completion_tokens": max(1, len(words))}})
            resp = web.StreamResponse()
            resp.content_type = "text/event-stream"
            await resp.prepare(req)
            base = {"id": "c", "object": "chat.completion.chunk", "model": body.get("model")}
            if m == "stall":   # an overloaded API: accepts the request, then only sends keep-alives
                for _ in range(600):
                    await resp.write(b": keep-alive\n\n")
                    await asyncio.sleep(0.05)
                return resp
            await resp.write(_sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}}]}))
            if m == "stream_error":
                await resp.write(_sse({"error": {"message": "engine crashed"}}))
                return resp
            for i, w in enumerate(words):
                await asyncio.sleep(state.chunk_delay)
                await resp.write(_sse({**base, "choices": [{"index": 0, "delta": {"content": (" " if i else "") + w}}]}))
                if m == "midfail" and i == 1:
                    req.transport.close()
                    return resp
            if state.tool:
                await resp.write(_sse({**base, "choices": [{"index": 0, "delta": {"tool_calls": [
                    {"index": 0, "id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": ""}}]}}]}))
                await resp.write(_sse({**base, "choices": [{"index": 0, "delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": '{"city": "Paris"}'}}]}}]}))
            await resp.write(_sse({**base, "choices": [{"index": 0, "delta": {},
                                                        "finish_reason": "tool_calls" if state.tool else "stop"}]}))
            if (body.get("stream_options") or {}).get("include_usage"):
                await resp.write(_sse({**base, "choices": [],
                                       "usage": {"prompt_tokens": 7, "completion_tokens": len(words)}}))
            await resp.write(b"data: [DONE]\n\n")
            return resp
        finally:
            state.active -= 1

    async def status(req):
        return web.json_response({"busy": state.active > 0, "queued": 0})

    async def models(req):
        return web.json_response({"object": "list", "data": [{"id": "fake"}]})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    async def completions(req):
        return web.json_response({"choices": [{"text": "legacy"}]})

    app.router.add_post("/v1/completions", completions)
    app.router.add_get("/status", status)
    app.router.add_get("/v1/models", models)
    return app


def fake_anthropic(state: FakeState) -> web.Application:
    async def messages(req: web.Request):
        body = await req.json()
        state.bodies.append(body)
        state.headers.append(dict(req.headers))
        await asyncio.sleep(state.delay)
        if state.mode == "500":
            return web.json_response({"type": "error", "error": {"type": "api_error", "message": "boom"}}, status=500)
        text = state.text
        if not body.get("stream"):
            content = [{"type": "text", "text": text}]
            if state.tool:
                content.append({"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                                "input": {"city": "Paris"}})
            return web.json_response({"id": "msg_1", "type": "message", "role": "assistant",
                                      "model": body.get("model"), "content": content,
                                      "stop_reason": "tool_use" if state.tool else "end_turn",
                                      "usage": {"input_tokens": 9, "output_tokens": 3}})
        resp = web.StreamResponse()
        resp.content_type = "text/event-stream"
        await resp.prepare(req)
        await resp.write(_sse({"type": "message_start", "message": {"id": "msg_1", "type": "message",
                               "role": "assistant", "content": [], "model": body.get("model"),
                               "usage": {"input_tokens": 9, "output_tokens": 0}}}, "message_start"))
        await resp.write(_sse({"type": "content_block_start", "index": 0,
                               "content_block": {"type": "text", "text": ""}}, "content_block_start"))
        for w in text.split(" "):
            await resp.write(_sse({"type": "content_block_delta", "index": 0,
                                   "delta": {"type": "text_delta", "text": w + " "}}, "content_block_delta"))
        await resp.write(_sse({"type": "content_block_stop", "index": 0}, "content_block_stop"))
        if state.tool:
            await resp.write(_sse({"type": "content_block_start", "index": 1, "content_block": {
                "type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}}, "content_block_start"))
            await resp.write(_sse({"type": "content_block_delta", "index": 1, "delta": {
                "type": "input_json_delta", "partial_json": '{"city": "Paris"}'}}, "content_block_delta"))
            await resp.write(_sse({"type": "content_block_stop", "index": 1}, "content_block_stop"))
        await resp.write(_sse({"type": "message_delta", "delta": {"stop_reason": "tool_use" if state.tool else "end_turn"},
                               "usage": {"output_tokens": 4}}, "message_delta"))
        await resp.write(_sse({"type": "message_stop"}, "message_stop"))
        return resp

    app = web.Application()
    app.router.add_post("/v1/messages", messages)
    async def amodels(req):
        return web.json_response({"data": [{"id": "claude-fake"}]})

    app.router.add_get("/v1/models", amodels)
    return app


@dataclass
class Stack:
    api: object
    admin: object
    core: Core
    fakes: dict
    hooks: list


@pytest.fixture
async def stack(aiohttp_server, aiohttp_client, tmp_path):
    """make(endpoints={name: {"kind": "openai"|"anthropic"|"dead", **endpoint cfg}}, models={...}, **top)"""
    made = []

    async def make(endpoints: dict, models: dict, **top) -> Stack:
        fakes, eps = {}, {}
        hooks: list = []
        for name, spec in endpoints.items():
            spec = dict(spec)
            kind = spec.pop("kind", "openai")
            if kind == "dead":
                eps[name] = {"url": "http://127.0.0.1:9", "probe": "none", **spec}
                continue
            st = FakeState(**spec.pop("fake", {}))
            srv = await aiohttp_server(fake_openai(st) if kind == "openai" else fake_anthropic(st))
            fakes[name] = st
            eps[name] = {"url": str(srv.make_url("")), "dialect": kind,
                         "probe": "strata" if kind == "openai" else "none", **spec}
        if top.pop("webhook", False):
            async def hook(req):
                hooks.append(await req.json())
                return web.json_response({"ok": True})
            happ = web.Application()
            happ.router.add_post("/hook", hook)
            hsrv = await aiohttp_server(happ)
            top.setdefault("webhooks", [{"url": str(hsrv.make_url("/hook"))}])
        raw = {"data_dir": str(tmp_path), "endpoints": eps, "models": models, "keepalive_after": 0, **top}
        path = tmp_path / f"config{len(made)}.yaml"
        path.write_text("# test config\n" + yaml.safe_dump(raw, sort_keys=False))
        cfg = load(path)
        store = Store(tmp_path / f"t{len(made)}.db")
        core = Core(cfg, store)
        await core.start()
        api = await aiohttp_client(build_api_app(core))
        admin = await aiohttp_client(build_admin_app(core))
        s = Stack(api, admin, core, fakes, hooks)
        made.append((core, store))
        return s

    yield make
    for core, store in made:
        await core.stop()
        store.close()
