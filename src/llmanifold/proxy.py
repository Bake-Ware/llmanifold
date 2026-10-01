"""The request path: auth, endpoint selection, attempts with fallback, and the reply.

A request goes to its model's pool first (waiting in the queue up to
`queue_timeout`), then down the model's fallback list when the failure matches
one of its `fallback_on` triggers. Nothing reaches the client until the chosen
endpoint has produced real content, so a failure before that point can still
switch endpoints. Long waits send keepalives (SSE comments, or JSON leading
whitespace) so proxies such as Cloudflare don't time the request out.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

import aiohttp
from aiohttp import web

from . import dialects as D
from .config import Config, Model
from .router import EndpointState, Router
from .store import Store

log = logging.getLogger("llmanifold")

CLIENT_GONE = (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)
UPSTREAM_DOWN = (aiohttp.ClientConnectorError, aiohttp.ServerDisconnectedError, aiohttp.ClientOSError)


def conversation_key(body: dict, dialect: str) -> str | None:
    msgs = body.get("messages") or []
    if dialect == "anthropic":
        head = [body.get("system"), msgs[:1]]
    else:
        head = [msgs[:2]]
    if not any(head):
        return None
    return hashlib.sha1(json.dumps(head, sort_keys=True, default=str).encode()).hexdigest()


def estimate_tokens(body: dict) -> int:
    size = len(json.dumps([body.get("system"), body.get("messages"), body.get("tools")], default=str))
    return int(size / 3.5)


def classify(status: int, text: str) -> str:
    if status == 429:
        return "429"
    if status >= 500:
        return "5xx"
    if D.is_context_error(status, text):
        return "context"
    return "4xx"


@dataclass
class Outcome:
    ok: bool
    trigger: str | None = None       # why it failed (a fallback trigger name), or "client"
    status: int | None = None
    error: str | None = None
    committed: bool = False          # content already reached the client: no fallback possible
    tokens_in: int = 0
    tokens_out: int = 0
    ttft: float | None = None


@dataclass
class Ctx:
    rid: str
    model: Model
    requested: str
    dialect: str                   # the client's dialect
    stream: bool
    background: bool
    authed: bool
    client: str
    body: dict
    key: str | None
    include_usage: bool
    started: float = field(default_factory=time.monotonic)
    queued: float = 0.0
    attempts: list = field(default_factory=list)


class Responder:
    """Owns the client response: commit headers once, keepalives, content, errors."""

    def __init__(self, request: web.Request, dialect: str, stream: bool, keepalive_after: float) -> None:
        self.request = request
        self.dialect = dialect
        self.stream = stream
        self.keepalive_after = keepalive_after
        self.resp: web.StreamResponse | None = None
        self.endpoint: str | None = None
        self._ka: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    @property
    def committed(self) -> bool:
        return self.resp is not None

    def start_keepalive(self) -> None:
        if self.keepalive_after > 0 and self._ka is None:
            self._ka = asyncio.create_task(self._keepalive())

    async def _keepalive(self) -> None:
        try:
            await asyncio.sleep(self.keepalive_after)
            async with self._lock:
                await self._commit(200)
            while True:
                async with self._lock:
                    await self.resp.write(b": keepalive\n\n" if self.stream else b" ")
                await asyncio.sleep(15)
        except asyncio.CancelledError:
            pass
        except CLIENT_GONE:
            pass

    async def _stop_keepalive(self) -> None:
        if self._ka:
            self._ka.cancel()
            try:
                await self._ka
            except BaseException:
                pass
            self._ka = None

    async def _commit(self, status: int, endpoint: str | None = None) -> None:
        if self.resp is not None:
            return
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        if endpoint:
            headers["X-LLManifold-Endpoint"] = endpoint
        resp = web.StreamResponse(status=status, headers=headers)
        resp.content_type = "text/event-stream" if self.stream else "application/json"
        await resp.prepare(self.request)
        self.resp = resp

    async def start_stream(self, endpoint: str) -> None:
        await self._stop_keepalive()
        self.endpoint = endpoint
        async with self._lock:
            await self._commit(200, endpoint)

    async def write(self, data: bytes) -> None:
        if data:
            async with self._lock:
                await self.resp.write(data)

    async def send_json(self, status: int, obj: dict, endpoint: str | None = None) -> None:
        await self._stop_keepalive()
        self.endpoint = endpoint or self.endpoint
        async with self._lock:
            if self.resp is None:
                body = json.dumps(obj).encode()
                headers = {"X-LLManifold-Endpoint": endpoint} if endpoint else {}
                resp = web.Response(status=status, body=body, content_type="application/json", headers=headers)
                await resp.prepare(self.request)
                await resp.write_eof()
                self.resp = resp
                return
            await self.resp.write(json.dumps(obj).encode())
            await self.resp.write_eof()

    async def send_error(self, status: int, message: str, kind: str = "api_error") -> None:
        await self._stop_keepalive()
        body = D.error_body(self.dialect, message, kind)
        if self.resp is None or not self.stream:
            if self.resp is None or not isinstance(self.resp, web.Response):
                try:
                    await self.send_json(status, body)
                except CLIENT_GONE:
                    pass
            return
        try:
            async with self._lock:
                if self.dialect == "anthropic":
                    await self.resp.write(D.sse(body, "error"))
                else:
                    await self.resp.write(D.sse(body) + b"data: [DONE]\n\n")
                await self.resp.write_eof()
        except CLIENT_GONE:
            pass

    async def end(self) -> None:
        await self._stop_keepalive()
        if self.resp is not None and not isinstance(self.resp, web.Response):
            try:
                await self.resp.write_eof()
            except CLIENT_GONE:
                pass

    def result(self) -> web.StreamResponse:
        # nothing committed means the client went away first; aiohttp still wants a response
        return self.resp if self.resp is not None else web.Response(status=499)


class Core:
    def __init__(self, cfg: Config, store: Store) -> None:
        self.cfg = cfg
        self.store = store
        self.router = Router(cfg)
        self.session: aiohttp.ClientSession | None = None
        self.recent: deque[dict] = deque(maxlen=500)
        self.started = time.time()
        self.reload_error: str | None = None
        self.counters: dict[str, int] = {"requests": 0, "ok": 0, "errors": 0, "fallbacks": 0,
                                         "metered": 0, "unauthorized": 0}
        self._tasks: list[asyncio.Task] = []

    # ---------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        self.session = aiohttp.ClientSession(auto_decompress=True)
        self._tasks.append(asyncio.create_task(self._probe_loop()))
        self._tasks.append(asyncio.create_task(self._prune_loop()))

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except BaseException:
                pass
        if self.session:
            await self.session.close()

    def apply_config(self, cfg: Config) -> None:
        self.cfg = cfg
        self.router.update(cfg)
        self.reload_error = None

    # ---------------------------------------------------------------- health + busy probes
    async def probe(self, st: EndpointState) -> None:
        ep = st.cfg
        if ep.probe == "none":
            return
        path = {"models": "/v1/models", "strata": "/status", "llamacpp": "/slots"}[ep.probe]
        headers = dict(ep.headers)
        key = ep.api_key()
        if key:
            headers["Authorization" if ep.dialect == "openai" else "x-api-key"] = \
                f"Bearer {key}" if ep.dialect == "openai" else key
        if ep.dialect == "anthropic":
            headers.setdefault("anthropic-version", "2023-06-01")
        st.last_check = time.time()
        try:
            async with self.session.get(ep.url + path, headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status >= 500:
                    st.mark_failure(f"probe HTTP {r.status}")
                    return
                busy = 0
                if ep.probe in ("strata", "llamacpp"):
                    try:
                        j = await r.json(content_type=None)
                    except ValueError:
                        j = None
                    if ep.probe == "strata" and isinstance(j, dict):
                        busy = (1 if j.get("busy") else 0) + int(j.get("queued") or 0)
                    elif ep.probe == "llamacpp" and isinstance(j, list):
                        busy = sum(1 for s in j if s.get("is_processing"))
                st.probe_busy, st.probe_at = busy, time.monotonic()
                st.mark_ok()
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            st.mark_failure(f"probe: {type(e).__name__}: {e}".strip(": "))
        self.router.wake()

    async def _probe_loop(self) -> None:
        last_health: dict[str, float] = {}
        while True:
            try:
                now = time.monotonic()
                jobs = []
                for st in list(self.router.states.values()):
                    if st.cfg.probe == "none":
                        continue
                    busy_probe = st.cfg.probe in ("strata", "llamacpp")
                    due = (now - last_health.get(st.name, 0)) >= self.cfg.health_interval
                    if busy_probe or due or not st.healthy:
                        last_health[st.name] = now
                        jobs.append(self.probe(st))
                if jobs:
                    await asyncio.gather(*jobs, return_exceptions=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("probe loop")
            await asyncio.sleep(2.0)

    async def _prune_loop(self) -> None:
        while True:
            try:
                await asyncio.to_thread(self.store.prune, self.cfg.history_days)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("prune")
            await asyncio.sleep(3600)

    # ---------------------------------------------------------------- auth
    def effective_auth(self, model: Model) -> str:
        return self.store.auth_override(model.name) or model.auth

    @staticmethod
    def client_token(request: web.Request) -> str | None:
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return request.headers.get("x-api-key")

    # ---------------------------------------------------------------- listing
    def models_listing(self) -> dict:
        data = []
        for m in self.cfg.models.values():
            for name in [m.name, *m.aliases]:
                entry = {"id": name, "object": "model", "type": "model", "created": int(self.started),
                         "owned_by": "llmanifold", "display_name": name,
                         "auth": self.effective_auth(m)}
                if name != m.name:
                    entry["alias_of"] = m.name
                if m.context:
                    entry.update({"context_length": m.context, "max_model_len": m.context,
                                  "meta": {"n_ctx": m.context}})
                data.append(entry)
        return {"object": "list", "data": data, "has_more": False}

    # ---------------------------------------------------------------- main entry
    async def handle(self, request: web.Request, dialect: str) -> web.StreamResponse:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
        except ValueError as e:
            return web.json_response(D.error_body(dialect, f"invalid JSON body: {e}", "invalid_request_error"),
                                     status=400)
        requested = body.get("model") or ""
        model = self.cfg.resolve(requested)
        if model is None:
            return web.json_response(D.error_body(dialect, f"unknown model {requested!r}; GET /v1/models lists them",
                                                  "not_found_error"), status=404)
        tok = self.store.lookup_token(self.client_token(request))
        authed = bool(tok) and ("*" in tok["models"] or model.name in tok["models"]
                                or requested in tok["models"])
        if self.effective_auth(model) == "token" and not authed:
            self.counters["unauthorized"] += 1
            return web.json_response(D.error_body(dialect, f"model {requested!r} needs a valid token",
                                                  "authentication_error"), status=401)
        if tok:
            asyncio.get_running_loop().run_in_executor(None, self.store.touch_token, tok["id"])
        prio = request.headers.get(self.cfg.priority_header, "").lower()
        background = (bool(tok) and tok["background"]) or prio == "background"
        stream = bool(body.get("stream"))
        so = body.get("stream_options") or {}
        ctx = Ctx(rid=uuid.uuid4().hex[:12], model=model, requested=requested, dialect=dialect, stream=stream,
                  background=background, authed=authed,
                  client=(tok["label"] if tok else (request.headers.get("X-Forwarded-For") or request.remote or "?")),
                  body=body, key=conversation_key(body, dialect) if model.affinity else None,
                  include_usage=bool(so.get("include_usage")))
        self.counters["requests"] += 1
        responder = Responder(request, dialect, stream, self.cfg.keepalive_after)
        responder.start_keepalive()
        outcome = None
        served: EndpointState | None = None
        try:
            outcome, served = await self._route(ctx, responder)
            if not outcome.ok and not outcome.committed and outcome.trigger != "client":
                status = outcome.status if outcome.status and outcome.trigger == "4xx" else \
                    {"queue": 503, "timeout": 504, "slow": 504}.get(outcome.trigger or "", 502)
                msg = outcome.error or "no endpoint could serve the request"
                if outcome.trigger == "4xx":
                    msg = _upstream_message(msg)
                await responder.send_error(status, f"llmanifold: {msg}" if outcome.trigger != "4xx" else msg,
                                           "invalid_request_error" if outcome.trigger == "4xx" else "api_error")
        except CLIENT_GONE:
            outcome = outcome or Outcome(False, "client", committed=True)
        finally:
            await responder._stop_keepalive()
            self._finish(ctx, outcome, served)
        return responder.result()

    async def _route(self, ctx: Ctx, responder: Responder) -> tuple[Outcome, EndpointState | None]:
        model = ctx.model
        min_ctx = None
        est = estimate_tokens(ctx.body)
        if model.context:
            min_ctx = est + int(ctx.body.get("max_tokens") or 0) if est > 2048 else None
        last = Outcome(False, "queue", error="no endpoint available")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + model.queue_timeout
        desc = f"{'streaming to' if ctx.stream else 'answering'} {ctx.client}"
        # 1. the pool
        tried_pool = 0
        while model.pool and tried_pool < max(1, len(model.pool)):
            tried_pool += 1
            t_q = time.monotonic()
            st = await self.router.acquire(model, key=ctx.key, background=ctx.background,
                                           timeout=max(0.0, deadline - loop.time()),
                                           rid=ctx.rid, desc=desc, min_context=min_ctx)
            ctx.queued += time.monotonic() - t_q
            if st is None:
                trig = "queue" if self.router.pool_available(model, min_ctx) else "connect"
                if not self.router.pool_available(model, None) and min_ctx:
                    trig = "context"
                last = Outcome(False, trig, error={"queue": "all pool endpoints stayed busy",
                                                   "connect": "no healthy pool endpoint",
                                                   "context": "prompt is larger than the pool's context"}[trig])
                ctx.attempts.append({"endpoint": "pool", "trigger": trig})
                break
            last = await self._attempt(ctx, st, responder)
            ctx.attempts.append({"endpoint": st.name, "ok": last.ok, "trigger": last.trigger,
                                 "status": last.status})
            if last.ok or last.committed or last.trigger == "client":
                return last, st
            if last.trigger != "connect":
                break       # other pool members would most likely fail the same way
        # 2. fallbacks
        if model.fallback and (last.trigger in model.fallback_on):
            for name in model.fallback:
                ep = self.cfg.endpoints.get(name)
                if ep is None:
                    continue
                if ep.metered and not ctx.authed and not model.allow_metered_unauthenticated:
                    ctx.attempts.append({"endpoint": name, "skipped": "metered endpoint needs a token"})
                    continue
                st = await self.router.try_endpoint(name, background=ctx.background, rid=ctx.rid, desc=desc)
                if st is None:
                    ctx.attempts.append({"endpoint": name, "skipped": "busy or unhealthy"})
                    continue
                self.counters["fallbacks"] += 1
                out = await self._attempt(ctx, st, responder)
                ctx.attempts.append({"endpoint": name, "ok": out.ok, "trigger": out.trigger,
                                     "status": out.status, "fallback": True})
                if out.ok or out.committed or out.trigger == "client":
                    return out, st
                last = out
                if out.trigger not in model.fallback_on:
                    break
        return last, None

    async def _attempt(self, ctx: Ctx, st: EndpointState, responder: Responder) -> Outcome:
        ep = st.cfg
        try:
            up = D.translate_request(ctx.body, ctx.dialect, ep.dialect)
            up["model"] = ep.model or ctx.model.name
            if ctx.stream and ep.dialect == "openai" and ep.stream_usage:
                # always ask the engine for token counts; stripped again for clients that didn't ask
                up["stream_options"] = {**(up.get("stream_options") or {}), "include_usage": True}
            path = "/v1/chat/completions" if ep.dialect == "openai" else "/v1/messages"
            headers = {"Content-Type": "application/json"}
            key = ep.api_key()
            if ep.dialect == "openai":
                if key:
                    headers["Authorization"] = f"Bearer {key}"
            else:
                if key:
                    headers["x-api-key"] = key
                headers["anthropic-version"] = responder.request.headers.get("anthropic-version", "2023-06-01")
                beta = responder.request.headers.get("anthropic-beta")
                if beta and ctx.dialect == "anthropic":
                    headers["anthropic-beta"] = beta
            headers.update(ep.headers)
            return await self._call(ctx, st, responder, ep.url + path, up, headers)
        finally:
            await self.router.release(st, ctx.background, ctx.rid)

    async def _call(self, ctx: Ctx, st: EndpointState, responder: Responder, url: str, up: dict,
                    headers: dict) -> Outcome:
        ep = st.cfg
        t0 = time.monotonic()
        content_started = False
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=ep.connect_timeout, sock_read=ep.timeout)
        try:
            async with self.session.post(url, json=up, headers=headers, timeout=timeout) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    trig = classify(resp.status, text)
                    if trig == "5xx":
                        st.mark_failure(f"HTTP {resp.status}")
                    return Outcome(False, trig, resp.status, text[:2000])
                if not ctx.stream:
                    try:
                        data = await resp.json(content_type=None)
                    except ValueError:
                        return Outcome(False, "5xx", resp.status, "upstream returned invalid JSON")
                    if not isinstance(data, dict) or not D.response_has_content(data, ep.dialect):
                        return Outcome(False, "empty", resp.status, "upstream returned no content")
                    out = D.translate_response(data, ep.dialect, ctx.dialect, ctx.requested)
                    tin, tout = D.response_usage(data, ep.dialect)
                    elapsed = time.monotonic() - t0
                    await responder.send_json(200, out, st.name)
                    st.mark_ok()
                    st.observe(tin, tout, elapsed, None)
                    return Outcome(True, status=200, tokens_in=tin, tokens_out=tout, committed=True)
                # streaming: hold until real content, so a fast failure can still fall back
                tr = D.stream_translator(ep.dialect, ctx.dialect, ctx.requested,
                                         include_usage=ctx.include_usage, rename=True,
                                         strip_usage=ep.stream_usage and not ctx.include_usage)
                parser = D.SSEParser()
                held: list[bytes] = []
                first_deadline = t0 + ctx.model.first_token_timeout
                ttft = None
                it = resp.content.iter_any().__aiter__()
                while True:
                    wait = None if content_started else first_deadline - time.monotonic()
                    if wait is not None and wait <= 0:
                        return Outcome(False, "slow", 200, "no first token before first_token_timeout")
                    try:
                        chunk = await (asyncio.wait_for(it.__anext__(), wait) if wait is not None else it.__anext__())
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        return Outcome(False, "slow", 200, "no first token before first_token_timeout")
                    outs: list[bytes] = []
                    for ev in parser.feed(chunk):
                        outs += tr.feed(*ev)
                    if not content_started:
                        if tr.error:
                            return Outcome(False, "5xx", 200, f"upstream stream error: {tr.error}")
                        held += outs
                        if tr.has_content:
                            ttft = time.monotonic() - t0
                            await responder.start_stream(st.name)
                            content_started = True
                            await responder.write(b"".join(held))
                            held = []
                    elif outs:
                        await responder.write(b"".join(outs))
                outs = []
                for ev in parser.flush():
                    outs += tr.feed(*ev)
                if not content_started:
                    if tr.error:
                        return Outcome(False, "5xx", 200, f"upstream stream error: {tr.error}")
                    return Outcome(False, "empty", 200, "upstream stream ended without content")
                outs += tr.finish()
                await responder.write(b"".join(outs))
                await responder.end()
                gen = time.monotonic() - t0 - (ttft or 0)
                tout = tr.tokens_out or tr.deltas
                st.mark_ok()
                st.observe(tr.tokens_in, tout, gen, ttft)
                return Outcome(True, status=200, tokens_in=tr.tokens_in, tokens_out=tout,
                               ttft=ttft, committed=True)
        except CLIENT_GONE:
            return Outcome(False, "client", committed=True, error="client disconnected")
        except asyncio.TimeoutError:
            if content_started:
                await responder.send_error(504, "llmanifold: upstream stopped sending")
                return Outcome(False, "timeout", committed=True, error="upstream read timeout mid-stream")
            return Outcome(False, "timeout", error="upstream timed out")
        except UPSTREAM_DOWN as e:
            st.mark_failure(f"{type(e).__name__}: {e}", immediate=not content_started)
            if content_started:
                await responder.send_error(502, "llmanifold: upstream connection lost")
                return Outcome(False, "connect", committed=True, error=str(e))
            return Outcome(False, "connect", error=f"{st.name}: {type(e).__name__}")
        except aiohttp.ClientPayloadError as e:
            if content_started:
                await responder.send_error(502, "llmanifold: upstream stream broke")
                return Outcome(False, "5xx", committed=True, error=str(e))
            return Outcome(False, "5xx", error=str(e))

    # ---------------------------------------------------------------- bookkeeping
    def _finish(self, ctx: Ctx, outcome: Outcome | None, served: EndpointState | None) -> None:
        outcome = outcome or Outcome(False, "client", error="aborted")
        duration = time.monotonic() - ctx.started
        ok = outcome.ok
        self.counters["ok" if ok else "errors"] += 1
        metered = bool(served and served.cfg.metered and ok)
        fallback = any(a.get("fallback") for a in ctx.attempts)
        tps = None
        if ok and outcome.tokens_out and duration - (outcome.ttft or 0) > 0.2:
            tps = round(outcome.tokens_out / max(0.001, duration - ctx.queued - (outcome.ttft or 0)), 1)
        row = {"id": ctx.rid, "ts": time.time(), "model": ctx.model.name, "endpoint": served.name if served else None,
               "client": ctx.client, "priority": "background" if ctx.background else "interactive",
               "dialect_in": ctx.dialect, "dialect_out": served.cfg.dialect if served else None,
               "stream": int(ctx.stream), "status": outcome.status or (200 if ok else None), "ok": int(ok),
               "attempts": ctx.attempts, "fallback": int(fallback), "metered": int(metered),
               "tokens_in": outcome.tokens_in, "tokens_out": outcome.tokens_out,
               "queued_ms": int(ctx.queued * 1000),
               "ttft_ms": int(outcome.ttft * 1000) if outcome.ttft is not None else None,
               "duration_ms": int(duration * 1000), "tps": tps,
               "error": None if ok else _trail(ctx.attempts, outcome)}
        self.recent.appendleft(row)
        if metered:
            self.counters["metered"] += 1
        loop = asyncio.get_running_loop()
        loop.create_task(self._record(row, metered))

    async def _record(self, row: dict, metered: bool) -> None:
        try:
            await self.store.arecord(row)
        except Exception:
            log.exception("history write failed")
        if metered:
            await self.fire_webhooks("metered", {k: row[k] for k in (
                "id", "ts", "model", "endpoint", "client", "tokens_in", "tokens_out", "attempts")})

    async def fire_webhooks(self, event: str, data: dict) -> None:
        for hook in self.cfg.webhooks:
            if event not in hook.events:
                continue
            payload = {"event": event, "source": "llmanifold", **data,
                       "text": f"llmanifold: {data.get('model')} was served by metered endpoint "
                               f"{data.get('endpoint')} for {data.get('client')} "
                               f"({data.get('tokens_in', 0)} in / {data.get('tokens_out', 0)} out tokens)"}
            try:
                async with self.session.post(hook.url, json=payload, headers=hook.headers,
                                             timeout=aiohttp.ClientTimeout(total=10)) as r:
                    if r.status >= 400:
                        log.warning("webhook %s -> HTTP %s", hook.url, r.status)
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                log.warning("webhook %s failed: %s", hook.url, e)


def _trail(attempts: list[dict], outcome: Outcome) -> str:
    """One readable line: what each endpoint did, e.g. 'lane-a: boom; deepseek skipped: ...'."""
    parts = []
    for a in attempts:
        if a.get("skipped"):
            parts.append(f"{a['endpoint']} skipped: {a['skipped']}")
        elif a.get("trigger"):
            parts.append(f"{a['endpoint']}: {a['trigger']}")
    msg = _upstream_message(outcome.error) if outcome.error else (outcome.trigger or "failed")
    if parts:
        return f"{msg} ({'; '.join(parts)})"[:500]
    return msg[:500]


def _upstream_message(text: str) -> str:
    try:
        j = json.loads(text)
    except ValueError:
        return text[:500]
    err = j.get("error") if isinstance(j, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or err)[:500]
    if isinstance(err, str):
        return err[:500]
    return text[:500]
