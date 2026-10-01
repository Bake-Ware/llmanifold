"""The two HTTP listeners.

* API listener: the model API (OpenAI + Anthropic shapes) plus a few
  compatibility endpoints. Safe to publish (e.g. through a public tunnel):
  nothing here can change state, and it never shows backend URLs.
* Admin listener: dashboard, admin API and metrics. Every request is checked
  against `admin.allow_from` / `admin.trusted_proxies`; token and model-auth
  changes additionally need a *human*, i.e. a request a trusted proxy (such as
  Cloudflare Access) vouches for with a user email.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import aiohttp
from aiohttp import web

from . import __version__
from . import dialects as D
from .config import Config, ConfigError, load
from .proxy import Core, estimate_tokens

log = logging.getLogger("llmanifold")
STATIC = Path(__file__).parent / "static"
CORE = web.AppKey("core", Core)


# ======================================================================= API listener

async def chat(request: web.Request) -> web.StreamResponse:
    return await request.app[CORE].handle(request, "openai")


async def messages(request: web.Request) -> web.StreamResponse:
    return await request.app[CORE].handle(request, "anthropic")


async def count_tokens(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except ValueError:
        body = {}
    return web.json_response({"input_tokens": estimate_tokens(body or {})})


async def models(request: web.Request) -> web.Response:
    return web.json_response(request.app[CORE].models_listing())


async def healthz(request: web.Request) -> web.Response:
    core = request.app[CORE]
    states = core.router.states.values()
    return web.json_response({"ok": any(s.healthy for s in states), "version": __version__,
                              "models": sorted(core.cfg.models)})


def _public_slots(core: Core) -> list[dict]:
    out, i = [], 0
    for m in core.cfg.models.values():
        for n in m.pool:
            st = core.router.states.get(n)
            if st is None:
                continue
            for _ in range(st.cfg.max_concurrency):
                out.append({"id": i, "model": m.name, "endpoint": n, "healthy": st.healthy,
                            "draining": st.draining, "is_processing": st.load() > 0, "n_ctx": st.cfg.context})
                i += 1
    return out


async def slots(request: web.Request) -> web.Response:
    return web.json_response(_public_slots(request.app[CORE]))


async def status(request: web.Request) -> web.Response:
    core = request.app[CORE]
    s = _public_slots(core)
    return web.json_response({"slots": len(s), "busy": sum(1 for x in s if x["is_processing"]),
                              "queued": core.router.queue_depth(), "detail": s})


async def props(request: web.Request) -> web.Response:
    """llama.cpp-style /props, which some clients read for the context size."""
    core = request.app[CORE]
    m = core.cfg.resolve(request.query.get("model"), use_default=True) or next(iter(core.cfg.models.values()), None)
    ctx = (m.context if m else None) or 0
    return web.json_response({"model_alias": m.name if m else None,
                              "default_generation_settings": {"n_ctx": ctx}, "n_ctx": ctx})


async def passthrough(request: web.Request) -> web.StreamResponse:
    """Any other POST /v1/* (completions, embeddings, ...): forwarded unchanged to an
    OpenAI-dialect pool endpoint of the named model. No translation, no fallback."""
    core = request.app[CORE]
    raw = await request.read()
    try:
        body = json.loads(raw or b"{}")
    except ValueError:
        return web.json_response(D.error_body("openai", "invalid JSON body"), status=400)
    model = core.cfg.resolve(body.get("model"), use_default=True)
    if model is None:
        return web.json_response(D.error_body("openai", f"unknown model {body.get('model')!r}"), status=404)
    tok = core.store.lookup_token(core.client_token(request))
    if core.effective_auth(model) == "token" and not (tok and ("*" in tok["models"] or model.name in tok["models"])):
        return web.json_response(D.error_body("openai", "this model needs a valid token"), status=401)
    st = await core.router.acquire(model, key=None, background=False, timeout=model.queue_timeout,
                                   rid=os.urandom(6).hex(), desc=f"passthrough {request.path}")
    if st is None or st.cfg.dialect != "openai":
        if st is not None:
            await core.router.release(st, False)
        return web.json_response(D.error_body("openai", "no OpenAI-compatible endpoint free"), status=503)
    try:
        if st.cfg.model:
            body["model"] = st.cfg.model
        headers = {"Content-Type": "application/json", **st.cfg.headers}
        key = st.cfg.api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        async with core.session.post(st.cfg.url + request.path, json=body, headers=headers,
                                     timeout=aiohttp.ClientTimeout(total=None, sock_connect=st.cfg.connect_timeout,
                                                                   sock_read=st.cfg.timeout)) as up:
            resp = web.StreamResponse(status=up.status,
                                      headers={"Content-Type": up.headers.get("Content-Type", "application/json"),
                                               "X-LLManifold-Endpoint": st.name})
            await resp.prepare(request)
            async for chunk in up.content.iter_any():
                await resp.write(chunk)
            await resp.write_eof()
            return resp
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return web.json_response(D.error_body("openai", f"upstream: {e}"), status=502)
    finally:
        await core.router.release(st, False)


def build_api_app(core: Core) -> web.Application:
    app = web.Application(client_max_size=64 * 1024 ** 2)
    app[CORE] = core
    r = app.router
    r.add_post("/v1/chat/completions", chat)
    r.add_post("/chat/completions", chat)
    r.add_post("/v1/messages", messages)
    r.add_post("/v1/messages/count_tokens", count_tokens)
    for p in ("/v1/models", "/models"):
        r.add_get(p, models)
    r.add_get("/healthz", healthz)
    r.add_get("/health", healthz)
    r.add_get("/slots", slots)
    r.add_get("/status", status)
    r.add_get("/props", props)
    r.add_post("/v1/{tail:.+}", passthrough)
    return app


# ======================================================================= admin listener

def _who(request: web.Request, core: Core) -> dict | None:
    """None = not allowed at all. Otherwise {human, email, ip}."""
    a = core.cfg.admin
    ip = request.remote or ""
    if a.trusted_proxies and a.ip_in(ip, a.trusted_proxies):
        email = request.headers.get(a.human_header)
        if not email:
            return None          # through the proxy but not vouched for: refuse
        if a.allowed_emails and email.lower() not in [e.lower() for e in a.allowed_emails]:
            return None
        return {"human": True, "email": email, "ip": ip}
    if a.ip_in(ip, a.allow_from):
        loopback = ip.startswith("127.") or ip == "::1"
        return {"human": bool(a.local_humans and loopback), "email": None, "ip": ip}
    return None


@web.middleware
async def access(request: web.Request, handler):
    who = _who(request, request.app[CORE])
    if who is None:
        return web.json_response({"error": "forbidden"}, status=403)
    request["who"] = who
    return await handler(request)


def _human(request: web.Request) -> web.Response | None:
    if not request["who"]["human"]:
        return web.json_response({"error": "this action needs a person signed in through the admin site"},
                                 status=403)
    return None


async def a_index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC / "index.html")


async def a_whoami(request: web.Request) -> web.Response:
    return web.json_response(request["who"])


def model_rows(core: Core) -> list[dict]:
    rows = []
    for m in core.cfg.models.values():
        mode = core.effective_auth(m)
        metered_fb = [n for n in m.fallback if core.cfg.endpoints[n].metered]
        warn = None
        if mode == "open" and metered_fb and m.allow_metered_unauthenticated:
            warn = "open model can fall back to metered endpoints for anyone"
        rows.append({"name": m.name, "aliases": m.aliases, "pool": m.pool, "fallback": m.fallback,
                     "fallback_on": list(m.fallback_on), "auth": mode, "auth_default": m.auth,
                     "auth_overridden": core.store.auth_override(m.name) is not None,
                     "metered_fallback": metered_fb,
                     "allow_metered_unauthenticated": m.allow_metered_unauthenticated,
                     "queue_timeout": m.queue_timeout, "first_token_timeout": m.first_token_timeout,
                     "background_max_lanes": m.background_max_lanes, "context": m.context,
                     "queued": core.router.queue_depth(m.name), "warning": warn,
                     "description": m.description})
    return rows


async def a_status(request: web.Request) -> web.Response:
    core = request.app[CORE]
    return web.json_response({
        "version": __version__, "uptime": int(time.time() - core.started), "config": core.cfg.path,
        "reload_error": core.reload_error, "counters": core.counters,
        "endpoints": [s.snapshot() for s in core.router.states.values()],
        "models": model_rows(core), "queue": core.router.queue(), "who": request["who"],
        "listen": {"api": core.cfg.api_listen, "admin": core.cfg.admin_listen},
        "webhooks": [{"url": w.url.split("?")[0], "events": list(w.events)} for w in core.cfg.webhooks]})


async def a_requests(request: web.Request) -> web.Response:
    core = request.app[CORE]
    limit = min(1000, int(request.query.get("limit", 100)))
    if request.query.get("source") == "db":
        rows = await asyncio.to_thread(core.store.recent, limit)
    else:
        rows = list(core.recent)[:limit]
    return web.json_response(rows)


async def a_timeseries(request: web.Request) -> web.Response:
    core = request.app[CORE]
    minutes = min(7 * 24 * 60, int(request.query.get("minutes", 60)))
    bucket = max(10, int(request.query.get("bucket", 60)))
    return web.json_response(await asyncio.to_thread(core.store.timeseries, minutes, bucket))


async def a_drain(request: web.Request) -> web.Response:
    core = request.app[CORE]
    name = request.match_info["name"]
    on = request.match_info["action"] == "drain"
    if not await core.router.set_draining(name, on):
        return web.json_response({"error": f"unknown endpoint {name!r}"}, status=404)
    log.info("%s %s by %s", "drain" if on else "undrain", name, request["who"])
    return web.json_response({"ok": True, "endpoint": name, "draining": on})


async def a_reload(request: web.Request) -> web.Response:
    core = request.app[CORE]
    try:
        cfg = load(core.cfg.path)
    except (ConfigError, OSError) as e:
        core.reload_error = str(e)
        return web.json_response({"ok": False, "error": str(e)}, status=400)
    core.apply_config(cfg)
    return web.json_response({"ok": True, "models": sorted(cfg.models), "endpoints": sorted(cfg.endpoints)})


async def a_tokens(request: web.Request) -> web.Response:
    if (deny := _human(request)) is not None:
        return deny
    return web.json_response(await asyncio.to_thread(request.app[CORE].store.list_tokens))


async def a_token_create(request: web.Request) -> web.Response:
    if (deny := _human(request)) is not None:
        return deny
    core = request.app[CORE]
    body = await request.json()
    label = str(body.get("label") or "").strip()
    if not label:
        return web.json_response({"error": "label is required"}, status=400)
    models_ = body.get("models") or ["*"]
    if isinstance(models_, str):
        models_ = [m.strip() for m in models_.split(",") if m.strip()]
    bad = [m for m in models_ if m != "*" and core.cfg.resolve(m) is None]
    if bad:
        return web.json_response({"error": f"unknown models {bad}"}, status=400)
    tok = await asyncio.to_thread(core.store.create_token, label, models_, bool(body.get("background")),
                                  request["who"].get("email") or request["who"]["ip"])
    return web.json_response(tok)


async def a_token_delete(request: web.Request) -> web.Response:
    if (deny := _human(request)) is not None:
        return deny
    ok = await asyncio.to_thread(request.app[CORE].store.delete_token, int(request.match_info["id"]))
    return web.json_response({"ok": ok}, status=200 if ok else 404)


async def a_model_auth(request: web.Request) -> web.Response:
    if (deny := _human(request)) is not None:
        return deny
    core = request.app[CORE]
    name = request.match_info["name"]
    m = core.cfg.resolve(name)
    if m is None:
        return web.json_response({"error": f"unknown model {name!r}"}, status=404)
    body = await request.json()
    mode = body.get("mode")
    if mode not in ("open", "token", None):
        return web.json_response({"error": "mode must be open, token or null (use the config default)"}, status=400)
    await asyncio.to_thread(core.store.set_auth, m.name, mode, request["who"].get("email"))
    return web.json_response({"ok": True, "model": m.name, "auth": core.effective_auth(m)})


async def a_metrics(request: web.Request) -> web.Response:
    core = request.app[CORE]
    lines = ["# HELP llmanifold_up 1 while llmanifold runs", "# TYPE llmanifold_up gauge", "llmanifold_up 1"]
    for k, v in core.counters.items():
        lines += [f"# TYPE llmanifold_{k}_total counter", f"llmanifold_{k}_total {v}"]
    gauges = {"inflight": "inflight", "healthy": "healthy", "draining": "draining",
              "requests_total": "requests", "errors_total": "errors",
              "tokens_in_total": "tokens_in", "tokens_out_total": "tokens_out",
              "tokens_per_second": "tps_ema", "ttft_seconds": "ttft_ema"}
    for metric, attr in gauges.items():
        lines.append(f"# TYPE llmanifold_endpoint_{metric} {'counter' if metric.endswith('total') else 'gauge'}")
        for st in core.router.states.values():
            v = getattr(st, attr)
            if v is None:
                continue
            lines.append(f'llmanifold_endpoint_{metric}{{endpoint="{st.name}"}} {float(v):g}')
    lines.append("# TYPE llmanifold_queue_depth gauge")
    for m in core.cfg.models:
        lines.append(f'llmanifold_queue_depth{{model="{m}"}} {core.router.queue_depth(m)}')
    return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")


def build_admin_app(core: Core) -> web.Application:
    app = web.Application(middlewares=[access])
    app[CORE] = core
    r = app.router
    r.add_get("/", a_index)
    r.add_static("/static", STATIC)
    r.add_get("/api/whoami", a_whoami)
    r.add_get("/api/status", a_status)
    r.add_get("/api/requests", a_requests)
    r.add_get("/api/timeseries", a_timeseries)
    r.add_post("/api/endpoints/{name}/{action:drain|undrain}", a_drain)
    r.add_post("/api/reload", a_reload)
    r.add_get("/api/tokens", a_tokens)
    r.add_post("/api/tokens", a_token_create)
    r.add_delete("/api/tokens/{id:\\d+}", a_token_delete)
    r.add_post("/api/models/{name}/auth", a_model_auth)
    r.add_get("/metrics", a_metrics)
    return app


# ======================================================================= config watcher + server

async def watch_config(core: Core, interval: float = 2.0) -> None:
    path = core.cfg.path
    if not path:
        return
    try:
        mtime = os.stat(path).st_mtime
    except OSError:
        mtime = 0.0
    while True:
        await asyncio.sleep(interval)
        try:
            m = os.stat(path).st_mtime
        except OSError:
            continue
        if m == mtime:
            continue
        mtime = m
        try:
            cfg = load(path)
        except (ConfigError, OSError) as e:
            core.reload_error = str(e)
            log.error("config reload failed, keeping the old config: %s", e)
            continue
        if (cfg.api_listen, cfg.admin_listen, cfg.data_dir) != (core.cfg.api_listen, core.cfg.admin_listen,
                                                                core.cfg.data_dir):
            log.warning("listen addresses / data_dir changed: restart llmanifold to apply those")
        core.apply_config(cfg)
        log.info("config reloaded: %d models, %d endpoints", len(cfg.models), len(cfg.endpoints))


def _split(hostport: str) -> tuple[str, int]:
    host, _, port = hostport.rpartition(":")
    return host.strip("[]") or "0.0.0.0", int(port)


async def serve(cfg: Config) -> None:
    from .store import Store
    store = Store(Path(cfg.data_dir).expanduser() / "llmanifold.db")
    core = Core(cfg, store)
    await core.start()
    runners = []
    try:
        for app, addr in ((build_api_app(core), cfg.api_listen), (build_admin_app(core), cfg.admin_listen)):
            runner = web.AppRunner(app, access_log=None)
            await runner.setup()
            host, port = _split(addr)
            await web.TCPSite(runner, host, port).start()
            runners.append(runner)
            log.info("listening on %s", addr)
        log.info("llmanifold %s: api %s, admin %s, %d models, %d endpoints", __version__, cfg.api_listen,
                 cfg.admin_listen, len(cfg.models), len(cfg.endpoints))
        await watch_config(core)
        await asyncio.Event().wait()
    finally:
        for r in runners:
            await r.cleanup()
        await core.stop()
        store.close()
