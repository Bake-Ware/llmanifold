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
from .config import DIALECTS, PROBES, TRIGGERS, Config, ConfigError, load
from .editor import ENDPOINT_FIELDS, FLOW_FIELDS, EditError
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
    resp = await handler(request)
    # Proxies (Cloudflare caches .js/.css by default) and browsers must never pair a new page
    # with an old script: the page and API are never cached, static files are fingerprinted.
    if request.path.startswith("/static/") and request.query.get("v"):
        resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    else:
        resp.headers["Cache-Control"] = "no-store"
    return resp


def _fingerprint() -> str:
    import hashlib
    h = hashlib.sha256()
    for f in sorted(STATIC.iterdir()):
        if f.is_file():
            h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:12]


_INDEX: tuple[str, str] | None = None


def _index_html() -> str:
    """index.html with every /static/ URL carrying the assets' content hash."""
    global _INDEX
    fp = _fingerprint()
    if _INDEX is None or _INDEX[0] != fp:
        html = (STATIC / "index.html").read_text()
        for name in ("style.css", "app.js", "art.js"):
            html = html.replace(f"/static/{name}", f"/static/{name}?v={fp}")
        _INDEX = (fp, html)
    return _INDEX[1]


def _human(request: web.Request) -> web.Response | None:
    if not request["who"]["human"]:
        return web.json_response({"error": "this action needs a person signed in through the admin site"},
                                 status=403)
    return None


async def a_index(request: web.Request) -> web.Response:
    return web.Response(text=_index_html(), content_type="text/html")


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
                     "default": bool(core.cfg.default_model and core.cfg.resolve(core.cfg.default_model) is m),
                     "description": m.description})
    return rows


async def a_status(request: web.Request) -> web.Response:
    core = request.app[CORE]
    return web.json_response({
        "version": __version__, "uptime": int(time.time() - core.started), "config": core.cfg.path,
        "reload_error": core.reload_error, "counters": core.counters,
        "tps": core.throughput(),
        "endpoints": [s.snapshot() for s in core.router.states.values()],
        "models": model_rows(core), "queue": core.router.queue(), "who": request["who"],
        "listen": {"api": core.cfg.api_listen, "admin": core.cfg.admin_listen},
        "editable": bool(core.editor and core.editor.writable),
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
    """Pause/resume an endpoint (drain/undrain are the older names). Allowed for any admin
    caller, agents included: it's reversible and stops no running request."""
    core = request.app[CORE]
    name = request.match_info["name"]
    on = request.match_info["action"] in ("drain", "pause")
    who = request["who"]
    if not await core.set_paused(name, on, who.get("email") or who.get("ip")):
        return web.json_response({"error": f"unknown endpoint {name!r}"}, status=404)
    log.info("%s %s by %s", "pause" if on else "resume", name, who)
    return web.json_response({"ok": True, "endpoint": name, "draining": on, "paused": on})


# ---- config editing (people only)

def _endpoint_spec(core: Core, name: str) -> dict:
    e = core.cfg.endpoints[name]
    out = {f: getattr(e, f) for f in ENDPOINT_FIELDS}
    out["key"] = ("file" if e.key_file else "env" if e.key_env else "literal" if e.key else None)
    out["key_set"] = bool(e.api_key())
    out["login"] = e.login
    if e.login == "chatgpt":
        s = core.chatgpt.status(name)
        out["chatgpt"] = {k: s.get(k) for k in ("signed_in", "email", "plan", "last_error")}
    out["flows"] = [m.name for m in core.cfg.models.values() if name in m.pool or name in m.fallback]
    return out


async def a_config(request: web.Request) -> web.Response:
    core = request.app[CORE]
    return web.json_response({
        "path": core.cfg.path, "writable": bool(core.editor and core.editor.writable),
        "default_model": core.cfg.default_model,
        "endpoints": {n: _endpoint_spec(core, n) for n in core.cfg.endpoints},
        "flows": {m.name: {f: (list(getattr(m, f)) if isinstance(getattr(m, f), (list, tuple)) else getattr(m, f))
                           for f in FLOW_FIELDS} for m in core.cfg.models.values()},
        "choices": {"dialect": list(DIALECTS), "probe": list(PROBES), "fallback_on": list(TRIGGERS)}})


async def _config_edit(request: web.Request, what: str, coro_fn) -> web.Response:
    if (deny := _human(request)) is not None:
        return deny
    core = request.app[CORE]
    if core.editor is None:
        return web.json_response({"error": "llmanifold wasn't started from a config file, so it can't save changes"},
                                 status=409)
    try:
        cfg = await coro_fn(core.editor)
    except EditError as e:
        return web.json_response({"error": str(e)}, status=400)
    except OSError as e:
        return web.json_response({"error": f"couldn't write the config: {e}"}, status=500)
    core.apply_config(cfg)
    log.info("config edit (%s) by %s", what, request["who"].get("email") or request["who"]["ip"])
    return web.json_response({"ok": True})


async def _body(request: web.Request) -> dict:
    try:
        b = await request.json()
    except ValueError:
        return {}
    return b if isinstance(b, dict) else {}


def _split_endpoint_body(b: dict) -> tuple[dict, str | None, bool]:
    fields = {k: b[k] for k in ENDPOINT_FIELDS if k in b}
    key = str(b.get("key") or "").strip() or None
    return fields, key, bool(b.get("clear_key"))


async def a_endpoint_create(request: web.Request) -> web.Response:
    b = await _body(request)
    fields, key, clear = _split_endpoint_body(b)
    return await _config_edit(request, f"add model {b.get('name')}", lambda ed: ed.save_endpoint(
        str(b.get("name") or ""), fields, create=True, key=key, clear_key=clear))


async def a_endpoint_update(request: web.Request) -> web.Response:
    b = await _body(request)
    fields, key, clear = _split_endpoint_body(b)
    name = request.match_info["name"]
    return await _config_edit(request, f"edit model {name}", lambda ed: ed.save_endpoint(
        name, fields, create=False, key=key, clear_key=clear))


async def a_endpoint_delete(request: web.Request) -> web.Response:
    name = request.match_info["name"]
    force = request.query.get("force") in ("1", "true", "yes")
    resp = await _config_edit(request, f"remove model {name}", lambda ed: ed.delete_endpoint(name, force=force))
    if resp.status == 200:
        await asyncio.to_thread(request.app[CORE].store.set_paused, name, False, None)
        request.app[CORE].router.paused.discard(name)
    return resp


async def a_endpoint_test(request: web.Request) -> web.Response:
    if (deny := _human(request)) is not None:
        return deny
    core = request.app[CORE]
    b = await _body(request)
    url = str(b.get("url") or "").strip()
    if not url:
        return web.json_response({"error": "a URL is required"}, status=400)
    key = str(b.get("key") or "").strip() or None
    login_of = None
    if b.get("login") == "chatgpt":
        login_of = b.get("name") if b.get("name") in core.cfg.endpoints else None
        if login_of is None:
            return web.json_response({"ok": False, "error": "save the model first, then sign in to ChatGPT; "
                                                            "the test runs with that sign-in"})
    elif key is None and b.get("name") in core.cfg.endpoints:
        key = core.cfg.endpoints[b["name"]].api_key()
    return web.json_response(await core.test_endpoint(url, b.get("dialect") or "openai", key, login_of))


# ---- ChatGPT (Codex) sign-in, people only

def _chatgpt_endpoint(request: web.Request):
    core = request.app[CORE]
    name = request.match_info["name"]
    ep = core.cfg.endpoints.get(name)
    if ep is None or ep.login != "chatgpt":
        return None, web.json_response({"error": f"{name!r} isn't a model that signs in with ChatGPT"}, status=404)
    return name, None


async def a_chatgpt_status(request: web.Request) -> web.Response:
    name, err = _chatgpt_endpoint(request)
    return err or web.json_response(request.app[CORE].chatgpt.status(name))


async def a_chatgpt_start(request: web.Request) -> web.Response:
    if (deny := _human(request)) is not None:
        return deny
    name, err = _chatgpt_endpoint(request)
    if err:
        return err
    from .chatgpt import LoginError
    try:
        p = await request.app[CORE].chatgpt.start(name)
    except LoginError as e:
        return web.json_response({"error": str(e)}, status=400)
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return web.json_response({"error": f"couldn't reach OpenAI: {type(e).__name__}"}, status=502)
    log.info("chatgpt sign-in started for %s by %s", name, request["who"].get("email"))
    return web.json_response(p)


async def a_chatgpt_import(request: web.Request) -> web.Response:
    if (deny := _human(request)) is not None:
        return deny
    name, err = _chatgpt_endpoint(request)
    if err:
        return err
    from .chatgpt import LoginError
    b = await _body(request)
    core = request.app[CORE]
    try:
        core.chatgpt.import_auth_json(name, str(b.get("auth_json") or ""))
        await core.chatgpt.headers(name)          # proves the refresh token works
    except LoginError as e:
        return web.json_response({"error": str(e)}, status=400)
    st = core.router.states.get(name)
    if st:
        st.mark_ok()
    return web.json_response(core.chatgpt.status(name))


async def a_chatgpt_logout(request: web.Request) -> web.Response:
    if (deny := _human(request)) is not None:
        return deny
    name, err = _chatgpt_endpoint(request)
    if err:
        return err
    request.app[CORE].chatgpt.logout(name)
    return web.json_response({"ok": True})


async def a_flow_create(request: web.Request) -> web.Response:
    b = await _body(request)
    fields = {k: b[k] for k in FLOW_FIELDS if k in b}
    return await _config_edit(request, f"add flow {b.get('name')}", lambda ed: ed.save_flow(
        str(b.get("name") or ""), fields, create=True))


async def a_flow_update(request: web.Request) -> web.Response:
    b = await _body(request)
    fields = {k: b[k] for k in FLOW_FIELDS if k in b}
    name = request.match_info["name"]
    return await _config_edit(request, f"edit flow {name}", lambda ed: ed.save_flow(name, fields, create=False))


async def a_flow_delete(request: web.Request) -> web.Response:
    name = request.match_info["name"]
    return await _config_edit(request, f"remove flow {name}", lambda ed: ed.delete_flow(name))


async def a_member_add(request: web.Request) -> web.Response:
    b = await _body(request)
    flow = request.match_info["name"]
    return await _config_edit(request, f"add {b.get('endpoint')} to {flow}", lambda ed: ed.add_member(
        flow, str(b.get("endpoint") or ""), str(b.get("role") or "pool")))


async def a_member_remove(request: web.Request) -> web.Response:
    flow, ep = request.match_info["name"], request.match_info["endpoint"]
    return await _config_edit(request, f"remove {ep} from {flow}", lambda ed: ed.remove_member(flow, ep))


RANGES = {"1h": (60, 60), "6h": (360, 300), "24h": (1440, 900), "7d": (10080, 7200)}


async def a_history(request: web.Request) -> web.Response:
    core = request.app[CORE]
    minutes, bucket = RANGES.get(request.query.get("range", "1h"), RANGES["1h"])
    return web.json_response(await asyncio.to_thread(core.store.history, minutes, bucket, list(core.cfg.endpoints)))


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
    r.add_post("/api/endpoints/{name}/{action:drain|undrain|pause|resume}", a_drain)
    r.add_get("/api/config", a_config)
    r.add_get("/api/history", a_history)
    r.add_post("/api/endpoints", a_endpoint_create)
    r.add_post("/api/test-endpoint", a_endpoint_test)
    r.add_get("/api/endpoints/{name}/chatgpt", a_chatgpt_status)
    r.add_post("/api/endpoints/{name}/chatgpt/start", a_chatgpt_start)
    r.add_post("/api/endpoints/{name}/chatgpt/import", a_chatgpt_import)
    r.add_delete("/api/endpoints/{name}/chatgpt", a_chatgpt_logout)
    r.add_put("/api/endpoints/{name}", a_endpoint_update)
    r.add_delete("/api/endpoints/{name}", a_endpoint_delete)
    r.add_post("/api/flows", a_flow_create)
    r.add_put("/api/flows/{name}", a_flow_update)
    r.add_delete("/api/flows/{name}", a_flow_delete)
    r.add_post("/api/flows/{name}/members", a_member_add)
    r.add_delete("/api/flows/{name}/members/{endpoint}", a_member_remove)
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
            # handler_cancellation: when a client hangs up, stop waiting on its upstream and free the lane
            runner = web.AppRunner(app, access_log=None, handler_cancellation=True)
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
