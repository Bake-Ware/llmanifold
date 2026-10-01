"""llmanifold.* — see and steer the llmanifold router on this worker.

Loaded by Rook workers that use the plugin API (core API 1.1+) through the
``rook.plugins`` entry point. It talks only to llmanifold's admin listener,
over loopback by default, so it can do what a local operator can: read status
and history, drain and resume endpoints, reload the config. It cannot create
tokens or change which models need one; those stay with a person signed in
through the admin site.

    llmanifold.status()                -> models, endpoints, queue, counters
    llmanifold.endpoints()             -> one row per endpoint (load, health, speed)
    llmanifold.queue()                 -> requests waiting for a lane
    llmanifold.requests(limit=20)      -> recent requests, newest first
    llmanifold.drain(endpoint)         -> stop sending new requests to an endpoint
    llmanifold.undrain(endpoint)       -> put it back in rotation
    llmanifold.reload()                -> re-read the config file now

This module is stdlib-only, like the worker it loads into.
"""
from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request

try:  # inside a Rook worker
    from rook.core.plugin import Plugin, capability, setting
except ImportError:  # imported outside Rook (tests, docs): a minimal stand-in
    class Plugin:  # type: ignore[no-redef]
        NAMESPACE = ""
        SETTINGS: tuple = ()

        def __init__(self) -> None:
            self._values: dict = {}

        @property
        def settings(self):
            outer = self

            class _View:
                def get(self, name, default=None):
                    for s in outer.SETTINGS:
                        if s["name"] == name:
                            env = os.environ.get(s["env"]) if s.get("env") else None
                            v = outer._values.get(name, env if env is not None else s["default"])
                            return default if v is None else v
                    return default
            return _View()

    def capability(suffix: str = "", **_kw):  # type: ignore[no-redef]
        def deco(fn):
            fn._rook_cap_suffix = suffix
            return fn
        return deco

    def setting(name, type=str, default=None, **kw):  # type: ignore[no-redef]
        return {"name": name, "default": default, "env": kw.get("env")}


DEFAULT_ADMIN = "http://127.0.0.1:1240"


class LLManifold(Plugin):
    NAMESPACE = "llmanifold"
    NAME = "llmanifold"
    VERSION = "1.tidy.manifold"
    CORE_API = ">=1.1,<2"
    SETTINGS = (
        setting("admin_url", "url", DEFAULT_ADMIN, scope="worker", env="LLMANIFOLD_ADMIN",
                label="Admin URL", help="llmanifold's admin listener on this host."),
        setting("timeout", "float", 5.0, scope="worker", label="Request timeout (s)", min=0.5, max=60.0),
    )
    GUIDANCE = {
        "llmanifold": "llmanifold routes model requests on this host. Read llmanifold.status before "
                      "draining anything; drain is reversible with llmanifold.undrain. Tokens and "
                      "per-model token requirements are set by a person in the admin site, not here.",
    }
    SKILL = ("## llmanifold\n"
             "`llmanifold.status` shows models, endpoints and the queue on hosts that run llmanifold. "
             "`llmanifold.requests(limit)` lists recent requests. `llmanifold.drain(endpoint)` and "
             "`llmanifold.undrain(endpoint)` take an endpoint out of rotation and back; in-flight "
             "requests finish. `llmanifold.reload` re-reads the config.")
    PANEL = {"title": "llmanifold", "path": "panel.html"}

    # ---- plumbing
    def _admin(self) -> str:
        return str(self.settings.get("admin_url", DEFAULT_ADMIN)).rstrip("/")

    def _http(self, method: str, path: str) -> dict | list:
        req = urllib.request.Request(self._admin() + path, method=method,
                                     data=b"" if method == "POST" else None,
                                     headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=float(self.settings.get("timeout", 5.0))) as r:
                return json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read() or b"{}")
            except ValueError:
                body = {}
            raise RuntimeError(body.get("error") or f"llmanifold admin returned HTTP {e.code}") from None
        except (urllib.error.URLError, OSError) as e:
            reason = getattr(e, "reason", e)
            raise RuntimeError(f"can't reach llmanifold admin at {self._admin()}: {reason}") from None

    async def _get(self, path: str):
        return await asyncio.to_thread(self._http, "GET", path)

    async def _post(self, path: str):
        return await asyncio.to_thread(self._http, "POST", path)

    # ---- lifecycle
    def available(self) -> bool:
        """Load only where an llmanifold admin listener answers."""
        try:
            self._http("GET", "/api/whoami")
            return True
        except Exception:
            return False

    def heartbeat(self) -> dict | None:
        return None   # status is a call away; keep announces small

    # ---- caps
    @capability("status", risk="read", fields="*")
    async def status(self) -> dict:
        """Models, endpoints, queue and counters, compacted for reading."""
        s = await self._get("/api/status")
        return {
            "version": s.get("version"), "uptime_s": s.get("uptime"), "reload_error": s.get("reload_error"),
            "counters": s.get("counters"),
            "models": [{"name": m["name"], "aliases": m["aliases"], "pool": m["pool"],
                        "fallback": m["fallback"], "auth": m["auth"], "queued": m["queued"],
                        **({"warning": m["warning"]} if m.get("warning") else {})} for m in s.get("models", [])],
            "endpoints": [_endpoint_row(e) for e in s.get("endpoints", [])],
            "queue": s.get("queue", []),
        }

    @capability("endpoints", risk="read", fields="*")
    async def endpoints(self) -> list:
        """One row per endpoint: state, load, speed and what it's doing."""
        s = await self._get("/api/status")
        return [_endpoint_row(e) for e in s.get("endpoints", [])]

    @capability("queue", risk="read")
    async def queue(self) -> list:
        """Requests waiting for a free lane, interactive first."""
        s = await self._get("/api/status")
        return s.get("queue", [])

    @capability("requests", risk="read", limit=20)
    async def requests(self, limit: int = 20, model: str | None = None, errors_only: bool = False) -> list:
        """Recent requests, newest first. Filter by model (or alias) and to failures only."""
        rows = await self._get(f"/api/requests?limit={max(1, min(int(limit) * 5, 1000))}")
        out = []
        for r in rows:
            if model and model not in (r.get("model"), r.get("alias")):
                continue
            if errors_only and r.get("ok"):
                continue
            out.append(r)
            if len(out) >= int(limit):
                break
        return out

    @capability("drain", risk="write")
    async def drain(self, endpoint: str) -> dict:
        """Stop routing new requests to an endpoint. In-flight requests finish. Reversible."""
        return await self._post(f"/api/endpoints/{_path(endpoint)}/drain")

    @capability("undrain", risk="write")
    async def undrain(self, endpoint: str) -> dict:
        """Put a drained endpoint back in rotation."""
        return await self._post(f"/api/endpoints/{_path(endpoint)}/undrain")

    @capability("reload", risk="admin")
    async def reload(self) -> dict:
        """Re-read the config file now (it is also picked up automatically on change)."""
        return await self._post("/api/reload")


def _path(name: str) -> str:
    from urllib.parse import quote
    return quote(str(name), safe="")


def _endpoint_row(e: dict) -> dict:
    if e.get("draining"):
        state = "draining"
    elif not e.get("healthy"):
        state = "down"
    elif e.get("load"):
        state = "busy"
    else:
        state = "idle"
    row = {"name": e["name"], "state": state, "load": e.get("load"), "max": e.get("max_concurrency"),
           "requests": e.get("requests"), "errors": e.get("errors"), "tps": e.get("tps"),
           "ttft_s": e.get("ttft"), "current": e.get("current") or []}
    if e.get("metered"):
        row["metered"] = True
    if e.get("fallback"):
        row["fallback"] = True
    if e.get("last_error") and state == "down":
        row["last_error"] = e["last_error"]
    return row


PLUGIN = LLManifold
