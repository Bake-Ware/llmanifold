"""Command line: run the server, check a config, poke a running instance.

    llmanifold serve  -c config.yaml
    llmanifold check  -c config.yaml
    llmanifold ctl status | endpoints | models | queue | requests [N] | drain NAME | undrain NAME | reload
    llmanifold rook-caps          # custom-cap definitions for a build-167 Rook worker
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import urllib.error
import urllib.request

from . import __version__
from .config import ConfigError, load

DEFAULT_ADMIN = os.environ.get("LLMANIFOLD_ADMIN", "http://127.0.0.1:1240")


def _admin(url: str, path: str, method: str = "GET", body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url.rstrip("/") + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        try:
            return {"ok": False, "status": e.code, **json.loads(e.read() or b"{}")}
        except ValueError:
            return {"ok": False, "status": e.code}
    except urllib.error.URLError as e:
        return {"ok": False, "error": f"llmanifold admin API not reachable at {url}: {e.reason}"}


def _summary(st: dict) -> dict:
    if st.get("ok") is False:
        return st
    return {"version": st["version"], "uptime_s": st["uptime"], "reload_error": st["reload_error"],
            "counters": st["counters"],
            "endpoints": [{k: e[k] for k in ("name", "healthy", "draining", "inflight", "max_concurrency",
                                              "tps", "requests", "errors", "metered")} for e in st["endpoints"]],
            "models": [{k: m[k] for k in ("name", "aliases", "pool", "fallback", "auth", "queued")}
                       for m in st["models"]],
            "queue": st["queue"]}


def cmd_ctl(args) -> int:
    a, url = args.action, args.admin
    if a == "status":
        out = _summary(_admin(url, "/api/status"))
    elif a == "endpoints":
        st = _admin(url, "/api/status")
        out = st if st.get("ok") is False else st["endpoints"]
    elif a == "models":
        st = _admin(url, "/api/status")
        out = st if st.get("ok") is False else st["models"]
    elif a == "queue":
        st = _admin(url, "/api/status")
        out = st if st.get("ok") is False else st["queue"]
    elif a == "requests":
        n = int(args.arg or 20)
        out = _admin(url, f"/api/requests?limit={n}")
    elif a in ("drain", "undrain"):
        if not args.arg:
            print(f"usage: llmanifold ctl {a} ENDPOINT", file=sys.stderr)
            return 2
        out = _admin(url, f"/api/endpoints/{args.arg}/{a}", "POST", {})
    elif a == "reload":
        out = _admin(url, "/api/reload", "POST", {})
    else:
        print(f"unknown action {a}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=None if args.compact else 2, default=str))
    return 0 if not (isinstance(out, dict) and out.get("ok") is False) else 1


def rook_caps(binary: str, admin: str) -> dict:
    """Definitions for Rook's `customcap.add` on build-167 workers (caps appear as cmd.<name>)."""
    base = f"{binary} ctl --compact --admin {admin}"
    return {
        "llmanifold-status": {"command": f"{base} status", "args": [],
                              "description": "llmanifold: endpoints (health, load, tok/s), models, queue, counters.",
                              "timeout": 15.0},
        "llmanifold-requests": {"command": f"{base} requests {{n}}", "args": ["n"],
                                "description": "llmanifold: last n requests (endpoint, tokens, TTFT, fallbacks).",
                                "timeout": 15.0},
        "llmanifold-drain": {"command": f"{base} drain {{endpoint}}", "args": ["endpoint"],
                             "description": "llmanifold: stop sending new requests to an endpoint (in-flight ones finish).",
                             "timeout": 15.0},
        "llmanifold-undrain": {"command": f"{base} undrain {{endpoint}}", "args": ["endpoint"],
                               "description": "llmanifold: put a drained endpoint back in service.",
                               "timeout": 15.0},
        "llmanifold-reload": {"command": f"{base} reload", "args": [],
                              "description": "llmanifold: re-read the config file now.", "timeout": 15.0},
    }


def cmd_rook_caps(args) -> int:
    binary = args.bin or shutil.which("llmanifold") or "llmanifold"
    caps = rook_caps(binary, args.admin)
    if args.format == "json":
        print(json.dumps(caps, indent=2))
    else:
        for name, c in caps.items():
            print(json.dumps({"cap": "customcap.add", "args": {"name": name, **c}}))
    return 0


def cmd_check(args) -> int:
    try:
        cfg = load(args.config)
    except (ConfigError, OSError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 1
    print(f"ok: {len(cfg.models)} models, {len(cfg.endpoints)} endpoints; api {cfg.api_listen}, admin {cfg.admin_listen}")
    for m in cfg.models.values():
        chain = " -> ".join([f"[{', '.join(m.pool)}]"] * bool(m.pool) + m.fallback)
        print(f"  {m.name}{' (' + ', '.join(m.aliases) + ')' if m.aliases else ''}: {chain}  auth={m.auth}")
    return 0


def cmd_serve(args) -> int:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        cfg = load(args.config)
    except (ConfigError, OSError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 1
    from .apps import serve
    try:
        asyncio.run(serve(cfg))
    except KeyboardInterrupt:
        pass
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="llmanifold", description="One front door for many LLM endpoints.")
    p.add_argument("--version", action="version", version=f"llmanifold {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the server")
    s.add_argument("-c", "--config", default=os.environ.get("LLMANIFOLD_CONFIG", "config.yaml"))
    s.add_argument("--log-level", default="info")
    s.set_defaults(func=cmd_serve)
    c = sub.add_parser("check", help="validate a config file")
    c.add_argument("-c", "--config", default=os.environ.get("LLMANIFOLD_CONFIG", "config.yaml"))
    c.set_defaults(func=cmd_check)
    t = sub.add_parser("ctl", help="talk to a running instance's admin API")
    t.add_argument("action", choices=["status", "endpoints", "models", "queue", "requests", "drain", "undrain",
                                      "reload"])
    t.add_argument("arg", nargs="?")
    t.add_argument("--admin", default=DEFAULT_ADMIN)
    t.add_argument("--compact", action="store_true")
    t.set_defaults(func=cmd_ctl)
    r = sub.add_parser("rook-caps", help="print Rook customcap definitions (build-167 workers)")
    r.add_argument("--admin", default=DEFAULT_ADMIN)
    r.add_argument("--bin", help="path to the llmanifold executable on the worker")
    r.add_argument("--format", choices=["json", "calls"], default="json")
    r.set_defaults(func=cmd_rook_caps)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
