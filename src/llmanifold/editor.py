"""Edits to the config file made from the admin site.

The YAML file stays the single source of truth: people can still edit it by
hand, and every change made in the admin site is written back to it. Edits go
through ruamel.yaml's round-trip mode, so comments, key order and list style
survive. Each write is validated with the normal config parser before it
replaces the file, and the previous version is kept as `<file>.bak-<time>`.

API keys typed into the admin site never go into the YAML: they are written to
`<data_dir>/keys/<endpoint>.key` (mode 600) and the endpoint gets `key_file`.
"""
from __future__ import annotations

import asyncio
import io
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import yaml
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.tokens import CommentToken

from .config import DIALECTS, PROBES, Config, ConfigError, parse

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,99}$")
KEEP_BACKUPS = 20

# fields the admin site may set, in the order they're written for a new entry
ENDPOINT_FIELDS = ("url", "dialect", "login", "model", "max_concurrency", "context", "probe", "metered", "fallback",
                   "overflow_at", "first_token_timeout", "timeout", "connect_timeout", "reasoning_effort",
                   "service_tier")
FLOW_FIELDS = ("aliases", "pool", "fallback", "fallback_on", "auth", "allow_metered_unauthenticated",
               "queue_timeout", "first_token_timeout", "background_max_lanes", "affinity", "context",
               "input_modalities", "description")
LIST_FIELDS = {"aliases", "pool", "fallback", "fallback_on", "input_modalities"}


class EditError(ValueError):
    """A change that can't be made; the message is shown to the person who asked."""


def _flow_seq(items) -> CommentedSeq:
    s = CommentedSeq(list(items))
    s.fa.set_flow_style()
    return s


def _section(doc: CommentedMap, key: str) -> CommentedMap:
    sec = doc.get(key)
    if not isinstance(sec, dict):
        sec = CommentedMap()
        doc[key] = sec
    return sec


def _check_name(name: str, what: str) -> str:
    name = str(name or "").strip()
    if not NAME_RE.match(name):
        raise EditError(f"{what} name {name!r} isn't allowed: use letters, digits and . _ - : (no spaces or slashes)")
    return name


def _clean(value: Any, field: str, lists: set[str]) -> Any:
    if isinstance(value, str):
        value = value.strip()
    if field in lists:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",")]
        return [str(v).strip() for v in value if str(v).strip()]
    return value


def _deepest_last(m: CommentedMap) -> tuple[CommentedMap, Any]:
    """The map and key that end `m` in the file (descending into a trailing nested map)."""
    node = m
    while True:
        last = list(node.keys())[-1]
        v = node[last]
        if isinstance(v, CommentedMap) and len(v):
            node = v
            continue
        return node, last


def _put(m: CommentedMap, key: str, value: Any) -> None:
    """Set a key. A key appended to a map takes over the blank lines that followed the old
    end of that map, so the gaps between sections stay where they were; end-of-line comments
    stay on their own keys."""
    if key in m or not len(m):
        m[key] = value
        return
    owner, last = _deepest_last(m)
    tok = owner.ca.items.get(last)
    m[key] = value
    if not (tok and len(tok) > 2 and tok[2] is not None):
        return
    first, nl, rest = tok[2].value.partition("\n")
    if not rest:
        return
    tok[2].value = first + nl
    new_owner, new_last = (_deepest_last(value) if isinstance(value, CommentedMap) and len(value)
                           else (m, key))
    new_owner.ca.items[new_last] = [None, None, CommentToken("\n" + rest, tok[2].start_mark, None), None]


def _apply_fields(target: CommentedMap, fields: dict, allowed: tuple[str, ...], lists: set[str] = frozenset()) -> None:
    unknown = set(fields) - set(allowed)
    if unknown:
        raise EditError(f"can't set {sorted(unknown)} here")
    for f in allowed:
        if f not in fields:
            continue
        v = _clean(fields[f], f, lists)
        if v is None or v == "" or (f in lists and not v and f not in ("pool", "fallback")):
            target.pop(f, None)
        elif f in lists:
            _put(target, f, _flow_seq(v))
        else:
            _put(target, f, v)


def _flows_using(doc: CommentedMap, endpoint: str) -> list[str]:
    return [name for name, m in (doc.get("models") or {}).items()
            if endpoint in list((m or {}).get("pool") or []) + list((m or {}).get("fallback") or [])]


class ConfigEditor:
    def __init__(self, path: str | os.PathLike, data_dir: str | os.PathLike) -> None:
        self.path = Path(path)
        self.keys_dir = Path(data_dir).expanduser().resolve() / "keys"
        self._lock = asyncio.Lock()

    @property
    def writable(self) -> bool:
        return os.access(self.path, os.W_OK) and os.access(self.path.parent, os.W_OK)

    @staticmethod
    def _yaml() -> YAML:
        y = YAML()
        y.preserve_quotes = True
        y.width = 4096
        y.indent(mapping=2, sequence=4, offset=2)
        return y

    # ---------------------------------------------------------------- core write path
    async def edit(self, fn: Callable[[CommentedMap], None]) -> Config:
        async with self._lock:
            return await asyncio.to_thread(self._edit, fn)

    def _edit(self, fn: Callable[[CommentedMap], None]) -> Config:
        if not self.writable:
            raise EditError(f"the config file {self.path} isn't writable by llmanifold")
        y = self._yaml()
        old = self.path.read_text()
        doc = y.load(old) or CommentedMap()
        fn(doc)
        buf = io.StringIO()
        y.dump(doc, buf)
        new = buf.getvalue()
        try:
            cfg = parse(yaml.safe_load(new) or {}, str(self.path))
        except ConfigError as e:
            raise EditError(str(e)) from None
        backup = self.path.with_name(f"{self.path.name}.bak-{time.time() * 1000:.0f}")
        backup.write_text(old)
        shutil.copymode(self.path, backup)
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        tmp.write_text(new)
        shutil.copymode(self.path, tmp)
        os.replace(tmp, self.path)
        for stale in sorted(self.path.parent.glob(f"{self.path.name}.bak-*"))[:-KEEP_BACKUPS]:
            stale.unlink(missing_ok=True)
        return cfg

    # ---------------------------------------------------------------- keys
    def _key_path(self, endpoint: str) -> Path:
        return self.keys_dir / (re.sub(r"[^A-Za-z0-9._-]", "_", endpoint) + ".key")

    def _write_key(self, endpoint: str, key: str) -> Path:
        self.keys_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.keys_dir, 0o700)
        p = self._key_path(endpoint)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(key.strip() + "\n")
        os.chmod(p, 0o600)
        return p

    # ---------------------------------------------------------------- endpoints ("models" in the UI)
    async def save_endpoint(self, name: str, fields: dict, *, create: bool, key: str | None = None,
                            clear_key: bool = False) -> Config:
        name = _check_name(name, "model")
        fields = dict(fields)
        if create and not str(fields.get("url") or "").strip():
            raise EditError("a URL is required")
        if "dialect" in fields and fields["dialect"] not in DIALECTS:
            raise EditError(f"API type must be one of {DIALECTS}")
        if fields.get("probe") not in (None, "", *PROBES):
            raise EditError(f"health check must be one of {PROBES}")
        key_file = self._write_key(name, key) if key else None

        def fn(doc: CommentedMap) -> None:
            eps = _section(doc, "endpoints")
            if create and name in eps:
                raise EditError(f"there's already a model called {name!r}")
            if not create and name not in eps:
                raise EditError(f"no model called {name!r}")
            ep = eps.get(name) if not create else CommentedMap()
            if ep is None:
                ep = CommentedMap()
            _apply_fields(ep, fields, ENDPOINT_FIELDS)
            if key_file or clear_key:
                for k in ("key", "key_env", "key_file"):
                    ep.pop(k, None)
                if key_file:
                    _put(ep, "key_file", str(key_file))
            if create:
                _put(eps, name, ep)

        cfg = await self.edit(fn)
        if clear_key:
            self._key_path(name).unlink(missing_ok=True)
        return cfg

    async def delete_endpoint(self, name: str, *, force: bool = False) -> Config:
        def fn(doc: CommentedMap) -> None:
            eps = _section(doc, "endpoints")
            if name not in eps:
                raise EditError(f"no model called {name!r}")
            using = _flows_using(doc, name)
            if using and not force:
                raise EditError(f"{name} is used by {', '.join(using)}; remove it from those flows first")
            for flow in using:
                m = doc["models"][flow]
                for role in ("pool", "fallback"):
                    if name in list(m.get(role) or []):
                        m[role] = _flow_seq([x for x in m[role] if x != name])
                if not list(m.get("pool") or []) and not list(m.get("fallback") or []):
                    raise EditError(f"removing {name} would leave {flow} with nothing to route to; "
                                    "delete that flow first")
            del eps[name]

        cfg = await self.edit(fn)
        self._key_path(name).unlink(missing_ok=True)
        return cfg

    # ---------------------------------------------------------------- flows ("models" in the config)
    async def save_flow(self, name: str, fields: dict, *, create: bool, rename: str | None = None,
                        default: bool | None = None) -> Config:
        """Create or edit a flow. `rename` gives it a new name (default_model follows it);
        `default` True makes it the default flow, False stops it being the default."""
        name = _check_name(name, "alias flow")
        if rename is not None:
            rename = _check_name(rename, "alias flow")

        def fn(doc: CommentedMap) -> None:
            flows = _section(doc, "models")
            if create and name in flows:
                raise EditError(f"there's already a flow called {name!r}")
            if not create and name not in flows:
                raise EditError(f"no flow called {name!r}")
            m = flows.get(name) if not create else CommentedMap()
            if m is None:
                m = CommentedMap()
                flows[name] = m
            _apply_fields(m, fields, FLOW_FIELDS, LIST_FIELDS)
            if create and "pool" not in m:
                m["pool"] = _flow_seq([])
            if create:
                _put(flows, name, m)
            final = name
            if rename and rename != name:
                if rename in flows:
                    raise EditError(f"there's already a flow called {rename!r}")
                pos = list(flows).index(name)
                del flows[name]
                flows.insert(pos, rename, m)
                if doc.get("default_model") == name:
                    doc["default_model"] = rename
                final = rename
            if default:
                _put(doc, "default_model", final)
            elif default is False and "default_model" in doc:
                current = doc["default_model"]
                if current == final or current in list(m.get("aliases") or []):
                    del doc["default_model"]

        return await self.edit(fn)

    async def delete_flow(self, name: str) -> Config:
        def fn(doc: CommentedMap) -> None:
            flows = _section(doc, "models")
            if name not in flows:
                raise EditError(f"no flow called {name!r}")
            default = doc.get("default_model")
            if default and (default == name or default in list((flows[name] or {}).get("aliases") or [])):
                raise EditError(f"{name} is the default flow (default_model); point default_model elsewhere first")
            del flows[name]

        return await self.edit(fn)

    async def add_member(self, flow: str, endpoint: str, role: str) -> Config:
        if role not in ("pool", "fallback"):
            raise EditError("role must be pool or fallback")

        def fn(doc: CommentedMap) -> None:
            flows = _section(doc, "models")
            if flow not in flows:
                raise EditError(f"no flow called {flow!r}")
            if endpoint not in (doc.get("endpoints") or {}):
                raise EditError(f"no model called {endpoint!r}")
            m = flows[flow]
            for r in ("pool", "fallback"):
                if endpoint in list(m.get(r) or []):
                    raise EditError(f"{endpoint} is already in {flow}")
            _put(m, role, _flow_seq(list(m.get(role) or []) + [endpoint]))

        return await self.edit(fn)

    async def remove_member(self, flow: str, endpoint: str) -> Config:
        def fn(doc: CommentedMap) -> None:
            flows = _section(doc, "models")
            if flow not in flows:
                raise EditError(f"no flow called {flow!r}")
            m = flows[flow]
            found = False
            for r in ("pool", "fallback"):
                items = list(m.get(r) or [])
                if endpoint in items:
                    m[r] = _flow_seq([x for x in items if x != endpoint])
                    found = True
            if not found:
                raise EditError(f"{endpoint} isn't in {flow}")
            if not list(m.get("pool") or []) and not list(m.get("fallback") or []):
                raise EditError(f"{endpoint} is the last model in {flow}; delete the flow instead")

        return await self.edit(fn)
