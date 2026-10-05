"""Configuration: one YAML file, validated into plain dataclasses.

Everything that can change at runtime without a restart (endpoints, models,
rules) lives here and is re-read by `ConfigWatcher` when the file changes.
Listen addresses and the data directory are read once at start.
"""
from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DIALECTS = ("openai", "anthropic", "responses")   # responses: OpenAI Responses API (upstream only)
LOGINS = ("chatgpt",)
PROBES = ("none", "models", "strata", "llamacpp")
TRIGGERS = ("connect", "timeout", "slow", "5xx", "429", "4xx", "context", "empty", "schema", "queue")
DEFAULT_TRIGGERS = ("connect", "timeout", "slow", "5xx", "429", "context", "empty", "schema", "queue")
JSON_SCHEMA_MODES = ("native", "emulate")
LEGACY_ADMIN_KEYS = ("trusted_proxies", "human_header", "allowed_emails", "local_humans")
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


class ConfigError(ValueError):
    pass


@dataclass
class Endpoint:
    name: str
    url: str                       # base URL, without /v1
    dialect: str = "openai"        # API the endpoint speaks
    model: str | None = None       # model name to send upstream (None: keep the client's)
    key: str | None = None         # literal key (avoid; prefer key_env / key_file)
    key_env: str | None = None
    key_file: str | None = None
    max_concurrency: int = 1       # requests in flight at once (local engines: 1 per lane)
    context: int | None = None     # context window, for context-fit routing
    probe: str = "models"          # how to check health / busy state
    metered: bool = False          # costs money per request (remote API)
    fallback: bool = False         # only used as a fallback, never as a first choice
    overflow_at: int | None = None # as a flow's fallback: take requests once this many are waiting for a lane
    login: str | None = None       # "chatgpt": authenticate with a ChatGPT (Codex) sign-in instead of a key
    first_token_timeout: float | None = None  # give up (trigger "slow") if nothing arrives in this long; default: the flow's
    timeout: float = 1800.0        # whole-request read timeout (s)
    connect_timeout: float = 5.0
    stream_usage: bool = True      # ask OpenAI-style engines for token counts on streams
    reasoning_effort: str | None = None  # sent when the client names none: none | minimal | low | medium | high | ...
    service_tier: str | None = None      # sent upstream as service_tier (e.g. "priority" on the Codex backend)
    json_schema: str | None = None # response_format json_schema: "native" passes it through; "emulate" uses JSON
                                   # mode, checks the reply and asks again if it doesn't fit. Default: emulate for DeepSeek
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def is_deepseek(self) -> bool:
        host = self.url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0].lower()
        return host == "deepseek.com" or host.endswith(".deepseek.com")

    @property
    def emulates_json_schema(self) -> bool:
        if self.json_schema is not None:
            return self.json_schema == "emulate"
        return self.is_deepseek

    def api_key(self) -> str | None:
        if self.key:
            return self.key
        if self.key_env and os.environ.get(self.key_env):
            return os.environ[self.key_env]
        if self.key_file:
            try:
                return Path(self.key_file).expanduser().read_text().strip() or None
            except OSError:
                return None
        return None


@dataclass
class Model:
    name: str
    pool: list[str]                                  # first-choice endpoints, shared by load
    aliases: list[str] = field(default_factory=list)
    fallback: list[str] = field(default_factory=list)  # tried in order when the pool fails
    fallback_on: tuple[str, ...] = DEFAULT_TRIGGERS
    auth: str = "open"                               # open | token (default; the admin UI can override)
    allow_metered_unauthenticated: bool = False
    queue_timeout: float = 120.0                     # max wait for a free pool endpoint
    first_token_timeout: float = 300.0               # held stream: max wait for first content
    background_max_lanes: int | None = None          # background requests may hold at most N pool lanes
    affinity: bool = True                            # reuse the endpoint that last served a conversation
    context: int | None = None                       # advertised context (default: smallest in pool)
    input_modalities: list[str] | None = None        # advertised in /v1/models, e.g. [text, image]
    description: str = ""


@dataclass
class Webhook:
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    events: tuple[str, ...] = ("metered",)


@dataclass
class Admin:
    # Who may use the admin listener. Anyone allowed here has full control.
    allow_from: list[str] = field(default_factory=lambda: ["127.0.0.0/8", "::1/128"])

    def _nets(self, items: list[str]):
        out = []
        for item in items:
            try:
                out.append(ipaddress.ip_network(item, strict=False))
            except ValueError as e:
                raise ConfigError(f"admin: bad network {item!r}: {e}") from None
        return out

    def ip_in(self, ip: str, items: list[str]) -> bool:
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return False
        return any(addr in net for net in self._nets(items))


@dataclass
class Config:
    api_listen: str = "0.0.0.0:1234"
    admin_listen: str = "127.0.0.1:1240"
    data_dir: str = "./data"
    history_days: int = 30
    keepalive_after: float = 15.0        # start sending keepalives after this long without bytes
    health_interval: float = 10.0
    endpoints: dict[str, Endpoint] = field(default_factory=dict)
    models: dict[str, Model] = field(default_factory=dict)
    webhooks: list[Webhook] = field(default_factory=list)
    admin: Admin = field(default_factory=Admin)
    priority_header: str = "X-LLManifold-Priority"
    default_model: str | None = None     # used when a request names no model, or one we don't know
    path: str | None = None

    # alias or model name -> Model
    def resolve(self, name: str | None, use_default: bool = False) -> Model | None:
        if name:
            if name in self.models:
                return self.models[name]
            for m in self.models.values():
                if name in m.aliases:
                    return m
        if use_default and self.default_model:
            return self.resolve(self.default_model)
        return None


def _hostport(value: str, what: str) -> str:
    host, sep, port = str(value).rpartition(":")
    if not sep or not port.isdigit():
        raise ConfigError(f"{what}: expected host:port, got {value!r}")
    return value


def _take(d: dict, cls, name: str, known: set[str], what: str) -> dict:
    extra = set(d) - known
    if extra:
        raise ConfigError(f"{what} {name!r}: unknown keys {sorted(extra)}")
    return d


def parse(raw: dict[str, Any], path: str | None = None) -> Config:
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    cfg = Config(path=path)
    listen = raw.get("listen") or {}
    cfg.api_listen = _hostport(listen.get("api", cfg.api_listen), "listen.api")
    cfg.admin_listen = _hostport(listen.get("admin", cfg.admin_listen), "listen.admin")
    cfg.data_dir = str(raw.get("data_dir", cfg.data_dir))
    cfg.history_days = int(raw.get("history_days", cfg.history_days))
    cfg.keepalive_after = float(raw.get("keepalive_after", cfg.keepalive_after))
    cfg.health_interval = float(raw.get("health_interval", cfg.health_interval))
    cfg.priority_header = str(raw.get("priority_header", cfg.priority_header))
    cfg.default_model = raw.get("default_model") or None

    ep_keys = set(Endpoint.__dataclass_fields__) - {"name"}
    for name, d in (raw.get("endpoints") or {}).items():
        d = _take(dict(d or {}), Endpoint, name, ep_keys, "endpoint")
        if "url" not in d:
            raise ConfigError(f"endpoint {name!r}: url is required")
        ep = Endpoint(name=name, **d)
        ep.url = ep.url.rstrip("/")
        if ep.url.endswith("/v1"):
            ep.url = ep.url[:-3]
        if ep.dialect not in DIALECTS:
            raise ConfigError(f"endpoint {name!r}: dialect must be one of {DIALECTS}")
        if ep.login is not None and ep.login not in LOGINS:
            raise ConfigError(f"endpoint {name!r}: login must be one of {LOGINS}")
        if ep.login == "chatgpt" and ep.dialect != "responses":
            raise ConfigError(f"endpoint {name!r}: a ChatGPT sign-in needs dialect: responses")
        if ep.probe not in PROBES:
            raise ConfigError(f"endpoint {name!r}: probe must be one of {PROBES}")
        if ep.reasoning_effort is not None and ep.reasoning_effort not in REASONING_EFFORTS:
            raise ConfigError(f"endpoint {name!r}: reasoning_effort must be one of {REASONING_EFFORTS}")
        if ep.json_schema is not None and ep.json_schema not in JSON_SCHEMA_MODES:
            raise ConfigError(f"endpoint {name!r}: json_schema must be one of {JSON_SCHEMA_MODES}")
        if ep.max_concurrency < 1:
            raise ConfigError(f"endpoint {name!r}: max_concurrency must be >= 1")
        if ep.overflow_at is not None and (not isinstance(ep.overflow_at, int) or ep.overflow_at < 1):
            raise ConfigError(f"endpoint {name!r}: overflow_at must be a whole number >= 1")
        cfg.endpoints[name] = ep

    model_keys = set(Model.__dataclass_fields__) - {"name"}
    seen: dict[str, str] = {}
    for name, d in (raw.get("models") or {}).items():
        d = _take(dict(d or {}), Model, name, model_keys, "model")
        d.setdefault("pool", [])
        if isinstance(d.get("fallback_on"), (list, tuple)):
            # YAML reads a bare 429 as an int
            d["fallback_on"] = tuple(str(t) for t in d["fallback_on"])
        m = Model(name=name, **d)
        for t in m.fallback_on:
            if t not in TRIGGERS:
                raise ConfigError(f"model {name!r}: unknown fallback trigger {t!r} (known: {TRIGGERS})")
        if m.auth not in ("open", "token"):
            raise ConfigError(f"model {name!r}: auth must be open or token")
        if not m.pool and not m.fallback:
            raise ConfigError(f"model {name!r}: needs a pool or a fallback")
        for ep in [*m.pool, *m.fallback]:
            if ep not in cfg.endpoints:
                raise ConfigError(f"model {name!r}: unknown endpoint {ep!r}")
        for ep in m.pool:
            if cfg.endpoints[ep].fallback:
                raise ConfigError(f"model {name!r}: endpoint {ep!r} is fallback-only and can't be in a pool")
        for n in [name, *m.aliases]:
            if n in seen:
                raise ConfigError(f"model name/alias {n!r} used by both {seen[n]!r} and {name!r}")
            seen[n] = name
        if m.context is None:
            ctxs = [cfg.endpoints[e].context for e in m.pool if cfg.endpoints[e].context]
            m.context = min(ctxs) if ctxs else None
        cfg.models[name] = m

    if cfg.default_model and cfg.resolve(cfg.default_model) is None:
        raise ConfigError(f"default_model {cfg.default_model!r} is not a model or alias")

    for w in raw.get("webhooks") or []:
        if "url" not in w:
            raise ConfigError("webhook: url is required")
        cfg.webhooks.append(Webhook(url=w["url"], headers=dict(w.get("headers") or {}),
                                    events=tuple(w.get("events") or ("metered",))))

    a = dict(raw.get("admin") or {})
    for old in LEGACY_ADMIN_KEYS:     # the auth-proxy settings are gone; older configs still load
        a.pop(old, None)
    admin_keys = set(Admin.__dataclass_fields__)
    _take(dict(a), Admin, "admin", admin_keys, "section")
    cfg.admin = Admin(**a)
    cfg.admin._nets(cfg.admin.allow_from)
    return cfg


def load(path: str | os.PathLike) -> Config:
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{p}: {e}") from None
    return parse(raw, str(p))
