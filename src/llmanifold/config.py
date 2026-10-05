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
# a flow can set these for each of its endpoints; the endpoint's own value is the default
MEMBER_LIMITS = ("max_concurrency", "overflow_at", "first_token_timeout", "rate_limit")


class ConfigError(ValueError):
    pass


# the connection to an API: an endpoint naming a provider takes these from it and may not set them itself
CONNECTION_FIELDS = ("url", "dialect", "login", "key", "key_env", "key_file", "headers")
# also taken from the provider, unless the endpoint sets its own
SHARED_FIELDS = ("probe", "metered", "timeout", "connect_timeout", "stream_usage", "json_schema")


@dataclass
class Provider:
    """One account with a remote API (a URL plus its key or sign-in). Several endpoints can
    serve different models from it, so the key is entered and the sign-in done once."""
    name: str
    url: str
    dialect: str = "openai"
    login: str | None = None       # "chatgpt": a ChatGPT (Codex) sign-in instead of a key
    key: str | None = None
    key_env: str | None = None
    key_file: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    max_concurrency: int | None = None   # requests in flight across all its endpoints; None: no shared limit
    probe: str = "none"
    metered: bool = True
    timeout: float = 1800.0
    connect_timeout: float = 5.0
    stream_usage: bool = True
    json_schema: str | None = None


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
    rate_limit: int | None = None  # requests started per minute; over it the endpoint counts as busy
    json_schema: str | None = None # response_format json_schema: "native" passes it through; "emulate" uses JSON
                                   # mode, checks the reply and asks again if it doesn't fit. Default: emulate for DeepSeek
    headers: dict[str, str] = field(default_factory=dict)
    provider: str | None = None    # take the URL, key or sign-in (and defaults) from this provider

    @property
    def account(self) -> str:
        """Whose key or sign-in this endpoint uses: its provider's, or its own."""
        return self.provider or self.name

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
    limits: dict[str, dict] = field(default_factory=dict)   # endpoint -> {max_concurrency, overflow_at,
                                                            # first_token_timeout} for this flow only

    def limit(self, endpoint: str, name: str):
        """This flow's own setting for one of its endpoints, or None to use the endpoint's."""
        return (self.limits.get(endpoint) or {}).get(name)


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
    providers: dict[str, Provider] = field(default_factory=dict)
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


def _base_url(url: str) -> str:
    url = str(url).rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def _check_connection(c, what: str) -> None:
    if c.dialect not in DIALECTS:
        raise ConfigError(f"{what} {c.name!r}: dialect must be one of {DIALECTS}")
    if c.login is not None and c.login not in LOGINS:
        raise ConfigError(f"{what} {c.name!r}: login must be one of {LOGINS}")
    if c.login == "chatgpt" and c.dialect != "responses":
        raise ConfigError(f"{what} {c.name!r}: a ChatGPT sign-in needs dialect: responses")
    if c.probe not in PROBES:
        raise ConfigError(f"{what} {c.name!r}: probe must be one of {PROBES}")
    if c.json_schema is not None and c.json_schema not in JSON_SCHEMA_MODES:
        raise ConfigError(f"{what} {c.name!r}: json_schema must be one of {JSON_SCHEMA_MODES}")


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

    pv_keys = set(Provider.__dataclass_fields__) - {"name"}
    for name, d in (raw.get("providers") or {}).items():
        d = _take(dict(d or {}), Provider, name, pv_keys, "provider")
        if "url" not in d:
            raise ConfigError(f"provider {name!r}: url is required")
        pv = Provider(name=name, **d)
        pv.url = _base_url(pv.url)
        _check_connection(pv, "provider")
        if pv.max_concurrency is not None and pv.max_concurrency < 1:
            raise ConfigError(f"provider {name!r}: max_concurrency must be >= 1")
        cfg.providers[name] = pv

    ep_keys = set(Endpoint.__dataclass_fields__) - {"name"}
    for name, d in (raw.get("endpoints") or {}).items():
        d = _take(dict(d or {}), Endpoint, name, ep_keys, "endpoint")
        if d.get("provider") is not None:
            pv = cfg.providers.get(d["provider"])
            if pv is None:
                raise ConfigError(f"endpoint {name!r}: unknown provider {d['provider']!r}")
            own = sorted(set(d) & set(CONNECTION_FIELDS))
            if own:
                raise ConfigError(f"endpoint {name!r}: {', '.join(own)} come from provider {pv.name!r}; "
                                  "set them there")
            d = {**{f: getattr(pv, f) for f in CONNECTION_FIELDS + SHARED_FIELDS}, **d}
            d["headers"] = dict(pv.headers)
        if "url" not in d:
            raise ConfigError(f"endpoint {name!r}: url (or a provider) is required")
        ep = Endpoint(name=name, **d)
        if ep.provider is None and ep.login and name in cfg.providers:
            raise ConfigError(f"endpoint {name!r} signs in on its own but shares its name with a provider; "
                              f"rename it or give it provider: {name}")
        ep.url = _base_url(ep.url)
        _check_connection(ep, "endpoint")
        if ep.reasoning_effort is not None and ep.reasoning_effort not in REASONING_EFFORTS:
            raise ConfigError(f"endpoint {name!r}: reasoning_effort must be one of {REASONING_EFFORTS}")
        if ep.max_concurrency < 1:
            raise ConfigError(f"endpoint {name!r}: max_concurrency must be >= 1")
        if ep.rate_limit is not None and (not isinstance(ep.rate_limit, int) or ep.rate_limit < 1):
            raise ConfigError(f"endpoint {name!r}: rate_limit must be a whole number of requests per minute >= 1")
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
        m.limits = {str(k): dict(v or {}) for k, v in (m.limits or {}).items()}
        for ep, lim in m.limits.items():
            if ep not in m.pool and ep not in m.fallback:
                raise ConfigError(f"model {name!r}: limits for {ep!r}, which isn't in its pool or fallback")
            bad = set(lim) - set(MEMBER_LIMITS)
            if bad:
                raise ConfigError(f"model {name!r}: limits for {ep!r} can only set {MEMBER_LIMITS}")
            for k in ("max_concurrency", "overflow_at", "rate_limit"):
                if lim.get(k) is not None and (not isinstance(lim[k], int) or lim[k] < 1):
                    raise ConfigError(f"model {name!r}: limits.{ep}.{k} must be a whole number >= 1")
            if lim.get("first_token_timeout") is not None and not (
                    isinstance(lim["first_token_timeout"], (int, float)) and lim["first_token_timeout"] > 0):
                raise ConfigError(f"model {name!r}: limits.{ep}.first_token_timeout must be a number > 0")
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
