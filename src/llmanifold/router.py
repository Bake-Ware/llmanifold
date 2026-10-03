"""Runtime state and endpoint selection.

`Router` owns one `EndpointState` per configured endpoint and hands out
endpoints for requests. Choosing an endpoint and reserving it happen together
under one lock, so two requests arriving at the same instant can never both
claim the same free lane. Waiters queue in priority order: interactive before
background, FIFO within a class.
"""
from __future__ import annotations

import asyncio
import itertools
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field

from .config import Config, Endpoint, Model

AFFINITY_MAX = 2048
LIVE_WINDOW = 5.0    # seconds of streamed tokens behind an endpoint's live tokens-per-second figure
BUSY_FRESH = 4.0     # seconds a probed busy flag stays trustworthy
SLOW_COOLDOWN = 60.0 # a remote API that stalled is skipped this long, then gets one request at a time


@dataclass
class EndpointState:
    cfg: Endpoint
    inflight: int = 0
    bg_inflight: int = 0
    healthy: bool = True
    draining: bool = False
    fail_streak: int = 0
    last_error: str | None = None
    last_ok: float = 0.0
    last_check: float = 0.0
    probe_busy: int = 0            # busy + queued as the engine itself reports it
    probe_at: float = 0.0
    requests: int = 0
    errors: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tps_ema: float | None = None   # decode speed, tokens/s
    ttft_ema: float | None = None  # seconds
    current: dict = field(default_factory=dict)   # request id -> short description
    cooldown_until: float = 0.0    # monotonic; skipped until then (stalled remote API)
    probation: bool = False        # after a stall: one request at a time until one succeeds
    balance: list | None = None    # prepaid credit left, where the API reports it: [{"amount", "currency"}]
    live: deque = field(default_factory=deque)   # (when, tokens) streamed in the last LIVE_WINDOW seconds
    quota: list | None = None      # plan allowance used: [{"used_percent", "window_seconds", "reset_at"}]

    @property
    def capacity(self) -> int:
        return 1 if self.probation else self.cfg.max_concurrency

    def mark_slow(self, error: str) -> None:
        """A remote API accepted a request but sent nothing in time. Back off from it: skip it for a
        while, then let one request through at a time until one gets an answer. Local pool lanes are
        left alone (a long prompt can legitimately take a while, and probes track them)."""
        self.errors += 1
        self.last_error = error
        if self.cfg.fallback or self.cfg.metered:
            self.cooldown_until = time.monotonic() + SLOW_COOLDOWN
            self.probation = True

    def streamed(self, tokens: int) -> None:
        now = time.monotonic()
        self.live.append((now, tokens))
        while self.live and self.live[0][0] < now - LIVE_WINDOW:
            self.live.popleft()

    def tps_now(self) -> float:
        """Tokens per second across everything this endpoint is streaming right now."""
        cutoff = time.monotonic() - LIVE_WINDOW
        while self.live and self.live[0][0] < cutoff:
            self.live.popleft()
        return round(sum(n for _, n in self.live) / LIVE_WINDOW, 1)

    def served(self) -> None:
        """A real request got an answer."""
        self.probation = False
        self.cooldown_until = 0.0

    @property
    def name(self) -> str:
        return self.cfg.name

    @property
    def usable(self) -> bool:
        return self.healthy and not self.draining and time.monotonic() >= self.cooldown_until

    def load(self, now: float | None = None) -> int:
        now = now or time.monotonic()
        probe = self.probe_busy if (now - self.probe_at) < BUSY_FRESH else 0
        return max(self.inflight, probe)

    def observe(self, tokens_in: int, tokens_out: int, gen_seconds: float | None, ttft: float | None) -> None:
        self.requests += 1
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        if gen_seconds and gen_seconds > 0.2 and tokens_out > 4:
            tps = tokens_out / gen_seconds
            self.tps_ema = tps if self.tps_ema is None else 0.7 * self.tps_ema + 0.3 * tps
        if ttft is not None:
            self.ttft_ema = ttft if self.ttft_ema is None else 0.7 * self.ttft_ema + 0.3 * ttft

    def mark_failure(self, error: str, immediate: bool = False) -> None:
        self.errors += 1
        self.fail_streak += 1
        self.last_error = error
        if immediate or self.fail_streak >= 2:
            self.healthy = False

    def mark_ok(self) -> None:
        self.fail_streak = 0
        self.healthy = True
        self.last_ok = time.time()

    def snapshot(self) -> dict:
        return {"name": self.name, "url": self.cfg.url, "dialect": self.cfg.dialect, "model": self.cfg.model,
                "max_concurrency": self.cfg.max_concurrency, "capacity": self.capacity,
                "cooling_s": max(0, round(self.cooldown_until - time.monotonic())) or None,
                "inflight": self.inflight,
                "background_inflight": self.bg_inflight, "probe_busy": self.probe_busy,
                "load": self.load(), "healthy": self.healthy, "draining": self.draining,
                "metered": self.cfg.metered, "fallback": self.cfg.fallback, "overflow_at": self.cfg.overflow_at, "context": self.cfg.context,
                "last_error": self.last_error, "last_ok": self.last_ok or None,
                "requests": self.requests, "errors": self.errors, "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "tps": round(self.tps_ema, 1) if self.tps_ema else None, "tps_now": self.tps_now(),
                "ttft": round(self.ttft_ema, 2) if self.ttft_ema else None,
                "balance": self.balance, "quota": self.quota,
                "current": list(self.current.values())}


@dataclass(order=True)
class _Waiter:
    priority: int                     # 0 interactive, 1 background
    seq: int
    model: str = field(compare=False)
    since: float = field(compare=False, default_factory=time.monotonic)
    rid: str = field(compare=False, default="")


class Router:
    def __init__(self, cfg: Config, paused: set[str] | None = None) -> None:
        self.cfg = cfg
        self.paused: set[str] = set(paused or ())
        self.states: dict[str, EndpointState] = {n: EndpointState(e, draining=n in self.paused)
                                                 for n, e in cfg.endpoints.items()}
        self._cond = asyncio.Condition()
        self._waiting: list[_Waiter] = []
        self._seq = itertools.count()
        self._rr = itertools.count()
        self._affinity: OrderedDict[str, str] = OrderedDict()

    # ---- config reload
    def update(self, cfg: Config) -> None:
        new = {}
        for n, e in cfg.endpoints.items():
            st = self.states.get(n)
            if st is None:
                st = EndpointState(e, draining=n in self.paused)
            else:
                st.cfg = e
            new[n] = st
        self.states = new
        self.cfg = cfg
        self.wake()

    def wake(self) -> None:
        async def _notify():
            async with self._cond:
                self._cond.notify_all()
        try:
            asyncio.get_running_loop().create_task(_notify())
        except RuntimeError:
            pass

    # ---- queue view
    def queue(self) -> list[dict]:
        now = time.monotonic()
        return [{"model": w.model, "priority": "background" if w.priority else "interactive",
                 "waiting_s": round(now - w.since, 1), "id": w.rid} for w in sorted(self._waiting)]

    def queue_depth(self, model: str | None = None) -> int:
        return sum(1 for w in self._waiting if model is None or w.model == model)

    # ---- selection
    def _pool_states(self, model: Model, min_context: int | None) -> list[EndpointState]:
        out = []
        for n in model.pool:
            st = self.states.get(n)
            if st is None or not st.usable:
                continue
            if min_context and st.cfg.context and st.cfg.context < min_context:
                continue
            out.append(st)
        return out

    def pool_available(self, model: Model, min_context: int | None = None) -> bool:
        """Is there any usable endpoint in the pool at all (busy or not)?"""
        return bool(self._pool_states(model, min_context))

    def _pick(self, model: Model, key: str | None, background: bool,
              min_context: int | None) -> EndpointState | None:
        cands = self._pool_states(model, min_context)
        if not cands:
            return None
        if background and model.background_max_lanes is not None:
            if sum(s.bg_inflight for s in cands) >= model.background_max_lanes:
                return None
        now = time.monotonic()
        free = [s for s in cands if s.load(now) < s.capacity]
        if not free:
            return None
        pref = self._affinity.get(key) if (key and model.affinity) else None
        for s in free:
            if s.name == pref:
                return s
        low = min(s.load(now) for s in free)
        best = [s for s in free if s.load(now) == low]
        return best[next(self._rr) % len(best)]

    def _reserve(self, st: EndpointState, background: bool, key: str | None, rid: str, desc: str) -> None:
        st.inflight += 1
        if background:
            st.bg_inflight += 1
        st.current[rid] = desc
        if key:
            self._affinity[key] = st.name
            self._affinity.move_to_end(key)
            while len(self._affinity) > AFFINITY_MAX:
                self._affinity.popitem(last=False)

    async def acquire(self, model: Model, *, key: str | None, background: bool, timeout: float,
                      rid: str = "", desc: str = "", min_context: int | None = None,
                      overflow: list[tuple[str, int]] = ()) -> EndpointState | None:
        """Wait (up to `timeout`) for a pool endpoint and reserve it. None on timeout or if the pool is dead.

        `overflow` lists (fallback endpoint, n): while waiting, a request that is n-th or later in this
        model's queue goes to that endpoint instead, if it has room. So up to n-1 requests wait for the
        pool and the backlog beyond that is cleared by the overflow endpoint. The caller can tell an
        overflow by the endpoint not being in `model.pool`."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout)
        w = _Waiter(1 if background else 0, next(self._seq), model.name, rid=rid)
        async with self._cond:
            self._waiting.append(w)
            try:
                while True:
                    if not self.pool_available(model, min_context):
                        return None
                    ahead = [o for o in self._waiting if o.model == model.name and o < w]
                    if not ahead:
                        st = self._pick(model, key, background, min_context)
                        if st is not None:
                            self._reserve(st, background, key, rid, desc)
                            return st
                    position = len(ahead) + 1
                    for name, n in overflow:
                        ost = self.states.get(name)
                        if (position >= n and ost is not None and ost.usable
                                and ost.load() < ost.capacity
                                and not (min_context and ost.cfg.context and ost.cfg.context < min_context)):
                            self._reserve(ost, background, None, rid, desc)
                            return ost
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        return None
                    try:
                        # re-check periodically: probe-reported busy flags change without a notify
                        await asyncio.wait_for(self._cond.wait(), min(remaining, 1.0))
                    except asyncio.TimeoutError:
                        pass
            finally:
                self._waiting.remove(w)
                self._cond.notify_all()

    async def try_endpoint(self, name: str, *, background: bool, rid: str = "", desc: str = "") -> EndpointState | None:
        """Reserve a specific (fallback) endpoint if it has room right now."""
        async with self._cond:
            st = self.states.get(name)
            if st is None or not st.usable or st.load() >= st.capacity:
                return None
            self._reserve(st, background, None, rid, desc)
            return st

    async def release(self, st: EndpointState, background: bool, rid: str = "") -> None:
        async with self._cond:
            st.inflight = max(0, st.inflight - 1)
            # the engine's own busy count included this request: don't let a probe taken while it
            # ran make the lane look busy after it's free
            st.probe_busy = max(0, st.probe_busy - 1)
            if background:
                st.bg_inflight = max(0, st.bg_inflight - 1)
            st.current.pop(rid, None)
            self._cond.notify_all()

    async def set_draining(self, name: str, draining: bool) -> bool:
        async with self._cond:
            st = self.states.get(name)
            if st is None:
                return False
            st.draining = draining
            (self.paused.add if draining else self.paused.discard)(name)
            self._cond.notify_all()
            return True
