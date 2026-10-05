"""SQLite store: client tokens, per-model auth overrides, request history.

Writes are small and infrequent enough to run in a worker thread per call.
Tokens are stored hashed; the plaintext is shown once, at creation.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS tokens (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  label TEXT NOT NULL,
  hash TEXT NOT NULL UNIQUE,
  prefix TEXT NOT NULL,
  models TEXT NOT NULL DEFAULT '["*"]',
  background INTEGER NOT NULL DEFAULT 0,
  created REAL NOT NULL,
  created_by TEXT,
  last_used REAL,
  uses INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS model_auth (
  model TEXT PRIMARY KEY,
  mode TEXT NOT NULL,
  updated REAL NOT NULL,
  updated_by TEXT
);
CREATE TABLE IF NOT EXISTS requests (
  id TEXT PRIMARY KEY,
  ts REAL NOT NULL,
  model TEXT, endpoint TEXT, client TEXT, priority TEXT,
  dialect_in TEXT, dialect_out TEXT, stream INTEGER,
  status INTEGER, ok INTEGER, attempts TEXT, fallback INTEGER, metered INTEGER,
  tokens_in INTEGER, tokens_out INTEGER,
  queued_ms INTEGER, ttft_ms INTEGER, duration_ms INTEGER, tps REAL,
  error TEXT
);
CREATE INDEX IF NOT EXISTS requests_ts ON requests(ts);
CREATE TABLE IF NOT EXISTS paused (
  endpoint TEXT PRIMARY KEY,
  since REAL NOT NULL,
  paused_by TEXT
);
"""

TOKEN_PREFIX = "llm_"


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._db.commit()
        self._auth_cache: dict[str, str] = {}
        self._token_cache: dict[str, dict | None] = {}
        self._load_auth()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _x(self, sql: str, args=(), many=False):
        with self._lock:
            cur = self._db.executemany(sql, args) if many else self._db.execute(sql, args)
            self._db.commit()
            return cur

    def _q(self, sql: str, args=()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    # ---- model auth overrides (set by a human in the admin UI)
    def _load_auth(self) -> None:
        self._auth_cache = {r["model"]: r["mode"] for r in self._q("SELECT model, mode FROM model_auth")}

    def auth_override(self, model: str) -> str | None:
        return self._auth_cache.get(model)

    def set_auth(self, model: str, mode: str | None, by: str | None) -> None:
        if mode is None:
            self._x("DELETE FROM model_auth WHERE model=?", (model,))
            self._auth_cache.pop(model, None)
        else:
            self._x("INSERT INTO model_auth(model, mode, updated, updated_by) VALUES(?,?,?,?) "
                    "ON CONFLICT(model) DO UPDATE SET mode=excluded.mode, updated=excluded.updated, "
                    "updated_by=excluded.updated_by", (model, mode, time.time(), by))
            self._auth_cache[model] = mode

    def rename_model(self, old: str, new: str) -> None:
        """Carry a flow's auth override and token scopes over to its new name."""
        self._x("UPDATE model_auth SET model=? WHERE model=?", (new, old))
        if old in self._auth_cache:
            self._auth_cache[new] = self._auth_cache.pop(old)
        for r in self._q("SELECT id, models FROM tokens"):
            models = json.loads(r["models"])
            if old in models:
                self._x("UPDATE tokens SET models=? WHERE id=?",
                        (json.dumps([new if m == old else m for m in models]), r["id"]))
        self._token_cache.clear()

    # ---- paused endpoints (survive restarts)
    def paused(self) -> dict[str, dict]:
        return {r["endpoint"]: r for r in self._q("SELECT endpoint, since, paused_by FROM paused")}

    def set_paused(self, endpoint: str, paused: bool, by: str | None) -> None:
        if paused:
            self._x("INSERT INTO paused(endpoint, since, paused_by) VALUES(?,?,?) "
                    "ON CONFLICT(endpoint) DO NOTHING", (endpoint, time.time(), by))
        else:
            self._x("DELETE FROM paused WHERE endpoint=?", (endpoint,))

    # ---- tokens
    def create_token(self, label: str, models: list[str], background: bool, by: str | None) -> dict:
        token = TOKEN_PREFIX + secrets.token_urlsafe(24)
        cur = self._x("INSERT INTO tokens(label, hash, prefix, models, background, created, created_by) "
                      "VALUES(?,?,?,?,?,?,?)",
                      (label, _hash(token), token[:10], json.dumps(models or ["*"]), int(background),
                       time.time(), by))
        self._token_cache.clear()
        return {"id": cur.lastrowid, "label": label, "token": token, "models": models or ["*"],
                "background": background}

    def delete_token(self, token_id: int) -> bool:
        cur = self._x("DELETE FROM tokens WHERE id=?", (token_id,))
        self._token_cache.clear()
        return cur.rowcount > 0

    def list_tokens(self) -> list[dict]:
        rows = self._q("SELECT id, label, prefix, models, background, created, created_by, last_used, uses "
                       "FROM tokens ORDER BY id")
        for r in rows:
            r["models"] = json.loads(r["models"])
            r["background"] = bool(r["background"])
        return rows

    def lookup_token(self, token: str | None) -> dict | None:
        if not token or not token.startswith(TOKEN_PREFIX):
            return None
        h = _hash(token)
        if h not in self._token_cache:
            rows = self._q("SELECT id, label, models, background FROM tokens WHERE hash=?", (h,))
            if rows:
                r = rows[0]
                r["models"] = json.loads(r["models"])
                r["background"] = bool(r["background"])
                self._token_cache[h] = r
            else:
                self._token_cache[h] = None
        return self._token_cache[h]

    def touch_token(self, token_id: int) -> None:
        self._x("UPDATE tokens SET last_used=?, uses=uses+1 WHERE id=?", (time.time(), token_id))

    # ---- request history
    def record(self, row: dict) -> None:
        cols = ("id", "ts", "model", "endpoint", "client", "priority", "dialect_in", "dialect_out", "stream",
                "status", "ok", "attempts", "fallback", "metered", "tokens_in", "tokens_out", "queued_ms",
                "ttft_ms", "duration_ms", "tps", "error")
        vals = [row.get(c) for c in cols]
        vals[cols.index("attempts")] = json.dumps(row.get("attempts") or [])
        self._x(f"INSERT OR REPLACE INTO requests({','.join(cols)}) VALUES({','.join('?' * len(cols))})", vals)

    def recent(self, limit: int = 100, since: float | None = None) -> list[dict]:
        if since is not None:
            rows = self._q("SELECT * FROM requests WHERE ts>=? ORDER BY ts DESC LIMIT ?", (since, limit))
        else:
            rows = self._q("SELECT * FROM requests ORDER BY ts DESC LIMIT ?", (limit,))
        for r in rows:
            r["attempts"] = json.loads(r["attempts"] or "[]")
        return rows

    def timeseries(self, minutes: int = 60, bucket: int = 60) -> list[dict]:
        since = time.time() - minutes * 60
        return self._q(
            "SELECT CAST(ts/? AS INTEGER)*? AS t, endpoint, COUNT(*) AS requests, "
            "SUM(COALESCE(tokens_out,0)) AS tokens_out, SUM(1-ok) AS errors "
            "FROM requests WHERE ts>=? GROUP BY t, endpoint ORDER BY t", (bucket, bucket, since))

    def history(self, minutes: int, bucket: int, endpoints: list[str]) -> dict:
        """Per-bucket request counts, speed and first-token latency for the charts, plus the
        latest failures and fallbacks in the window."""
        now = time.time()
        since = now - minutes * 60
        start = int(since // bucket) * bucket
        n = int((now - start) // bucket) + 1
        rows = self._q("SELECT ts, model, endpoint, ok, fallback, metered, tokens_out, ttft_ms, tps "
                       "FROM requests WHERE ts>=? ORDER BY ts", (since,))
        known = list(endpoints)
        seen = {r["endpoint"] for r in rows if r["endpoint"]}
        known += sorted(seen - set(known))           # endpoints since removed from the config
        buckets = [{"t": start + i * bucket, "requests": {}, "errors": 0, "fallbacks": 0, "metered": 0,
                    "tokens_out": 0, "_tps": {}, "_ttft": []} for i in range(n)]
        tot = {"requests": 0, "ok": 0, "errors": 0, "fallbacks": 0, "metered": 0, "tokens_out": 0, "_ttft": []}
        for r in rows:
            i = min(n - 1, max(0, int((r["ts"] - start) // bucket)))
            b = buckets[i]
            ep = r["endpoint"]
            tot["requests"] += 1
            if r["ok"]:
                tot["ok"] += 1
                if ep:
                    b["requests"][ep] = b["requests"].get(ep, 0) + 1
            else:
                b["errors"] += 1
                tot["errors"] += 1
            for k in ("fallback", "metered"):
                if r[k]:
                    b[k + "s" if k == "fallback" else k] += 1
                    tot[k + "s" if k == "fallback" else k] += 1
            b["tokens_out"] += r["tokens_out"] or 0
            tot["tokens_out"] += r["tokens_out"] or 0
            if r["tps"] and ep:
                b["_tps"].setdefault(ep, []).append(r["tps"])
            if r["ttft_ms"] is not None and r["ok"]:
                b["_ttft"].append(r["ttft_ms"])
                tot["_ttft"].append(r["ttft_ms"])

        def pct(xs: list, q: float):
            if not xs:
                return None
            xs = sorted(xs)
            return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]

        for b in buckets:
            b["tps"] = {ep: round(sum(v) / len(v), 1) for ep, v in b.pop("_tps").items()}
            t = b.pop("_ttft")
            b["ttft_p50"], b["ttft_p95"] = pct(t, 0.5), pct(t, 0.95)
        t = tot.pop("_ttft")
        tot["ttft_p50"], tot["ttft_p95"] = pct(t, 0.5), pct(t, 0.95)
        problems = self._q("SELECT * FROM requests WHERE ts>=? AND (ok=0 OR fallback=1) ORDER BY ts DESC LIMIT 25",
                           (since,))
        for r in problems:
            r["attempts"] = json.loads(r["attempts"] or "[]")
        return {"bucket": bucket, "start": start, "now": now, "endpoints": known, "buckets": buckets,
                "totals": tot, "problems": problems}

    def prune(self, days: int) -> int:
        return self._x("DELETE FROM requests WHERE ts<?", (time.time() - days * 86400,)).rowcount

    # async wrappers
    async def arecord(self, row: dict) -> None:
        await asyncio.to_thread(self.record, row)
