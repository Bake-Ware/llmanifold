"""Sign in with ChatGPT, so a ChatGPT plan can serve requests through the Codex backend.

This follows the Codex CLI's own login (github.com/openai/codex, codex-rs/login):

* device code: POST {issuer}/api/accounts/deviceauth/usercode {client_id} gives a user code
  the person enters at {issuer}/codex/device; polling /deviceauth/token returns an
  authorization code plus PKCE verifier, exchanged at {issuer}/oauth/token (form encoded,
  redirect_uri {issuer}/deviceauth/callback);
* refresh: POST {issuer}/oauth/token as JSON {grant_type: refresh_token, client_id, refresh_token};
* requests carry `Authorization: Bearer <access token>` and `ChatGPT-Account-ID`, the
  account id from the id token's "https://api.openai.com/auth" claims.

A Codex CLI `auth.json` can also be imported instead of signing in here. Credentials live in
`<data_dir>/chatgpt/<endpoint>.json` (mode 600) and are never shown back.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import aiohttp

log = logging.getLogger("llmanifold.chatgpt")

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"     # the Codex CLI's public OAuth client
ISSUER = "https://auth.openai.com"
REFRESH_MARGIN = 300                           # refresh when the access token has < 5 min left
DEVICE_TIMEOUT = 15 * 60


class LoginError(RuntimeError):
    pass


def jwt_claims(token: str | None) -> dict:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return {}


def _identity(tokens: dict) -> dict:
    idc = jwt_claims(tokens.get("id_token"))
    auth = idc.get("https://api.openai.com/auth") or jwt_claims(tokens.get("access_token")).get(
        "https://api.openai.com/auth") or {}
    exp = jwt_claims(tokens.get("access_token")).get("exp")
    return {"account_id": tokens.get("account_id") or auth.get("chatgpt_account_id"),
            "email": idc.get("email") or (idc.get("https://api.openai.com/profile") or {}).get("email"),
            "plan": auth.get("chatgpt_plan_type"), "expires_at": exp}


class ChatGPTLogins:
    def __init__(self, data_dir: str | os.PathLike, issuer: str = ISSUER, client_id: str = CLIENT_ID) -> None:
        self.dir = Path(data_dir).expanduser().resolve() / "chatgpt"
        self.issuer = issuer.rstrip("/")
        self.client_id = client_id
        self.session: aiohttp.ClientSession | None = None
        self._locks: dict[str, asyncio.Lock] = {}
        self._pending: dict[str, dict] = {}       # endpoint -> device login in progress
        self._tasks: dict[str, asyncio.Task] = {}

    # ---------------------------------------------------------------- storage
    def _path(self, name: str) -> Path:
        return self.dir / (re.sub(r"[^A-Za-z0-9._-]", "_", name) + ".json")

    def load(self, name: str) -> dict | None:
        try:
            return json.loads(self._path(name).read_text())
        except (OSError, ValueError):
            return None

    def _save(self, name: str, tokens: dict) -> dict:
        data = {k: tokens.get(k) for k in ("id_token", "access_token", "refresh_token")}
        data.update(_identity({**tokens, **data}))
        data["saved_at"] = time.time()
        data["last_error"] = tokens.get("last_error")
        self.dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.dir, 0o700)
        p = self._path(name)
        tmp = p.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, p)
        return data

    def logout(self, name: str) -> None:
        self._path(name).unlink(missing_ok=True)
        self._pending.pop(name, None)
        t = self._tasks.pop(name, None)
        if t:
            t.cancel()

    def status(self, name: str) -> dict:
        d = self.load(name)
        out: dict[str, Any] = {"signed_in": bool(d and d.get("refresh_token"))}
        if d:
            out.update({k: d.get(k) for k in ("email", "plan", "account_id", "expires_at", "saved_at")})
            out["last_error"] = d.get("last_error")
        p = self._pending.get(name)
        if p:
            out["pending"] = {k: p.get(k) for k in ("user_code", "url", "expires_at", "error", "state")}
        return out

    # ---------------------------------------------------------------- HTTP helpers
    async def _post(self, url: str, *, json_body: dict | None = None, form: dict | None = None) -> tuple[int, Any]:
        kw: dict[str, Any] = {"timeout": aiohttp.ClientTimeout(total=30)}
        if form is not None:
            kw["data"] = form
        else:
            kw["json"] = json_body
        async with self.session.post(url, **kw) as r:
            text = await r.text()
            try:
                return r.status, json.loads(text)
            except ValueError:
                return r.status, text

    async def _exchange_code(self, code: str, verifier: str) -> dict:
        status, body = await self._post(f"{self.issuer}/oauth/token", form={
            "grant_type": "authorization_code", "client_id": self.client_id, "code": code,
            "redirect_uri": f"{self.issuer}/deviceauth/callback", "code_verifier": verifier})
        if status >= 400 or not isinstance(body, dict) or not body.get("access_token"):
            raise LoginError(f"token exchange failed (HTTP {status})")
        return body

    # ---------------------------------------------------------------- device login
    async def start(self, name: str) -> dict:
        status, body = await self._post(f"{self.issuer}/api/accounts/deviceauth/usercode",
                                        json_body={"client_id": self.client_id})
        if status == 404:
            raise LoginError("OpenAI didn't offer a device code (device sign-in may be off for this account); "
                             "import a Codex auth.json instead")
        if status >= 400 or not isinstance(body, dict):
            raise LoginError(f"OpenAI refused the device code request (HTTP {status})")
        code = body.get("user_code") or body.get("usercode")
        pending = {"user_code": code, "url": f"{self.issuer}/codex/device", "state": "waiting",
                   "expires_at": time.time() + DEVICE_TIMEOUT, "device_auth_id": body.get("device_auth_id"),
                   "interval": max(1, int(str(body.get("interval") or "5").strip() or 5))}
        old = self._tasks.pop(name, None)
        if old:
            old.cancel()
        self._pending[name] = pending
        self._tasks[name] = asyncio.get_running_loop().create_task(self._poll(name, pending))
        return self.status(name)["pending"]

    async def _poll(self, name: str, p: dict) -> None:
        try:
            while time.time() < p["expires_at"]:
                await asyncio.sleep(p["interval"])
                status, body = await self._post(f"{self.issuer}/api/accounts/deviceauth/token", json_body={
                    "device_auth_id": p["device_auth_id"], "user_code": p["user_code"]})
                if status in (403, 404):
                    continue                      # not approved yet
                if status >= 400 or not isinstance(body, dict):
                    raise LoginError(f"device sign-in failed (HTTP {status})")
                tokens = await self._exchange_code(body["authorization_code"], body["code_verifier"])
                self._save(name, tokens)
                p["state"] = "done"
                log.info("chatgpt sign-in completed for %s", name)
                return
            p["state"], p["error"] = "expired", "the code expired before it was entered"
        except asyncio.CancelledError:
            raise
        except Exception as e:
            p["state"], p["error"] = "error", str(e)
            log.warning("chatgpt sign-in for %s failed: %s", name, e)

    def import_auth_json(self, name: str, text: str) -> dict:
        """Accept a Codex CLI auth.json ({"tokens": {...}}) or a bare token object."""
        try:
            data = json.loads(text)
        except ValueError:
            raise LoginError("that isn't valid JSON") from None
        tokens = data.get("tokens") if isinstance(data, dict) and isinstance(data.get("tokens"), dict) else data
        if not isinstance(tokens, dict) or not tokens.get("refresh_token") or not tokens.get("access_token"):
            raise LoginError("no access_token and refresh_token found; use the auth.json from `codex login`")
        return self._save(name, tokens)

    # ---------------------------------------------------------------- use
    async def headers(self, name: str, force_refresh: bool = False) -> dict:
        lock = self._locks.setdefault(name, asyncio.Lock())
        async with lock:
            d = self.load(name)
            if not d or not d.get("refresh_token"):
                raise LoginError("not signed in to ChatGPT")
            exp = d.get("expires_at") or 0
            if force_refresh or not d.get("access_token") or exp - time.time() < REFRESH_MARGIN:
                d = await self._refresh(name, d)
            h = {"Authorization": f"Bearer {d['access_token']}"}
            if d.get("account_id"):
                h["ChatGPT-Account-ID"] = d["account_id"]
            return h

    async def _refresh(self, name: str, d: dict) -> dict:
        status, body = await self._post(f"{self.issuer}/oauth/token", json_body={
            "grant_type": "refresh_token", "client_id": self.client_id, "refresh_token": d["refresh_token"]})
        if status >= 400 or not isinstance(body, dict) or not body.get("access_token"):
            msg = f"refreshing the ChatGPT sign-in failed (HTTP {status}); sign in again"
            d["last_error"] = msg
            self._save(name, d)
            raise LoginError(msg)
        merged = {**d, **{k: v for k, v in body.items() if v}}
        merged["last_error"] = None
        saved = self._save(name, merged)
        saved["last_error"] = None
        return saved
