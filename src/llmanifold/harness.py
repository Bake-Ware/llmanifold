"""Agent CLIs as engines: Claude Code, Codex and OpenCode on this host.

An endpoint (or provider) with `dialect: claude-code`, `codex-cli` or
`opencode` has no URL. Each request runs the CLI once, headless, with its own
tools, settings, plugins, MCP servers and session files switched off as far as
it allows, so it answers like a plain model. It is signed in the way the CLI
is on this host (`claude auth login`, `codex login`, `opencode auth login`),
which is how a Claude or ChatGPT plan, or any provider OpenCode knows, can
serve requests.

What the CLIs can't do is call the client's tools, so a request that offers
tools (or sends images) is refused with a 400 before anything runs, and the
flow's fallbacks can take it (trigger `4xx`). The conversation is flattened
into one prompt; system messages become the system prompt.

The reply comes back as OpenAI chat chunks (SSE bytes), so the proxy relays it
like any other OpenAI engine: Claude Code streams token deltas, Codex and
OpenCode send each finished message whole.
"""
from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

from . import dialects as D

HARNESSES = ("claude-code", "codex-cli", "opencode")
#: How long a version/sign-in check is trusted before it is run again.
STATUS_TTL = 300.0
DEFAULT_SYSTEM = "You are a helpful assistant."
MAX_PROMPT = 1_000_000
#: llmanifold's reasoning levels -> what each CLI accepts.
EFFORTS = {
    "claude-code": {"low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max",
                    "minimal": "low", "none": "low"},
    "codex-cli": {"none": "minimal", "minimal": "minimal", "low": "low", "medium": "medium", "high": "high",
                  "xhigh": "xhigh", "max": "xhigh"},
    "opencode": {"none": "none", "minimal": "minimal", "low": "low", "medium": "medium", "high": "high",
                 "xhigh": "xhigh", "max": "max"},
}
#: Codex built-in tool features switched off (unknown names are ignored, so this may run ahead).
CODEX_OFF = ("shell_tool", "unified_exec", "multi_agent", "image_generation", "browser_use",
             "browser_use_external", "computer_use", "plugins", "apps", "view_image", "hooks",
             "skill_search", "tool_suggest", "sleep_tool", "goals", "memories", "in_app_browser",
             "in_app_local_automation", "workspace_dependencies")
OPENCODE_AGENT = "llmanifold"


class HarnessError(Exception):
    """The run failed before any content: `trigger` is the fallback trigger it counts as."""

    def __init__(self, trigger: str, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.trigger = trigger
        self.status = status


def is_harness(dialect: str | None) -> bool:
    return dialect in HARNESSES


def _home() -> str:
    h = os.environ.get("HOME")
    if h:
        return h
    with contextlib.suppress(Exception):
        import pwd
        return pwd.getpwuid(os.getuid()).pw_dir
    return os.path.expanduser("~")


def find(dialect: str) -> str | None:
    """The CLI's binary: PATH first, then where its installer puts it."""
    binary = {"claude-code": "claude", "codex-cli": "codex", "opencode": "opencode"}[dialect]
    exe = shutil.which(binary)
    if exe:
        return exe
    for rel in (f".local/bin/{binary}", f".{binary}/bin/{binary}", f".npm-global/bin/{binary}"):
        p = Path(_home()) / rel
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


def strip_jsonc(text: str) -> str:
    """JSON with comments and trailing commas (opencode.jsonc) as JSON."""
    out, i, n, in_str = [], 0, len(text), False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 1
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
            out.append(c)
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
            continue
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        else:
            out.append(c)
        i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def _opencode_user_config() -> dict:
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path(_home()) / ".config") / "opencode"
    for name in ("opencode.json", "opencode.jsonc", "config.json"):
        p = base / name
        if p.is_file():
            with contextlib.suppress(OSError, ValueError):
                return json.loads(strip_jsonc(p.read_text(encoding="utf-8")))
    return {}


# ---------------------------------------------------------------- the request

def unsupported(body: dict) -> str | None:
    """Why an OpenAI-shaped request can't run on a CLI, or None."""
    if body.get("tools") and body.get("tool_choice") != "none":
        return "this model runs an agent CLI, which can't call the client's tools"
    for m in body.get("messages") or []:
        if m.get("role") == "tool" or m.get("tool_calls"):
            return "this model runs an agent CLI, which can't take tool calls or tool results"
        content = m.get("content")
        if isinstance(content, list) and any(isinstance(p, dict) and p.get("type") not in ("text", None)
                                             for p in content):
            return "this model runs an agent CLI, which takes text only"
    return None


def flatten(body: dict) -> tuple[str, str]:
    """OpenAI chat messages as `(system, prompt)` for one headless run."""
    msgs = body.get("messages") or []
    system = "\n\n".join(D.text_of(m.get("content")) for m in msgs
                         if m.get("role") in ("system", "developer") and D.text_of(m.get("content")))
    rest = [m for m in msgs if m.get("role") in ("user", "assistant") and D.text_of(m.get("content"))]
    if len(rest) == 1 and rest[0]["role"] == "user":
        return system, D.text_of(rest[0]["content"])
    if not rest:
        return system, ""
    lines = [("Assistant: " if m["role"] == "assistant" else "User: ") + D.text_of(m["content"])
             for m in rest[:-1]]
    prompt = ("The conversation so far:\n\n" + "\n\n".join(lines)
              + "\n\nThe message to answer now:\n" + D.text_of(rest[-1]["content"])
              + "\n\nWrite only your reply.")
    return system, prompt


def _effort(dialect: str, body: dict, default: str | None) -> str | None:
    e = body.get("reasoning_effort") or (body.get("reasoning") or {}).get("effort") or default
    return EFFORTS[dialect].get(str(e)) if e else None


def command(dialect: str, exe: str, tmp: Path, *, model: str | None, system: str,
            effort: str | None) -> tuple[list[str], dict]:
    """`(argv, extra env)` for one run; the prompt goes in on stdin."""
    if dialect == "claude-code":
        argv = [exe, "-p", "--output-format", "stream-json", "--verbose", "--include-partial-messages",
                "--system-prompt", system, "--tools", "", "--setting-sources", "",
                "--strict-mcp-config", "--permission-mode", "dontAsk",
                "--no-session-persistence", "--disable-slash-commands"]
        if model:
            argv += ["--model", model]
        if effort:
            argv += ["--effort", effort]
        return argv, {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1", "DISABLE_AUTOUPDATER": "1"}
    if dialect == "codex-cli":
        argv = [exe, "exec", "--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config",
                "--ignore-rules", "-s", "read-only", "-C", str(tmp),
                "-c", 'approval_policy="never"', "-c", 'web_search="disabled"',
                "-c", "include_apply_patch_tool=false",
                "-c", f"developer_instructions={json.dumps(system, ensure_ascii=True)}"]
        for feature in CODEX_OFF:
            argv += ["-c", f"features.{feature}=false"]
        if model:
            argv += ["-m", model]
        if effort:
            argv += ["-c", f'model_reasoning_effort="{effort}"']
        return argv + ["-"], {}
    # opencode: a generated config in a private dir; the user's own config (with its MCP servers,
    # plugins and agents) is never loaded, only its model providers are copied in
    user = _opencode_user_config()
    cfg: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json", "autoupdate": False, "share": "disabled",
        "mcp": {}, "tools": {"*": False},
        "permission": {"edit": "deny", "bash": "deny", "webfetch": "deny"},
        "agent": {OPENCODE_AGENT: {"mode": "primary", "prompt": system, "tools": {"*": False}}}}
    for key in ("provider", "model", "small_model", "enabled_providers", "disabled_providers"):
        if key in user:
            cfg[key] = user[key]
    d = tmp / "config" / "opencode"
    d.mkdir(parents=True)
    (d / "opencode.json").write_text(json.dumps(cfg), encoding="utf-8")
    # --print-logs at ERROR: its JSON error events only say "check server logs"; the log says why
    argv = [exe, "run", "--format", "json", "--pure", "--agent", OPENCODE_AGENT, "--print-logs",
            "--log-level", "ERROR"]
    if model:
        argv += ["-m", model]
    if effort:
        argv += ["--variant", effort]
    return argv, {"XDG_CONFIG_HOME": str(tmp / "config"), "OPENCODE_DISABLE_AUTOUPDATE": "1"}


class Events:
    """Folds a CLI's JSON lines into text pieces, usage and errors."""

    def __init__(self, dialect: str) -> None:
        self.dialect = dialect
        self.model = ""
        self.tokens_in = 0
        self.tokens_out = 0
        self.error = ""
        self.done = False
        self.streamed = False          # claude: text came as deltas, so skip the whole message
        self.quota: list[dict] | None = None
        self.any_text = False

    def feed(self, ev: dict) -> list[str]:
        """The new text in one event."""
        texts = getattr(self, "_" + self.dialect.replace("-", "_"))(ev)
        self.any_text = self.any_text or bool(texts)
        return texts

    def _claude_code(self, ev: dict) -> list[str]:
        kind = ev.get("type")
        if kind == "system" and ev.get("subtype") == "init":
            self.model = str(ev.get("model") or "")
        elif kind == "stream_event":
            e = ev.get("event") or {}
            d = e.get("delta") or {}
            if e.get("type") == "content_block_delta" and d.get("type") == "text_delta" and d.get("text"):
                self.streamed = True
                return [str(d["text"])]
        elif kind == "assistant" and not self.streamed:
            msg = ev.get("message") or {}
            return [str(b["text"]) for b in msg.get("content") or []
                    if isinstance(b, dict) and b.get("type") == "text" and b.get("text")]
        elif kind == "rate_limit_event":
            info = ev.get("rate_limit_info") or {}
            wins = info.get("unifiedWindows") or {}
            self.quota = [{"used_percent": round(float(w["utilization"]) * 100, 1),
                           "window_seconds": {"five_hour": 18000, "seven_day": 604800}.get(k),
                           "reset_at": w.get("resetsAt")}
                          for k, w in wins.items() if isinstance(w, dict) and w.get("utilization") is not None]
            if info.get("status") == "rejected":
                self.error = "429: the Claude plan's usage limit is reached"
        elif kind == "result":
            self.done = True
            u = ev.get("usage") or {}
            self.tokens_in = int(u.get("input_tokens") or 0) + int(u.get("cache_read_input_tokens") or 0) \
                + int(u.get("cache_creation_input_tokens") or 0)
            self.tokens_out = int(u.get("output_tokens") or 0)
            if ev.get("is_error"):
                self.error = self.error or str(ev.get("result") or "the model returned an error")[:500]
            elif not self.any_text and isinstance(ev.get("result"), str) and ev["result"]:
                return [ev["result"]]     # otherwise it repeats the text already sent
        return []

    def _codex_cli(self, ev: dict) -> list[str]:
        kind = ev.get("type")
        item = ev.get("item") if isinstance(ev.get("item"), dict) else {}
        if kind == "item.completed" and item.get("type") == "agent_message" and item.get("text"):
            return [str(item["text"])]
        if kind == "turn.completed":
            self.done = True
            u = ev.get("usage") or {}
            self.tokens_in = int(u.get("input_tokens") or 0)
            self.tokens_out = int(u.get("output_tokens") or 0)
        elif kind in ("turn.failed", "error"):
            err = ev.get("error") if isinstance(ev.get("error"), dict) else {}
            self.error = str(err.get("message") or ev.get("message") or "codex failed")[:500]
        return []

    def _opencode(self, ev: dict) -> list[str]:
        kind = ev.get("type")
        part = ev.get("part") if isinstance(ev.get("part"), dict) else {}
        if kind == "text" and part.get("text") and not part.get("synthetic"):
            return [str(part["text"])]
        if kind == "step_finish":
            tok = part.get("tokens") if isinstance(part.get("tokens"), dict) else {}
            self.tokens_in += int(tok.get("input") or 0)
            self.tokens_out += int(tok.get("output") or 0) + int(tok.get("reasoning") or 0)
            if part.get("reason") == "stop":
                self.done = True
        elif kind == "error":
            err = ev.get("error") if isinstance(ev.get("error"), dict) else {}
            data = err.get("data") if isinstance(err.get("data"), dict) else {}
            self.error = str(data.get("message") or err.get("name") or "opencode failed")[:500]
        return []


def _trigger(message: str) -> str:
    low = message.lower()
    if low.startswith("429") or "rate limit" in low or "usage limit" in low or "quota" in low:
        return "429"
    if any(w in low for w in ("not logged in", "not signed in", "log in", "unauthorized", "401",
                                "authentication", "expired")):
        return "connect"
    if "context" in low and ("long" in low or "window" in low or "exceed" in low):
        return "context"
    return "5xx"


async def run(dialect: str, body: dict, *, model: str | None, effort: str | None, workdir: Path,
              timeout: float, token: str | None = None, on_quota=None) -> AsyncIterator[bytes]:
    """Run one request; yields OpenAI chat-completion chunks as SSE bytes. Raises HarnessError
    when the run fails before any text (a failure after text ends the stream with an error event).
    `token` (claude-code only): a `claude setup-token` token instead of this host's sign-in."""
    exe = find(dialect)
    if exe is None:
        raise HarnessError("connect", f"{dialect}: the CLI is not installed on this host")
    system, prompt = flatten(body)
    if not prompt.strip():
        raise HarnessError("4xx", "nothing to answer: the request has no user message", 400)
    if len(prompt) > MAX_PROMPT:
        raise HarnessError("context", "the conversation is too long for a CLI prompt", 400)
    workdir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="run-", dir=workdir))
    os.chmod(tmp, 0o700)
    proc = None
    try:
        argv, extra = command(dialect, exe, tmp, model=model, system=system or DEFAULT_SYSTEM,
                              effort=_effort(dialect, body, effort))
        env = {**os.environ, "HOME": _home(), **extra}
        env.pop("ANTHROPIC_API_KEY", None)       # use the plan sign-in, not an API key that happens to be set
        env.pop("OPENAI_API_KEY", None)
        if token and dialect == "claude-code":
            env["CLAUDE_CODE_OAUTH_TOKEN"] = token
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=str(tmp), env=env, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, limit=16 * 1024 * 1024, start_new_session=True)
        err_tail: collections.deque = collections.deque(maxlen=10)

        async def pump_err() -> None:
            async for raw in proc.stderr:
                line = raw.decode("utf-8", "replace").rstrip()
                if line:
                    err_tail.append(line[:300])

        err_task = asyncio.create_task(pump_err())
        proc.stdin.write(prompt.encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()
        rid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
        ev_state = Events(dialect)
        sent = 0

        def chunk(delta: dict, finish: str | None = None, usage: dict | None = None) -> bytes:
            obj = {"id": rid, "object": "chat.completion.chunk", "created": created,
                   "model": ev_state.model or model or dialect,
                   "choices": [] if usage else [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if usage:
                obj["usage"] = usage
            return D.sse(obj)

        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise asyncio.TimeoutError
            raw = await asyncio.wait_for(proc.stdout.readline(), left)
            if not raw:
                break
            try:
                ev = json.loads(raw.decode("utf-8", "replace"))
            except ValueError:
                continue
            if not isinstance(ev, dict):
                continue
            texts = ev_state.feed(ev)
            if ev_state.quota is not None and on_quota:
                on_quota(ev_state.quota)
            for t in texts:
                if sent and dialect != "claude-code":
                    t = "\n\n" + t          # separate whole messages
                yield chunk({"role": "assistant", "content": t} if not sent else {"content": t})
                sent += 1
        await asyncio.wait_for(proc.wait(), max(1.0, deadline - time.monotonic()))
        with contextlib.suppress(Exception):
            await asyncio.wait_for(err_task, 2)
        problem = ev_state.error or (
            "" if ev_state.done or (sent and proc.returncode == 0)
            else f"{dialect} exited with {proc.returncode}: " + (" | ".join(err_tail) or "no output"))
        if problem and dialect == "opencode":
            why = [m.group(1) for m in (re.search(r'error="([^"]+)"', ln) for ln in err_tail) if m]
            if why:
                problem = why[-1]
        if problem and not sent:
            raise HarnessError(_trigger(problem), f"{dialect}: {problem}")
        if problem:
            yield D.sse({"error": {"message": f"{dialect}: {problem}", "type": "api_error"}})
            return
        yield chunk({}, "stop")
        yield chunk({}, usage={"prompt_tokens": ev_state.tokens_in, "completion_tokens": ev_state.tokens_out,
                               "total_tokens": ev_state.tokens_in + ev_state.tokens_out})
        yield b"data: [DONE]\n\n"
    finally:
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, 15)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.wait(), 5)
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, 9)
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- health and models

async def _cmd(argv: list[str], timeout: float = 30.0) -> tuple[int, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, env={**os.environ, "HOME": _home()})
    except OSError:
        return 127, ""
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        return 124, ""
    return proc.returncode or 0, out.decode("utf-8", "replace")


class Status:
    """Each CLI's version and sign-in, re-read every STATUS_TTL seconds (or sooner after a failure)."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, dict]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def forget(self, dialect: str) -> None:
        self._cache.pop(dialect, None)

    def cached(self, dialect: str) -> dict | None:
        hit = self._cache.get(dialect)
        return hit[1] if hit else None

    async def get(self, dialect: str) -> dict:
        """`{installed, version, logged_in}`; logged_in is None when the CLI can't say."""
        hit = self._cache.get(dialect)
        if hit and time.monotonic() - hit[0] < STATUS_TTL:
            return hit[1]
        lock = self._locks.setdefault(dialect, asyncio.Lock())
        async with lock:
            hit = self._cache.get(dialect)
            if hit and time.monotonic() - hit[0] < STATUS_TTL:
                return hit[1]
            out = await self._read(dialect)
            self._cache[dialect] = (time.monotonic(), out)
            return out

    @staticmethod
    async def _read(dialect: str) -> dict:
        exe = find(dialect)
        if exe is None:
            return {"installed": False, "version": None, "logged_in": False}
        _, text = await _cmd([exe, "--version"])
        m = re.search(r"\d+(\.\d+)+", text or "")
        out = {"installed": True, "version": m.group(0) if m else None, "logged_in": None}
        if dialect == "claude-code":
            _, text = await _cmd([exe, "auth", "status"])
            with contextlib.suppress(ValueError, AttributeError):
                out["logged_in"] = bool(json.loads(text or "{}").get("loggedIn"))
        elif dialect == "codex-cli":
            code, text = await _cmd([exe, "login", "status"])
            out["logged_in"] = code == 0 and "logged in" in (text or "").lower()
        return out


async def models(dialect: str) -> list[str]:
    """Model names the CLI can use here (claude: its aliases)."""
    exe = find(dialect)
    if exe is None:
        return []
    if dialect == "claude-code":
        return ["haiku", "sonnet", "opus"]
    if dialect == "codex-cli":
        _, text = await _cmd([exe, "debug", "models"])
        with contextlib.suppress(ValueError, AttributeError, TypeError, KeyError):
            return [m["slug"] for m in json.loads(text)["models"] if m.get("visibility") == "list"]
        return []
    _, text = await _cmd([exe, "models"], timeout=60)
    return [ln.strip() for ln in (text or "").splitlines() if "/" in ln and " " not in ln.strip()]
