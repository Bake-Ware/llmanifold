"""Agent CLIs as engines (dialect claude-code / codex-cli / opencode), with fake CLIs on PATH."""
import json
import os
import stat
import sys

import pytest

from llmanifold import harness as H
from llmanifold.config import ConfigError, parse

FAKE = r'''#!PYTHON
import json, os, sys
name = os.path.basename(sys.argv[0])
args = sys.argv[1:]
log = os.environ["FAKE_LOG"]
mode = open(os.environ["FAKE_MODE"]).read().strip() if os.path.exists(os.environ["FAKE_MODE"]) else "ok"
if args[:1] == ["--version"]:
    print(f"{name} 9.8.7"); sys.exit(0)
if name == "claude" and args[:2] == ["auth", "status"]:
    print(json.dumps({"loggedIn": mode != "signedout"})); sys.exit(0)
if name == "codex" and args[:2] == ["login", "status"]:
    print("Logged in using ChatGPT" if mode != "signedout" else "Not logged in"); sys.exit(0 if mode != "signedout" else 1)
if name == "codex" and args[:2] == ["debug", "models"]:
    print(json.dumps({"models": [{"slug": "gpt-9", "visibility": "list"}, {"slug": "hidden", "visibility": "hide"}]})); sys.exit(0)
if name == "opencode" and args[:1] == ["models"]:
    print("kaiju/qwopus\nopencode/free"); sys.exit(0)
prompt = sys.stdin.read()
cfg = None
if name == "opencode":
    cfg = json.load(open(os.path.join(os.environ["XDG_CONFIG_HOME"], "opencode", "opencode.json")))
with open(log, "a") as f:
    f.write(json.dumps({"name": name, "args": args, "prompt": prompt, "cwd": os.getcwd(), "cfg": cfg,
                        "oauth": os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"),
                        "api_key": os.environ.get("ANTHROPIC_API_KEY")}) + "\n")
def out(o):
    print(json.dumps(o), flush=True)
if mode == "fail":
    sys.stderr.write("something broke\n"); sys.exit(3)
if name == "claude":
    out({"type": "system", "subtype": "init", "model": "claude-haiku-9"})
    if mode == "limit":
        out({"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "unifiedWindows": {
            "five_hour": {"utilization": 1.0, "resetsAt": 1}}}})
        out({"type": "result", "is_error": True, "result": "usage limit reached", "usage": {}}); sys.exit(1)
    out({"type": "rate_limit_event", "rate_limit_info": {"status": "allowed", "unifiedWindows": {
        "five_hour": {"utilization": 0.25, "resetsAt": 1}}}})
    for w in ["Hello", " from", " claude"]:
        out({"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": w}}})
    out({"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello from claude"}]}})
    out({"type": "result", "result": "Hello from claude", "usage": {"input_tokens": 3, "cache_read_input_tokens": 7,
                                                                    "output_tokens": 4}})
elif name == "codex":
    out({"type": "thread.started"})
    if mode == "limit":
        out({"type": "turn.failed", "error": {"message": "You've hit your usage limit"}}); sys.exit(1)
    out({"type": "item.completed", "item": {"type": "agent_message", "text": "Hello from codex"}})
    out({"type": "turn.completed", "usage": {"input_tokens": 11, "output_tokens": 5}})
else:
    out({"type": "text", "part": {"text": "Hello from opencode"}})
    out({"type": "step_finish", "part": {"reason": "stop", "tokens": {"input": 9, "output": 2, "reasoning": 1}}})
'''


@pytest.fixture
def clis(tmp_path, monkeypatch):
    """Fake claude, codex and opencode on PATH; returns (log reader, set_mode)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("claude", "codex", "opencode"):
        p = bindir / name
        p.write_text(FAKE.replace("PYTHON", sys.executable))
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    home = tmp_path / "home"
    (home / ".config" / "opencode").mkdir(parents=True)
    (home / ".config" / "opencode" / "opencode.jsonc").write_text(
        '{\n // my config\n "model": "kaiju/qwopus", "provider": {"kaiju": {"npm": "x"}},\n'
        ' "mcp": {"rook": {"type": "remote", "headers": {"Authorization": "Bearer SECRET"}}},\n}')
    log, mode = tmp_path / "calls.jsonl", tmp_path / "mode"
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("FAKE_MODE", str(mode))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")

    def calls():
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return calls, mode.write_text


def chat(text="hi", stream=False, model="cli", **extra):
    return {"model": model, "stream": stream,
            "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": text}], **extra}


def test_config_needs_no_url_and_forbids_login():
    cfg = parse({"providers": {"claude": {"dialect": "claude-code", "metered": False}},
                 "endpoints": {"haiku": {"provider": "claude", "model": "haiku"},
                               "codex": {"dialect": "codex-cli"}},
                 "models": {"m": {"pool": ["haiku", "codex"]}}})
    assert cfg.endpoints["haiku"].dialect == "claude-code" and cfg.endpoints["haiku"].probe == "none"
    assert cfg.endpoints["codex"].url == "cli://codex-cli"
    with pytest.raises(ConfigError):
        parse({"endpoints": {"x": {"dialect": "codex-cli", "login": "chatgpt"}}, "models": {"m": {"pool": ["x"]}}})


def test_flatten_and_unsupported():
    system, prompt = H.flatten({"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "a"},
                                             {"role": "assistant", "content": "b"},
                                             {"role": "user", "content": [{"type": "text", "text": "c"}]}]})
    assert system == "S" and "User: a" in prompt and "Assistant: b" in prompt and prompt.rstrip().endswith("reply.")
    assert "c" in prompt.split("answer now:")[1]
    assert H.unsupported({"messages": [], "tools": [{"type": "function"}]})
    assert H.unsupported({"messages": [], "tools": [{"type": "function"}], "tool_choice": "none"}) is None
    assert H.unsupported({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "x"}]}]})


@pytest.mark.parametrize("dialect,word", [("claude-code", "claude"), ("codex-cli", "codex"), ("opencode", "opencode")])
async def test_each_cli_answers_openai_and_anthropic_clients(stack, clis, dialect, word):
    calls, _ = clis
    s = await stack({"e": {"kind": "raw", "dialect": dialect, "model": "m1", "reasoning_effort": "high"}},
                    {"cli": {"pool": ["e"]}})
    r = await s.api.post("/v1/chat/completions", json=chat("hi there"))
    assert r.status == 200, await r.text()
    body = await r.json()
    assert body["choices"][0]["message"]["content"] == f"Hello from {word}"
    assert body["usage"]["completion_tokens"] > 0
    c = calls()[-1]
    assert c["prompt"] == "hi there" and c["api_key"] is None
    args = c["args"]
    if dialect == "claude-code":
        assert args[args.index("--system-prompt") + 1] == "Be brief."
        assert args[args.index("--tools") + 1] == "" and "--strict-mcp-config" in args
        assert args[args.index("--model") + 1] == "m1" and args[args.index("--effort") + 1] == "high"
        assert body["usage"]["prompt_tokens"] == 10
    elif dialect == "codex-cli":
        assert "--ignore-user-config" in args and "features.shell_tool=false" in args
        assert 'developer_instructions="Be brief."' in args and args[-1] == "-"
        assert 'model_reasoning_effort="high"' in args
    else:
        cfg = c["cfg"]
        assert cfg["mcp"] == {} and cfg["model"] == "kaiju/qwopus" and "SECRET" not in json.dumps(cfg)
        assert cfg["agent"]["llmanifold"]["prompt"] == "Be brief." and cfg["tools"] == {"*": False}
        assert args[args.index("--variant") + 1] == "high"
    assert not os.path.exists(c["cwd"])          # the run's private dir is gone

    r = await s.api.post("/v1/messages", json={"model": "cli", "max_tokens": 50, "stream": True, "system": "Be brief.",
                                               "messages": [{"role": "user", "content": "hi"}]})
    text = await r.text()
    assert r.status == 200 and "content_block_delta" in text and "message_stop" in text


async def test_claude_streams_deltas_and_reports_plan_quota(stack, clis):
    s = await stack({"e": {"kind": "raw", "dialect": "claude-code"}}, {"cli": {"pool": ["e"]}})
    r = await s.api.post("/v1/chat/completions", json=chat(stream=True))
    raw = await r.text()
    deltas = [json.loads(b[6:])["choices"][0]["delta"].get("content") for b in raw.split("\n\n")
              if b.startswith("data: {") and json.loads(b[6:]).get("choices")]
    assert [d for d in deltas if d] == ["Hello", " from", " claude"]
    assert raw.rstrip().endswith("data: [DONE]")
    assert s.core.router.states["e"].quota[0]["used_percent"] == 25.0


async def test_tools_fall_back_and_limits_fall_back(stack, clis):
    calls, set_mode = clis
    s = await stack({"cc": {"kind": "raw", "dialect": "claude-code"},
                     "fb": {"kind": "openai", "fallback": True, "fake": {"text": "fallback answer"}}},
                    {"cli": {"pool": ["cc"], "fallback": ["fb"], "fallback_on": ["4xx", "429", "5xx", "connect"]}})
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    r = await s.api.post("/v1/chat/completions", json=chat(tools=tools))
    assert (await r.json())["choices"][0]["message"]["content"] == "fallback answer"
    assert calls() == []                          # the CLI never ran
    set_mode("limit")
    r = await s.api.post("/v1/chat/completions", json=chat())
    assert r.headers["X-LLManifold-Endpoint"] == "fb"
    assert s.core.recent[0]["attempts"][0]["trigger"] == "429"


async def test_failure_without_fallback_is_an_error(stack, clis):
    _, set_mode = clis
    set_mode("fail")
    s = await stack({"e": {"kind": "raw", "dialect": "codex-cli"}}, {"cli": {"pool": ["e"]}})
    r = await s.api.post("/v1/chat/completions", json=chat())
    assert r.status >= 500
    assert "something broke" in await r.text()


async def test_probe_marks_signed_out_cli_unhealthy_and_test_lists_models(stack, clis):
    _, set_mode = clis
    s = await stack({"cx": {"kind": "raw", "dialect": "codex-cli"},
                     "cc": {"kind": "raw", "dialect": "claude-code", "key_env": "CC_TOKEN"}},
                    {"cli": {"pool": ["cx", "cc"]}})
    t = await (await s.admin.post("/api/test-endpoint", json={"dialect": "codex-cli", "url": ""})).json()
    assert t["ok"] and t["models"] == ["gpt-9"] and t["version"] == "9.8.7"
    set_mode("signedout")
    s.core.harness.forget("codex-cli")
    s.core.harness.forget("claude-code")
    await s.core.probe(s.core.router.states["cx"])
    assert not s.core.router.states["cx"].healthy and "not signed in" in s.core.router.states["cx"].last_error
    os.environ["CC_TOKEN"] = "oat-token"
    try:
        await s.core.probe(s.core.router.states["cc"])     # signed out, but it has a token to use
        assert s.core.router.states["cc"].healthy
    finally:
        del os.environ["CC_TOKEN"]
    t = await (await s.admin.post("/api/test-endpoint", json={"dialect": "codex-cli", "url": ""})).json()
    assert not t["ok"] and "codex login" in t["error"]
