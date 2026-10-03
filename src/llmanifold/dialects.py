"""Translating between the OpenAI chat API and the Anthropic messages API.

Requests and full responses are converted as JSON. Streams are converted event
by event by the `*StreamTranslator` classes, which also report whether real
content (text or a tool call) has appeared yet, so the proxy can hold a stream
until then and still fall back to another endpoint if nothing useful arrives.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

# ---------------------------------------------------------------- helpers

OA_TO_AN_STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
                 "function_call": "tool_use", "content_filter": "refusal"}
AN_TO_OA_STOP = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length",
                 "tool_use": "tool_calls", "refusal": "content_filter", "pause_turn": "stop"}

DEFAULT_ANTHROPIC_MAX_TOKENS = 4096


def _id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:24]}"


def text_of(content: Any) -> str:
    """Plain text from an OpenAI or Anthropic content value (str or list of parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for part in content:
        if isinstance(part, str):
            out.append(part)
        elif isinstance(part, dict) and part.get("type") in ("text", "input_text", "output_text"):
            out.append(part.get("text") or "")
        elif isinstance(part, dict) and part.get("type") == "tool_result":
            out.append(text_of(part.get("content")))
    return "".join(out)


def _image_url_to_source(url: str) -> dict:
    if url.startswith("data:") and ";base64," in url:
        head, data = url.split(";base64,", 1)
        return {"type": "base64", "media_type": head[5:] or "image/png", "data": data}
    return {"type": "url", "url": url}


def _source_to_image_url(src: dict) -> str:
    if src.get("type") == "base64":
        return f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
    return src.get("url", "")


def _append(messages: list[dict], role: str, blocks: list[dict]) -> None:
    """Anthropic wants alternating roles: merge consecutive same-role messages."""
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"].extend(blocks)
    else:
        messages.append({"role": role, "content": list(blocks)})


# ---------------------------------------------------------------- requests

def openai_to_anthropic_request(body: dict) -> dict:
    out: dict[str, Any] = {
        "model": body.get("model"),
        "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens") or DEFAULT_ANTHROPIC_MAX_TOKENS,
    }
    system: list[str] = []
    messages: list[dict] = []
    for m in body.get("messages") or []:
        role = m.get("role")
        if role in ("system", "developer"):
            t = text_of(m.get("content"))
            if t:
                system.append(t)
        elif role == "tool":
            _append(messages, "user", [{"type": "tool_result", "tool_use_id": m.get("tool_call_id") or "",
                                        "content": text_of(m.get("content"))}])
        elif role == "assistant":
            blocks: list[dict] = []
            t = text_of(m.get("content"))
            if t:
                blocks.append({"type": "text", "text": t})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments") or "{}"
                try:
                    inp = json.loads(args) if isinstance(args, str) else args
                except ValueError:
                    inp = {"_raw": args}
                blocks.append({"type": "tool_use", "id": tc.get("id") or _id("toolu_"),
                               "name": fn.get("name", ""), "input": inp if isinstance(inp, dict) else {"value": inp}})
            _append(messages, "assistant", blocks or [{"type": "text", "text": ""}])
        else:
            c = m.get("content")
            if isinstance(c, str):
                blocks = [{"type": "text", "text": c}]
            else:
                blocks = []
                for part in c or []:
                    ptype = part.get("type")
                    if ptype in ("text", "input_text"):
                        blocks.append({"type": "text", "text": part.get("text", "")})
                    elif ptype == "image_url":
                        iu = part.get("image_url")
                        url = iu.get("url") if isinstance(iu, dict) else iu
                        blocks.append({"type": "image", "source": _image_url_to_source(url or "")})
            _append(messages, "user", blocks or [{"type": "text", "text": ""}])
    if system:
        out["system"] = "\n\n".join(system)
    out["messages"] = messages
    for k in ("temperature", "top_p", "top_k", "stream"):
        if body.get(k) is not None:
            out[k] = body[k]
    stop = body.get("stop")
    if stop:
        out["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    if body.get("user"):
        out["metadata"] = {"user_id": str(body["user"])}
    tools = body.get("tools")
    if tools:
        out["tools"] = [{"name": t["function"]["name"],
                         "description": t["function"].get("description", ""),
                         "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}}}
                        for t in tools if t.get("type", "function") == "function" and t.get("function")]
    tc = body.get("tool_choice")
    if tc == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif tc == "required":
        out["tool_choice"] = {"type": "any"}
    elif tc == "none":
        out["tool_choice"] = {"type": "none"}
    elif isinstance(tc, dict) and tc.get("function"):
        out["tool_choice"] = {"type": "tool", "name": tc["function"].get("name")}
    return out


def anthropic_to_openai_request(body: dict) -> dict:
    out: dict[str, Any] = {"model": body.get("model"), "messages": []}
    if body.get("max_tokens") is not None:
        out["max_tokens"] = body["max_tokens"]
    system = body.get("system")
    if system:
        out["messages"].append({"role": "system", "content": text_of(system)})
    for m in body.get("messages") or []:
        role = m.get("role")
        c = m.get("content")
        if isinstance(c, str):
            out["messages"].append({"role": role, "content": c})
            continue
        if role == "assistant":
            texts, calls = [], []
            for b in c or []:
                if b.get("type") == "text":
                    texts.append(b.get("text", ""))
                elif b.get("type") == "tool_use":
                    calls.append({"id": b.get("id") or _id("call_"), "type": "function",
                                  "function": {"name": b.get("name", ""),
                                               "arguments": json.dumps(b.get("input") or {})}})
            msg: dict[str, Any] = {"role": "assistant", "content": "".join(texts) or None}
            if calls:
                msg["tool_calls"] = calls
            out["messages"].append(msg)
        else:
            parts: list[dict] = []
            for b in c or []:
                btype = b.get("type")
                if btype == "tool_result":
                    # tool results must directly follow the assistant's tool_calls
                    out["messages"].append({"role": "tool", "tool_call_id": b.get("tool_use_id", ""),
                                            "content": text_of(b.get("content"))})
                elif btype == "text":
                    parts.append({"type": "text", "text": b.get("text", "")})
                elif btype == "image":
                    parts.append({"type": "image_url",
                                  "image_url": {"url": _source_to_image_url(b.get("source") or {})}})
            if parts:
                if all(p["type"] == "text" for p in parts):
                    out["messages"].append({"role": "user", "content": "".join(p["text"] for p in parts)})
                else:
                    out["messages"].append({"role": "user", "content": parts})
    for k in ("temperature", "top_p", "top_k", "stream"):
        if body.get(k) is not None:
            out[k] = body[k]
    if body.get("stop_sequences"):
        out["stop"] = list(body["stop_sequences"])
    if out.get("stream"):
        out["stream_options"] = {"include_usage": True}
    tools = body.get("tools")
    if tools:
        out["tools"] = [{"type": "function",
                         "function": {"name": t.get("name"), "description": t.get("description", ""),
                                      "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}}
                        for t in tools if t.get("name")]
    tc = body.get("tool_choice")
    if isinstance(tc, dict):
        kind = tc.get("type")
        if kind == "auto":
            out["tool_choice"] = "auto"
        elif kind == "any":
            out["tool_choice"] = "required"
        elif kind == "none":
            out["tool_choice"] = "none"
        elif kind == "tool":
            out["tool_choice"] = {"type": "function", "function": {"name": tc.get("name")}}
    meta = body.get("metadata") or {}
    if meta.get("user_id"):
        out["user"] = str(meta["user_id"])
    return out


def translate_request(body: dict, src: str, dst: str) -> dict:
    if dst == "responses":
        chat = body if src == "openai" else translate_request(body, src, "openai")
        return openai_to_responses_request(chat)
    if src == dst:
        return dict(body)
    if src == "openai" and dst == "anthropic":
        return openai_to_anthropic_request(body)
    if src == "anthropic" and dst == "openai":
        return anthropic_to_openai_request(body)
    raise ValueError(f"no translation {src} -> {dst}")


# ---------------------------------------------------------------- full responses

def openai_to_anthropic_response(resp: dict, model: str | None = None) -> dict:
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content: list[dict] = []
    thought = reasoning_of(msg)
    if thought:
        content.append({"type": "thinking", "thinking": thought, "signature": ""})
    t = text_of(msg.get("content"))
    if t:
        content.append({"type": "text", "text": t})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            inp = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            inp = {"_raw": fn.get("arguments")}
        content.append({"type": "tool_use", "id": tc.get("id") or _id("toolu_"),
                        "name": fn.get("name", ""), "input": inp if isinstance(inp, dict) else {"value": inp}})
    usage = resp.get("usage") or {}
    return {"id": resp.get("id") or _id("msg_"), "type": "message", "role": "assistant",
            "model": model or resp.get("model"), "content": content,
            "stop_reason": OA_TO_AN_STOP.get(choice.get("finish_reason") or "stop", "end_turn"),
            "stop_sequence": None,
            "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                      "output_tokens": usage.get("completion_tokens", 0)}}


def anthropic_to_openai_response(resp: dict, model: str | None = None) -> dict:
    texts, calls, thoughts = [], [], []
    for b in resp.get("content") or []:
        if b.get("type") == "text":
            texts.append(b.get("text", ""))
        elif b.get("type") == "thinking":
            thoughts.append(b.get("thinking", ""))
        elif b.get("type") == "tool_use":
            calls.append({"id": b.get("id") or _id("call_"), "type": "function",
                          "function": {"name": b.get("name", ""), "arguments": json.dumps(b.get("input") or {})}})
    msg: dict[str, Any] = {"role": "assistant", "content": "".join(texts) if texts or not calls else None}
    if calls:
        msg["tool_calls"] = calls
    if thoughts:
        msg["reasoning_content"] = "".join(thoughts)
    usage = resp.get("usage") or {}
    pin, pout = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    return {"id": resp.get("id") or _id("chatcmpl-"), "object": "chat.completion", "created": int(time.time()),
            "model": model or resp.get("model"),
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": AN_TO_OA_STOP.get(resp.get("stop_reason") or "end_turn", "stop")}],
            "usage": {"prompt_tokens": pin, "completion_tokens": pout, "total_tokens": pin + pout}}


def translate_response(resp: dict, src: str, dst: str, model: str | None = None) -> dict:
    """`src` is the endpoint's dialect, `dst` the client's."""
    if src == dst:
        if model and "model" in resp:
            resp = dict(resp, model=model)
        return resp
    if src == "openai" and dst == "anthropic":
        return openai_to_anthropic_response(resp, model)
    if src == "anthropic" and dst == "openai":
        return anthropic_to_openai_response(resp, model)
    raise ValueError(f"no translation {src} -> {dst}")


def reasoning_of(d: dict) -> str:
    """Reasoning text from an OpenAI-style message or delta (llama.cpp/vLLM/DeepSeek naming)."""
    r = d.get("reasoning_content") or d.get("reasoning") or ""
    return r if isinstance(r, str) else text_of(r)


def response_has_content(resp: dict, dialect: str) -> bool:
    """Did the model produce anything? Reasoning counts: a thinking model that spent its whole
    budget reasoning (finish_reason=length, content null) answered; it isn't a broken engine."""
    if dialect == "openai":
        msg = ((resp.get("choices") or [{}])[0]).get("message") or {}
        return bool(text_of(msg.get("content")).strip() or msg.get("tool_calls") or reasoning_of(msg).strip())
    return any((b.get("type") == "text" and (b.get("text") or "").strip()) or b.get("type") == "tool_use"
               or (b.get("type") == "thinking" and (b.get("thinking") or "").strip())
               for b in resp.get("content") or [])


def response_usage(resp: dict, dialect: str) -> tuple[int, int]:
    u = resp.get("usage") or {}
    if dialect == "openai":
        return int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
    return int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)


def is_context_error(status: int, text: str) -> bool:
    if status not in (400, 413, 422):
        return False
    t = text.lower()
    return "context" in t and any(w in t for w in ("length", "exceed", "too long", "maximum", "window"))


def error_body(dialect: str, message: str, kind: str = "api_error") -> dict:
    if dialect == "anthropic":
        return {"type": "error", "error": {"type": kind, "message": message}}
    return {"error": {"message": message, "type": kind}}


# ---------------------------------------------------------------- streams

class SSEParser:
    """Split a byte stream into server-sent events: (event_name, data, raw_bytes)."""

    def __init__(self) -> None:
        self._buf = b""

    def feed(self, chunk: bytes) -> list[tuple[str | None, str, bytes]]:
        # normalise on the whole buffer: a \r\n can be split across two network chunks
        self._buf = (self._buf + chunk).replace(b"\r\n", b"\n")
        out = []
        while b"\n\n" in self._buf:
            raw, self._buf = self._buf.split(b"\n\n", 1)
            ev = self._parse(raw)
            if ev:
                out.append(ev)
        return out

    def flush(self) -> list[tuple[str | None, str, bytes]]:
        raw, self._buf = self._buf, b""
        ev = self._parse(raw) if raw.strip() else None
        return [ev] if ev else []

    @staticmethod
    def _parse(raw: bytes):
        name, data = None, []
        for line in raw.decode("utf-8", "replace").split("\n"):
            if line.startswith(":") or not line:
                continue
            key, _, val = line.partition(":")
            val = val[1:] if val.startswith(" ") else val
            if key == "event":
                name = val
            elif key == "data":
                data.append(val)
        if name is None and not data:
            return None
        return name, "\n".join(data), raw + b"\n\n"


def sse(data: Any, event: str | None = None) -> bytes:
    payload = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {payload}\n\n".encode()


class _StreamBase:
    """Common bookkeeping: content seen, usage, upstream error, done."""

    def __init__(self, model: str | None) -> None:
        self.model = model
        self.has_content = False
        self.tokens_in = 0
        self.tokens_out = 0
        self.error: str | None = None
        self.done = False
        self.stop_reason: str | None = None
        self.deltas = 0          # content events seen: a token estimate when the engine reports no usage

    def feed(self, event: str | None, data: str, raw: bytes) -> list[bytes]:  # pragma: no cover
        raise NotImplementedError

    def finish(self) -> list[bytes]:
        return []


class OpenAIPassthrough(_StreamBase):
    """OpenAI stream to an OpenAI client: forward events, watching for content."""

    def __init__(self, model: str | None, rename: bool = False, strip_usage: bool = False) -> None:
        super().__init__(model)
        self.rename = rename
        self.strip_usage = strip_usage   # we asked the engine for usage; the client didn't

    def feed(self, event, data, raw):
        if data.strip() == "[DONE]":
            self.done = True
            return [raw]
        try:
            j = json.loads(data)
        except ValueError:
            return [raw]
        if isinstance(j, dict) and j.get("error"):
            self.error = text_of(json.dumps(j["error"]))
        for ch in j.get("choices") or []:
            d = ch.get("delta") or {}
            if (d.get("content") or "") or d.get("tool_calls") or reasoning_of(d):
                self.has_content = True
                self.deltas += 1
            if ch.get("finish_reason"):
                self.stop_reason = ch["finish_reason"]
        u = j.get("usage") or {}
        if u:
            self.tokens_in = u.get("prompt_tokens", self.tokens_in) or self.tokens_in
            self.tokens_out = u.get("completion_tokens", self.tokens_out) or self.tokens_out
            if self.strip_usage and not j.get("choices"):
                return []
        if self.rename and self.model and "model" in j:
            j["model"] = self.model
            return [sse(j)]
        return [raw]


class AnthropicPassthrough(_StreamBase):
    def __init__(self, model: str | None, rename: bool = False) -> None:
        super().__init__(model)
        self.rename = rename

    def feed(self, event, data, raw):
        try:
            j = json.loads(data)
        except ValueError:
            return [raw]
        t = j.get("type") or event
        if t == "error":
            self.error = json.dumps(j.get("error"))
        elif t == "message_start":
            u = (j.get("message") or {}).get("usage") or {}
            self.tokens_in = u.get("input_tokens", 0) or 0
            if self.rename and self.model and j.get("message"):
                j["message"]["model"] = self.model
                return [sse(j, event)]
        elif t == "content_block_start" and (j.get("content_block") or {}).get("type") == "tool_use":
            self.has_content = True
        elif t == "content_block_delta":
            d = j.get("delta") or {}
            if ((d.get("type") == "text_delta" and d.get("text")) or d.get("type") == "input_json_delta"
                    or (d.get("type") == "thinking_delta" and d.get("thinking"))):
                self.has_content = True
                self.deltas += 1
        elif t == "message_delta":
            self.tokens_out = ((j.get("usage") or {}).get("output_tokens")) or self.tokens_out
            self.stop_reason = (j.get("delta") or {}).get("stop_reason") or self.stop_reason
        elif t == "message_stop":
            self.done = True
        return [raw]


class OpenAIToAnthropicStream(_StreamBase):
    """OpenAI chat-completion chunks in, Anthropic message events out."""

    def __init__(self, model: str | None) -> None:
        super().__init__(model)
        self._started = False
        self._open: tuple[str, int] | None = None     # ("thinking"|"text"|"tool", anthropic index)
        self._next = 0
        self._tools: dict[int, int] = {}              # openai tool index -> anthropic block index
        self._finish: str | None = None
        self._msg_id = _id("msg_")

    def _start(self) -> list[bytes]:
        if self._started:
            return []
        self._started = True
        return [sse({"type": "message_start", "message": {
            "id": self._msg_id, "type": "message", "role": "assistant", "model": self.model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0}}}, "message_start")]

    def _close(self) -> list[bytes]:
        if self._open is None:
            return []
        idx = self._open[1]
        self._open = None
        return [sse({"type": "content_block_stop", "index": idx}, "content_block_stop")]

    def feed(self, event, data, raw):
        if data.strip() == "[DONE]":
            return self.finish()
        try:
            j = json.loads(data)
        except ValueError:
            return []
        if j.get("error"):
            self.error = json.dumps(j["error"])
            return []
        out = self._start()
        u = j.get("usage") or {}
        if u:
            self.tokens_in = u.get("prompt_tokens") or self.tokens_in
            self.tokens_out = u.get("completion_tokens") or self.tokens_out
        for ch in j.get("choices") or []:
            d = ch.get("delta") or {}
            thought = reasoning_of(d)
            if thought:
                if not self._open or self._open[0] != "thinking":
                    out += self._close()
                    idx = self._next
                    self._next += 1
                    self._open = ("thinking", idx)
                    out.append(sse({"type": "content_block_start", "index": idx,
                                    "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
                                   "content_block_start"))
                out.append(sse({"type": "content_block_delta", "index": self._open[1],
                                "delta": {"type": "thinking_delta", "thinking": thought}}, "content_block_delta"))
                self.has_content = True
                self.deltas += 1
            text = d.get("content")
            if text:
                if not self._open or self._open[0] != "text":
                    out += self._close()
                    idx = self._next
                    self._next += 1
                    self._open = ("text", idx)
                    out.append(sse({"type": "content_block_start", "index": idx,
                                    "content_block": {"type": "text", "text": ""}}, "content_block_start"))
                out.append(sse({"type": "content_block_delta", "index": self._open[1],
                                "delta": {"type": "text_delta", "text": text}}, "content_block_delta"))
                self.has_content = True
                self.deltas += 1
            for tc in d.get("tool_calls") or []:
                k = tc.get("index", 0)
                fn = tc.get("function") or {}
                if k not in self._tools:
                    out += self._close()
                    idx = self._next
                    self._next += 1
                    self._tools[k] = idx
                    self._open = ("tool", idx)
                    out.append(sse({"type": "content_block_start", "index": idx,
                                    "content_block": {"type": "tool_use", "id": tc.get("id") or _id("toolu_"),
                                                      "name": fn.get("name", ""), "input": {}}},
                                   "content_block_start"))
                    self.has_content = True
                if fn.get("arguments"):
                    out.append(sse({"type": "content_block_delta", "index": self._tools[k],
                                    "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]}},
                                   "content_block_delta"))
            if ch.get("finish_reason"):
                self._finish = ch["finish_reason"]
        return out

    def finish(self):
        if self.done:
            return []
        self.done = True
        out = self._start() + self._close()
        self.stop_reason = OA_TO_AN_STOP.get(self._finish or "stop", "end_turn")
        out.append(sse({"type": "message_delta", "delta": {"stop_reason": self.stop_reason, "stop_sequence": None},
                        "usage": {"output_tokens": self.tokens_out}}, "message_delta"))
        out.append(sse({"type": "message_stop"}, "message_stop"))
        return out


class AnthropicToOpenAIStream(_StreamBase):
    """Anthropic message events in, OpenAI chat-completion chunks out."""

    def __init__(self, model: str | None, include_usage: bool = False) -> None:
        super().__init__(model)
        self.include_usage = include_usage
        self._id = _id("chatcmpl-")
        self._created = int(time.time())
        self._tools: dict[int, int] = {}     # anthropic block index -> openai tool index
        self._finish: str | None = None

    def _chunk(self, delta: dict, finish: str | None = None) -> bytes:
        return sse({"id": self._id, "object": "chat.completion.chunk", "created": self._created,
                    "model": self.model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

    def feed(self, event, data, raw):
        try:
            j = json.loads(data)
        except ValueError:
            return []
        t = j.get("type") or event
        if t == "error":
            self.error = json.dumps(j.get("error"))
            return []
        if t == "message_start":
            u = (j.get("message") or {}).get("usage") or {}
            self.tokens_in = u.get("input_tokens", 0) or 0
            return [self._chunk({"role": "assistant", "content": ""})]
        if t == "content_block_start":
            b = j.get("content_block") or {}
            if b.get("type") == "tool_use":
                k = len(self._tools)
                self._tools[j.get("index", 0)] = k
                self.has_content = True
                return [self._chunk({"tool_calls": [{"index": k, "id": b.get("id") or _id("call_"),
                                                     "type": "function",
                                                     "function": {"name": b.get("name", ""), "arguments": ""}}]})]
            return []
        if t == "content_block_delta":
            d = j.get("delta") or {}
            if d.get("type") == "text_delta" and d.get("text"):
                self.has_content = True
                self.deltas += 1
                return [self._chunk({"content": d["text"]})]
            if d.get("type") == "thinking_delta" and d.get("thinking"):
                self.has_content = True
                self.deltas += 1
                return [self._chunk({"reasoning_content": d["thinking"]})]
            if d.get("type") == "input_json_delta":
                k = self._tools.get(j.get("index", 0), 0)
                return [self._chunk({"tool_calls": [{"index": k, "function": {"arguments": d.get("partial_json", "")}}]})]
            return []
        if t == "message_delta":
            self._finish = (j.get("delta") or {}).get("stop_reason") or self._finish
            self.tokens_out = (j.get("usage") or {}).get("output_tokens") or self.tokens_out
            return []
        if t == "message_stop":
            return self.finish()
        return []

    def finish(self):
        if self.done:
            return []
        self.done = True
        self.stop_reason = AN_TO_OA_STOP.get(self._finish or "end_turn", "stop")
        out = [self._chunk({}, self.stop_reason)]
        if self.include_usage:
            out.append(sse({"id": self._id, "object": "chat.completion.chunk", "created": self._created,
                            "model": self.model, "choices": [],
                            "usage": {"prompt_tokens": self.tokens_in, "completion_tokens": self.tokens_out,
                                      "total_tokens": self.tokens_in + self.tokens_out}}))
        out.append(b"data: [DONE]\n\n")
        return out


def stream_translator(src: str, dst: str, model: str | None, include_usage: bool = False,
                      rename: bool = False, strip_usage: bool = False) -> _StreamBase:
    """`src` is the endpoint's dialect, `dst` the client's."""
    if src == "responses":
        first = ResponsesToOpenAIStream(model, include_usage=include_usage or dst != "openai")
        return first if dst == "openai" else Chain(first, stream_translator("openai", dst, model))
    if src == dst == "openai":
        return OpenAIPassthrough(model, rename, strip_usage)
    if src == dst == "anthropic":
        return AnthropicPassthrough(model, rename)
    if src == "openai" and dst == "anthropic":
        return OpenAIToAnthropicStream(model)
    if src == "anthropic" and dst == "openai":
        return AnthropicToOpenAIStream(model, include_usage)
    raise ValueError(f"no translation {src} -> {dst}")


# ---------------------------------------------------------------- OpenAI Responses API (upstream only)
# Used for OpenAI's Responses API and for the ChatGPT/Codex backend. Clients never speak it to
# llmanifold; their OpenAI-chat or Anthropic requests are translated (via the chat shape) and the
# event stream comes back as chat chunks.

REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
DEFAULT_INSTRUCTIONS = "You are a helpful assistant."


def _responses_user_content(content: Any) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    parts = []
    for p in content or []:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "text":
            parts.append({"type": "input_text", "text": p.get("text", "")})
        elif p.get("type") == "image_url":
            url = p.get("image_url")
            url = url.get("url") if isinstance(url, dict) else url
            if url:
                parts.append({"type": "input_image", "image_url": url})
    return parts


def openai_to_responses_request(body: dict) -> dict:
    instructions: list[str] = []
    items: list[dict] = []
    for m in body.get("messages") or []:
        role, c = m.get("role"), m.get("content")
        if role in ("system", "developer"):
            t = text_of(c)
            if t:
                instructions.append(t)
        elif role == "user":
            parts = _responses_user_content(c)
            if parts:
                items.append({"type": "message", "role": "user", "content": parts})
        elif role == "assistant":
            t = text_of(c)
            if t:
                items.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": t}]})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                items.append({"type": "function_call", "call_id": tc.get("id") or _id("call_"),
                              "name": fn.get("name", ""), "arguments": fn.get("arguments") or "{}"})
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": m.get("tool_call_id") or "",
                          "output": text_of(c)})
    out: dict[str, Any] = {
        "model": body.get("model"),
        "instructions": "\n\n".join(instructions) or DEFAULT_INSTRUCTIONS,
        "input": items,
        "stream": True,                 # the ChatGPT backend only streams; non-stream clients get it collected
        "store": False,
        "parallel_tool_calls": bool(body.get("parallel_tool_calls", True)),
        "tool_choice": "auto",
    }
    tools = [t for t in body.get("tools") or [] if (t.get("type") == "function" and t.get("function"))]
    if tools:
        out["tools"] = [{"type": "function", "name": t["function"].get("name"),
                         "description": t["function"].get("description", ""),
                         "parameters": t["function"].get("parameters") or {"type": "object", "properties": {}},
                         "strict": False} for t in tools]
    tc = body.get("tool_choice")
    if tc in ("auto", "none", "required"):
        out["tool_choice"] = tc
    elif isinstance(tc, dict) and (tc.get("function") or {}).get("name"):
        out["tool_choice"] = {"type": "function", "name": tc["function"]["name"]}
    effort = body.get("reasoning_effort") or (body.get("reasoning") or {}).get("effort")
    out["reasoning"] = {"summary": "auto", **({"effort": effort} if effort in REASONING_EFFORTS else {})}
    return out


class ResponsesToOpenAIStream(_StreamBase):
    """Responses API events in, OpenAI chat-completion chunks out."""

    def __init__(self, model: str | None, include_usage: bool = False) -> None:
        super().__init__(model)
        self.include_usage = include_usage
        self._id = _id("chatcmpl-")
        self._created = int(time.time())
        self._started = False
        self._tools: dict[str, int] = {}        # item id -> tool index
        self._tool_args: dict[str, bool] = {}   # item id -> saw argument deltas
        self._text_items: set[str] = set()      # items that streamed text deltas
        self._finish = "stop"

    def _chunk(self, delta: dict, finish: str | None = None) -> bytes:
        return sse({"id": self._id, "object": "chat.completion.chunk", "created": self._created,
                    "model": self.model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

    def _start(self) -> list[bytes]:
        if self._started:
            return []
        self._started = True
        return [self._chunk({"role": "assistant", "content": ""})]

    def _text(self, t: str) -> list[bytes]:
        self.has_content = True
        self.deltas += 1
        return self._start() + [self._chunk({"content": t})]

    def _tool_start(self, item: dict) -> list[bytes]:
        key = item.get("id") or item.get("call_id") or str(len(self._tools))
        if key in self._tools:
            return []
        k = self._tools[key] = len(self._tools)
        self.has_content = True
        return self._start() + [self._chunk({"tool_calls": [{"index": k, "id": item.get("call_id") or _id("call_"),
                                                             "type": "function",
                                                             "function": {"name": item.get("name", ""), "arguments": ""}}]})]

    def feed(self, event, data, raw):
        if data.strip() == "[DONE]":
            return self.finish()
        try:
            j = json.loads(data)
        except ValueError:
            return []
        t = j.get("type") or event or ""
        if t == "error":
            e = j.get("error") if isinstance(j.get("error"), dict) else j
            self.error = e.get("message") or json.dumps(e)
            return []
        if t == "response.failed":
            err = ((j.get("response") or {}).get("error") or {})
            self.error = err.get("message") or json.dumps(err) or "response failed"
            return []
        if t == "response.created":
            return self._start()
        if t == "response.output_text.delta":
            self._text_items.add(j.get("item_id", ""))
            return self._text(j.get("delta", ""))
        if t in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            self.has_content = True
            self.deltas += 1
            return self._start() + [self._chunk({"reasoning_content": j.get("delta", "")})]
        if t == "response.reasoning_summary_part.done":
            return self._start() + [self._chunk({"reasoning_content": "\n\n"})]
        if t == "response.output_item.added":
            item = j.get("item") or {}
            if item.get("type") == "function_call":
                return self._tool_start(item)
            return []
        if t == "response.function_call_arguments.delta":
            key = j.get("item_id", "")
            if key not in self._tools:
                return []
            self._tool_args[key] = True
            return [self._chunk({"tool_calls": [{"index": self._tools[key], "function": {"arguments": j.get("delta", "")}}]})]
        if t == "response.output_item.done":
            item = j.get("item") or {}
            out: list[bytes] = []
            if item.get("type") == "function_call":
                out += self._tool_start(item)
                key = item.get("id") or item.get("call_id")
                if not self._tool_args.get(key) and item.get("arguments"):
                    out.append(self._chunk({"tool_calls": [{"index": self._tools[key],
                                                            "function": {"arguments": item["arguments"]}}]}))
            elif item.get("type") == "message" and item.get("id") not in self._text_items:
                txt = "".join(p.get("text", "") for p in item.get("content") or [] if isinstance(p, dict))
                if txt:
                    out += self._text(txt)
            return out
        if t in ("response.completed", "response.incomplete", "response.done"):
            r = j.get("response") or {}
            u = r.get("usage") or {}
            self.tokens_in = u.get("input_tokens") or self.tokens_in
            self.tokens_out = u.get("output_tokens") or self.tokens_out
            if t == "response.incomplete" or (r.get("incomplete_details") or {}).get("reason") == "max_output_tokens":
                self._finish = "length"
            return self.finish()
        return []

    def finish(self):
        if self.done:
            return []
        self.done = True
        self.stop_reason = "tool_calls" if self._tools and self._finish == "stop" else self._finish
        out = self._start() + [self._chunk({}, self.stop_reason)]
        if self.include_usage:
            out.append(sse({"id": self._id, "object": "chat.completion.chunk", "created": self._created,
                            "model": self.model, "choices": [],
                            "usage": {"prompt_tokens": self.tokens_in, "completion_tokens": self.tokens_out,
                                      "total_tokens": self.tokens_in + self.tokens_out}}))
        out.append(b"data: [DONE]\n\n")
        return out


class Chain(_StreamBase):
    """Two translators in a row (e.g. Responses -> OpenAI chat -> Anthropic)."""

    def __init__(self, first: _StreamBase, second: _StreamBase) -> None:
        super().__init__(first.model)
        self.first, self.second = first, second
        self._parser = SSEParser()

    def _sync(self) -> None:
        f = self.first
        self.has_content, self.error, self.deltas = f.has_content, f.error, f.deltas
        self.tokens_in, self.tokens_out = f.tokens_in, f.tokens_out
        self.done = f.done and self.second.done
        self.stop_reason = self.second.stop_reason or f.stop_reason

    def _pipe(self, chunks: list[bytes]) -> list[bytes]:
        out: list[bytes] = []
        for ev in self._parser.feed(b"".join(chunks)):
            out += self.second.feed(*ev)
        return out

    def feed(self, event, data, raw):
        out = self._pipe(self.first.feed(event, data, raw))
        self._sync()
        return out

    def finish(self):
        out = self._pipe(self.first.finish())
        if not self.second.done:
            if self.first.tokens_out:
                self.second.tokens_out = self.first.tokens_out
            out += self.second.finish()
        self._sync()
        return out


def collect_openai_chunks(chunks: list[bytes], model: str | None) -> dict:
    """Fold OpenAI chat-completion stream chunks into one chat.completion response."""
    text, thought, calls, finish, usage = [], [], {}, "stop", {}
    for _, data, _ in SSEParser().feed(b"".join(chunks)):
        if data.strip() == "[DONE]":
            continue
        j = json.loads(data)
        if j.get("usage"):
            usage = j["usage"]
        for ch in j.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content"):
                text.append(d["content"])
            if d.get("reasoning_content"):
                thought.append(d["reasoning_content"])
            for tc in d.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", 0), {"id": None, "type": "function",
                                                          "function": {"name": "", "arguments": ""}})
                c["id"] = tc.get("id") or c["id"]
                fn = tc.get("function") or {}
                c["function"]["name"] += fn.get("name") or ""
                c["function"]["arguments"] += fn.get("arguments") or ""
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    msg: dict[str, Any] = {"role": "assistant", "content": "".join(text) or (None if calls else "")}
    if thought:
        msg["reasoning_content"] = "".join(thought)
    if calls:
        msg["tool_calls"] = [calls[k] for k in sorted(calls)]
    return {"id": _id("chatcmpl-"), "object": "chat.completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0),
                      "total_tokens": usage.get("total_tokens", 0)}}
