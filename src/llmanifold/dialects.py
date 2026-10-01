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
    texts, calls = [], []
    for b in resp.get("content") or []:
        if b.get("type") == "text":
            texts.append(b.get("text", ""))
        elif b.get("type") == "tool_use":
            calls.append({"id": b.get("id") or _id("call_"), "type": "function",
                          "function": {"name": b.get("name", ""), "arguments": json.dumps(b.get("input") or {})}})
    msg: dict[str, Any] = {"role": "assistant", "content": "".join(texts) if texts or not calls else None}
    if calls:
        msg["tool_calls"] = calls
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


def response_has_content(resp: dict, dialect: str) -> bool:
    if dialect == "openai":
        msg = ((resp.get("choices") or [{}])[0]).get("message") or {}
        return bool(text_of(msg.get("content")).strip() or msg.get("tool_calls"))
    return any((b.get("type") == "text" and (b.get("text") or "").strip()) or b.get("type") == "tool_use"
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
            if (d.get("content") or "") or d.get("tool_calls"):
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
            if (d.get("type") == "text_delta" and d.get("text")) or d.get("type") == "input_json_delta":
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
        self._open: tuple[str, int] | None = None     # ("text"|"tool", anthropic index)
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
    if src == dst == "openai":
        return OpenAIPassthrough(model, rename, strip_usage)
    if src == dst == "anthropic":
        return AnthropicPassthrough(model, rename)
    if src == "openai" and dst == "anthropic":
        return OpenAIToAnthropicStream(model)
    if src == "anthropic" and dst == "openai":
        return AnthropicToOpenAIStream(model, include_usage)
    raise ValueError(f"no translation {src} -> {dst}")
