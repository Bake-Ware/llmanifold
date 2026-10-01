import json

from llmanifold import dialects as D


def parse_sse(blob: bytes):
    p = D.SSEParser()
    return [(e, d) for e, d, _ in p.feed(blob) + p.flush()]


def test_openai_to_anthropic_request_full():
    body = {
        "model": "m", "max_tokens": 100, "temperature": 0.2, "stop": "END", "stream": True,
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": [{"type": "text", "text": "what is this?"},
                                         {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}]},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "look", "arguments": '{"x": 1}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "a cat"},
            {"role": "user", "content": "thanks"},
        ],
        "tools": [{"type": "function", "function": {"name": "look", "description": "d",
                                                     "parameters": {"type": "object"}}}],
        "tool_choice": "required",
    }
    out = D.openai_to_anthropic_request(body)
    assert out["system"] == "be brief"
    assert out["max_tokens"] == 100 and out["stop_sequences"] == ["END"] and out["stream"] is True
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["user", "assistant", "user"]           # tool result merged with the next user turn
    assert out["messages"][0]["content"][1] == {"type": "image", "source": {
        "type": "base64", "media_type": "image/png", "data": "QUJD"}}
    assert out["messages"][1]["content"][0] == {"type": "tool_use", "id": "call_1", "name": "look", "input": {"x": 1}}
    assert out["messages"][2]["content"][0]["type"] == "tool_result"
    assert out["messages"][2]["content"][1] == {"type": "text", "text": "thanks"}
    assert out["tools"][0]["input_schema"] == {"type": "object"}
    assert out["tool_choice"] == {"type": "any"}


def test_anthropic_to_openai_request_full():
    body = {
        "model": "m", "max_tokens": 50, "system": [{"type": "text", "text": "sys"}], "stream": True,
        "stop_sequences": ["X"],
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "calling"},
                                              {"type": "tool_use", "id": "tu1", "name": "f", "input": {"a": 1}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1",
                                          "content": [{"type": "text", "text": "42"}]},
                                         {"type": "text", "text": "and?"}]},
        ],
        "tools": [{"name": "f", "description": "d", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "tool", "name": "f"},
    }
    out = D.anthropic_to_openai_request(body)
    assert out["messages"][0] == {"role": "system", "content": "sys"}
    assert out["messages"][2]["tool_calls"][0]["function"] == {"name": "f", "arguments": '{"a": 1}'}
    assert out["messages"][3] == {"role": "tool", "tool_call_id": "tu1", "content": "42"}
    assert out["messages"][4] == {"role": "user", "content": "and?"}
    assert out["stop"] == ["X"] and out["stream_options"] == {"include_usage": True}
    assert out["tool_choice"] == {"type": "function", "function": {"name": "f"}}


def test_response_round_trips():
    oa = {"id": "x", "choices": [{"message": {"role": "assistant", "content": "hi", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"q": 2}'}}]},
        "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 3, "completion_tokens": 5}}
    an = D.openai_to_anthropic_response(oa, "alias")
    assert an["model"] == "alias" and an["stop_reason"] == "tool_use"
    assert an["content"] == [{"type": "text", "text": "hi"},
                             {"type": "tool_use", "id": "c1", "name": "f", "input": {"q": 2}}]
    assert an["usage"] == {"input_tokens": 3, "output_tokens": 5}
    back = D.anthropic_to_openai_response(an, "alias")
    assert back["choices"][0]["finish_reason"] == "tool_calls"
    assert json.loads(back["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]) == {"q": 2}
    assert back["usage"]["total_tokens"] == 8


def test_content_detection_and_context_errors():
    assert not D.response_has_content({"choices": [{"message": {"content": "  "}}]}, "openai")
    # a reasoning model that ran out of budget mid-thought still answered
    assert D.response_has_content({"choices": [{"message": {"content": None, "reasoning_content": "hmm"}}]}, "openai")
    assert D.response_has_content({"content": [{"type": "thinking", "thinking": "hmm"}]}, "anthropic")
    assert D.response_has_content({"content": [{"type": "tool_use"}]}, "anthropic")
    assert D.is_context_error(400, "prompt exceeds the context (262144)")
    assert D.is_context_error(400, "This model's maximum context length is 8192 tokens")
    assert not D.is_context_error(400, "temperature must be <= 2")


def _oa_chunks(*deltas, finish="stop", usage=None):
    out = b""
    for d in deltas:
        out += D.sse({"choices": [{"index": 0, "delta": d}]})
    out += D.sse({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    if usage:
        out += D.sse({"choices": [], "usage": usage})
    return out + b"data: [DONE]\n\n"


def test_stream_openai_to_anthropic_text_and_tools():
    tr = D.OpenAIToAnthropicStream("alias")
    blob = _oa_chunks({"role": "assistant"}, {"content": "Hel"}, {"content": "lo"},
                      {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "f", "arguments": ""}}]},
                      {"tool_calls": [{"index": 0, "function": {"arguments": '{"a":1}'}}]},
                      finish="tool_calls", usage={"prompt_tokens": 4, "completion_tokens": 6})
    p = D.SSEParser()
    out = b""
    for ev in p.feed(blob):
        out += b"".join(tr.feed(*ev))
    out += b"".join(tr.finish())
    events = [(e, json.loads(d)) for e, d in parse_sse(out)]
    names = [e for e, _ in events]
    assert names[0] == "message_start" and names[-1] == "message_stop"
    assert names.count("content_block_start") == 2 and names.count("content_block_stop") == 2
    text = "".join(d["delta"]["text"] for e, d in events if e == "content_block_delta"
                   and d["delta"]["type"] == "text_delta")
    assert text == "Hello"
    tool_start = [d for e, d in events if e == "content_block_start" and d["content_block"]["type"] == "tool_use"][0]
    assert tool_start["content_block"]["name"] == "f" and tool_start["index"] == 1
    md = [d for e, d in events if e == "message_delta"][0]
    assert md["delta"]["stop_reason"] == "tool_use" and md["usage"]["output_tokens"] == 6
    assert tr.has_content and tr.tokens_in == 4


def test_stream_anthropic_to_openai():
    tr = D.AnthropicToOpenAIStream("alias", include_usage=True)
    events = [
        ("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 9}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": "hi"}}),
        ("content_block_start", {"type": "content_block_start", "index": 1,
                                 "content_block": {"type": "tool_use", "id": "t1", "name": "f"}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1,
                                 "delta": {"type": "input_json_delta", "partial_json": "{}"}}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                           "usage": {"output_tokens": 3}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    out = b"".join(b"".join(tr.feed(e, json.dumps(d), b"")) for e, d in events)
    datas = [d for _, d in parse_sse(out)]
    assert datas[-1] == "[DONE]"
    chunks = [json.loads(d) for d in datas[:-1]]
    assert chunks[1]["choices"][0]["delta"] == {"content": "hi"}
    assert chunks[2]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "f"
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1]["usage"] == {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}


def test_sse_parser_handles_split_chunks_and_crlf():
    p = D.SSEParser()
    blob = b'event: a\r\ndata: {"x":1}\r\n\r\ndata: [DONE]\n\n'
    got = []
    for i in range(0, len(blob), 3):
        got += p.feed(blob[i:i + 3])
    assert [(e, d) for e, d, _ in got] == [("a", '{"x":1}'), (None, "[DONE]")]


def test_passthrough_detects_stream_error():
    tr = D.OpenAIPassthrough("m")
    tr.feed(None, json.dumps({"error": {"message": "x"}}), b"")
    assert tr.error and not tr.has_content


def test_reasoning_translates_to_thinking_and_back():
    oa = {"choices": [{"message": {"content": "hi", "reasoning_content": "think"}, "finish_reason": "stop"}]}
    an = D.translate_response(oa, "openai", "anthropic")
    assert [b["type"] for b in an["content"]] == ["thinking", "text"]
    back = D.translate_response(an, "anthropic", "openai")
    assert back["choices"][0]["message"]["reasoning_content"] == "think"

    t = D.stream_translator("openai", "anthropic", "m", False)
    out = b""
    for ev in D.SSEParser().feed(_oa_chunks({"reasoning_content": "a"}, {"content": "b"})):
        out += b"".join(t.feed(*ev))
    text = out.decode()
    assert '"thinking_delta"' in text and '"text_delta"' in text and t.has_content
    assert text.index("thinking_delta") < text.index("text_delta")

    t = D.stream_translator("anthropic", "openai", "m", False)
    ev = D.sse({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "x"}},
               "content_block_delta")
    (name, data, raw), = D.SSEParser().feed(ev)
    assert b"reasoning_content" in b"".join(t.feed(name, data, raw)) and t.has_content
