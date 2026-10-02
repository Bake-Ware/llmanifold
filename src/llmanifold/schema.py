"""Structured output for endpoints without `response_format: json_schema`.

Some APIs (DeepSeek) only offer JSON mode: valid JSON, no schema. For those the
request is sent as `json_object` with the schema written into the prompt, the
reply is checked here, and a reply that doesn't fit is sent back to the model
with what was wrong and the format to use.

The validator covers the JSON Schema keywords structured-output schemas use;
keywords it doesn't know are ignored rather than failed.
"""
from __future__ import annotations

import json
import re
from typing import Any

MAX_ROUNDS = 3   # the first answer plus two corrections

_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def wanted(body: dict) -> dict | None:
    """The JSON schema an OpenAI-style request asks its reply to match, if any."""
    rf = body.get("response_format")
    if isinstance(rf, dict) and rf.get("type") == "json_schema":
        js = rf.get("json_schema") or {}
        schema = js.get("schema") if isinstance(js, dict) else None
        return schema if isinstance(schema, dict) else {}
    return None


def instruction(schema: dict) -> str:
    return ("Respond only with a JSON object, with no other text, in the format given by this JSON schema:\n"
            + json.dumps(schema))


def followup(schema: dict, errors: list[str]) -> str:
    return ("That response did not match the required format:\n- " + "\n- ".join(errors[:10])
            + "\nPlease respond in the format given by this JSON schema, as a JSON object with no other text:\n"
            + json.dumps(schema))


def emulated_request(up: dict, schema: dict) -> dict:
    """The upstream request with JSON mode in place of the schema, and the schema in the prompt."""
    up = dict(up)
    up["response_format"] = {"type": "json_object"}
    msgs = list(up.get("messages") or [])
    i = 0
    while i < len(msgs) and msgs[i].get("role") in ("system", "developer"):
        i += 1
    msgs.insert(i, {"role": "system", "content": instruction(schema)})
    up["messages"] = msgs
    return up


def check(text: str | None, schema: dict) -> list[str]:
    """Problems with `text` as an answer to `schema`; empty when it fits."""
    if not text or not text.strip():
        return ["the response was empty"]
    try:
        value = json.loads(text)
    except ValueError as e:
        return [f"the response is not valid JSON ({e})"]
    return validate(value, schema)


def validate(value: Any, schema: Any, root: dict | None = None, path: str = "$") -> list[str]:
    if schema is True or schema == {} or not isinstance(schema, dict):
        return [f"{path}: nothing is allowed here"] if schema is False else []
    root = schema if root is None else root
    errs: list[str] = []
    if "$ref" in schema:
        target = _resolve(schema["$ref"], root)
        if target is not None:
            errs += validate(value, target, root, path)
    t = schema.get("type")
    if t is not None and not any(_is(value, x) for x in (t if isinstance(t, list) else [t])):
        return errs + [f"{path}: expected {t if isinstance(t, str) else ' or '.join(t)}, got {_name(value)}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: must be one of {json.dumps(schema['enum'])}")
    if "const" in schema and value != schema["const"]:
        errs.append(f"{path}: must be {json.dumps(schema['const'])}")
    for sub in schema.get("allOf") or []:
        errs += validate(value, sub, root, path)
    for key in ("anyOf", "oneOf"):
        subs = schema.get(key)
        if subs:
            fits = sum(1 for sub in subs if not validate(value, sub, root, path))
            if fits == 0 or (key == "oneOf" and fits > 1):
                errs.append(f"{path}: must match {'exactly one' if key == 'oneOf' else 'one'} of the allowed forms")
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for k in schema.get("required") or []:
            if k not in value:
                errs.append(f"{path}: missing required property {k!r}")
        extra = schema.get("additionalProperties", True)
        for k, v in value.items():
            if k in props:
                errs += validate(v, props[k], root, f"{path}.{k}")
            elif extra is False:
                errs.append(f"{path}: unexpected property {k!r}")
            elif isinstance(extra, dict):
                errs += validate(v, extra, root, f"{path}.{k}")
    elif isinstance(value, list):
        if isinstance(schema.get("items"), (dict, bool)):
            for i, v in enumerate(value):
                errs += validate(v, schema["items"], root, f"{path}[{i}]")
        if "minItems" in schema and len(value) < schema["minItems"]:
            errs.append(f"{path}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: allows at most {schema['maxItems']} items")
    elif isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errs.append(f"{path}: must be at least {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(f"{path}: must be at most {schema['maxLength']} characters")
        if "pattern" in schema:
            try:
                if not re.search(schema["pattern"], value):
                    errs.append(f"{path}: must match the pattern {schema['pattern']}")
            except re.error:
                pass
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        for key, bad, word in (("minimum", lambda a, b: a < b, ">="), ("maximum", lambda a, b: a > b, "<="),
                               ("exclusiveMinimum", lambda a, b: a <= b, ">"),
                               ("exclusiveMaximum", lambda a, b: a >= b, "<")):
            lim = schema.get(key)
            if isinstance(lim, (int, float)) and not isinstance(lim, bool) and bad(value, lim):
                errs.append(f"{path}: must be {word} {lim}")
    return errs


def _is(value: Any, t: str) -> bool:
    if t == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (isinstance(value, float) and value.is_integer())
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    cls = _TYPES.get(t)
    return True if cls is None else isinstance(value, cls)


def _name(value: Any) -> str:
    for n, cls in (("null", type(None)), ("boolean", bool), ("number", (int, float)), ("string", str),
                   ("array", list), ("object", dict)):
        if isinstance(value, cls):
            return n
    return type(value).__name__


def _resolve(ref: str, root: dict) -> Any:
    if not isinstance(ref, str) or not ref.startswith("#"):
        return None
    node: Any = root
    for part in ref[1:].strip("/").split("/"):
        if not part:
            continue
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node
