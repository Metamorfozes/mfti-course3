from __future__ import annotations

import json

from tool_schema import ToolCall


def extract_first_json(text: str) -> dict | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = text[start : i + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    return None
    return None


def parse_toolcall(text: str) -> tuple[str, ToolCall | None]:
    stripped = text.strip()
    if stripped == "NO_TOOL":
        return "no_tool", None

    payload = extract_first_json(text)
    if payload is None:
        return "invalid", None
    try:
        toolcall = ToolCall.model_validate(payload)
    except Exception:
        return "invalid", None
    return "tool", toolcall
