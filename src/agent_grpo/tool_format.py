"""Qwen native tool-call parser; never executes model output."""
from __future__ import annotations

import json
import re

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def parse_tool_calls(text: str) -> list[dict]:
    """Parse complete Qwen tags; untagged prose is a no-call response."""
    matches = list(_CALL.finditer(text))
    if text.count(TOOL_CALL_START) != len(matches) or text.count(TOOL_CALL_END) != len(matches):
        raise ValueError("Unbalanced or nested tool_call tags")
    result = []
    for match in matches:
        call = json.loads(match.group(1), object_pairs_hook=_strict_object, parse_constant=_invalid_constant)
        if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
            raise ValueError("Tool call must contain exactly name and arguments")
        if not isinstance(call["name"], str) or not call["name"].strip():
            raise ValueError("Tool name must be a nonempty string")
        if not isinstance(call["arguments"], dict):
            raise ValueError("Tool arguments must be an object, not encoded JSON")
        result.append(call)
    return result
