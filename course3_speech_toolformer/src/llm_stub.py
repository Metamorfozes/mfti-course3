from __future__ import annotations

import json
import re

from tool_schema import ALL_UNITS, normalize_unit, unit_group

_RU_MAP_KEYS = {
    "км",
    "м",
    "см",
    "мм",
    "кг",
    "г",
    "фунт",
    "фунтов",
    "цельсий",
    "цельсия",
    "фаренгейт",
    "кельвин",
}

_UNIT_WORDS = sorted(ALL_UNITS | _RU_MAP_KEYS, key=len, reverse=True)
_UNIT_TOKEN_RE = r"(?:"
_UNIT_TOKEN_RE += "|".join(re.escape(u) for u in _UNIT_WORDS)
_UNIT_TOKEN_RE += r")"
_UNIT_RE = r"(?:^|\\W)(?P<unit>" + "|".join(re.escape(u) for u in _UNIT_WORDS) + r")(?:$|\\W)"
_NUM_RE = r"[-+]?\d+(?:\.\d+)?"


def _find_units(text: str) -> list[str]:
    units = []
    for match in re.finditer(_UNIT_RE, text):
        units.append(normalize_unit(match.group("unit")))
    return units


def _build_json(value: float, from_unit: str, to_unit: str) -> str:
    payload = {
        "name": "unit_convert",
        "arguments": {
            "value": float(value),
            "from_unit": from_unit,
            "to_unit": to_unit,
            "precision": 2,
        },
    }
    return json.dumps(payload, separators=(",", ":"))


def infer(messages: list[dict]) -> str:
    if not messages:
        return "NO_TOOL"
    text = messages[-1].get("content", "")
    lower = text.lower()

    pattern_num = r"(?P<num>" + _NUM_RE + r")"
    pattern_unit = r"(?:^|\\W)" + _UNIT_TOKEN_RE + r"(?:$|\\W)"

    patterns = [
        r"convert\s+"
        + pattern_num
        + r"\s+(?P<from_unit>"
        + _UNIT_TOKEN_RE
        + r")\s+to\s+(?P<to_unit>"
        + _UNIT_TOKEN_RE
        + r")",
        pattern_num
        + r"\s+(?P<from_unit>"
        + _UNIT_TOKEN_RE
        + r")\s+in\s+(?P<to_unit>"
        + _UNIT_TOKEN_RE
        + r")",
        r"переведи\s+"
        + pattern_num
        + r"\s+(?P<from_unit>"
        + _UNIT_TOKEN_RE
        + r")\s+в\s+(?P<to_unit>"
        + _UNIT_TOKEN_RE
        + r")",
        pattern_num
        + r"\s+(?P<from_unit>"
        + _UNIT_TOKEN_RE
        + r")\s+в\s+(?P<to_unit>"
        + _UNIT_TOKEN_RE
        + r")",
    ]

    for pat in patterns:
        match = re.search(pat, lower)
        if match:
            value = float(match.group("num"))
            from_unit = normalize_unit(match.group("from_unit"))
            to_unit = normalize_unit(match.group("to_unit"))
            if unit_group(from_unit) == unit_group(to_unit) and unit_group(from_unit) is not None:
                return _build_json(value, from_unit, to_unit)

    num_match = re.search(_NUM_RE, lower)
    units = _find_units(lower)
    if num_match and len(units) >= 2:
        from_unit = units[0]
        to_unit = units[1]
        if unit_group(from_unit) == unit_group(to_unit) and unit_group(from_unit) is not None:
            value = float(num_match.group(0))
            return _build_json(value, from_unit, to_unit)

    return "NO_TOOL"
