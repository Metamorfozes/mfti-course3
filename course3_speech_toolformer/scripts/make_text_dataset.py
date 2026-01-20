from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tool_schema import normalize_unit

random.seed(42)

EN_LENGTH = ["mm", "cm", "m", "km", "inch", "ft", "yd", "mile"]
EN_MASS = ["g", "kg", "oz", "lb"]
EN_TEMP = ["c", "f", "k"]

RU_LENGTH = ["мм", "см", "м", "км"]
RU_MASS = ["г", "кг", "фунт"]
RU_TEMP = ["цельсия", "фаренгейт", "кельвин"]

EN_TEMPLATES = [
    "convert {value} {from_unit} to {to_unit}",
    "{value} {from_unit} in {to_unit}",
    "please convert {value} {from_unit} to {to_unit}",
    "what is {value} {from_unit} in {to_unit}",
]

RU_TEMPLATES = [
    "переведи {value} {from_unit} в {to_unit}",
    "{value} {from_unit} в {to_unit}",
    "пожалуйста, переведи {value} {from_unit} в {to_unit}",
    "сколько будет {value} {from_unit} в {to_unit}",
]

EN_NO_TOOL = [
    "tell me about the {unit}",
    "I have {value} {unit} of apples",
    "the distance is {value} {unit} today",
    "temperature is {value} {unit} outside",
]

RU_NO_TOOL = [
    "расскажи про единицу {unit}",
    "у меня {value} {unit} яблок",
    "сегодня {value} {unit} на улице",
    "вес посылки {value} {unit}",
]


def _random_value() -> float:
    if random.random() < 0.5:
        return float(random.randint(1, 500))
    return float(round(random.uniform(1, 500), 1))


def _pick_units(lang: str) -> tuple[str, str]:
    if lang == "en":
        groups = [EN_LENGTH, EN_MASS, EN_TEMP]
    else:
        groups = [RU_LENGTH, RU_MASS, RU_TEMP]
    group = random.choice(groups)
    from_unit = random.choice(group)
    to_unit = random.choice(group)
    while to_unit == from_unit:
        to_unit = random.choice(group)
    return from_unit, to_unit


def _make_tool_sample(lang: str) -> dict:
    value = _random_value()
    from_unit, to_unit = _pick_units(lang)
    if lang == "en":
        template = random.choice(EN_TEMPLATES)
    else:
        template = random.choice(RU_TEMPLATES)
    text = template.format(value=value, from_unit=from_unit, to_unit=to_unit)

    expected_tool_call = {
        "name": "unit_convert",
        "arguments": {
            "value": float(value),
            "from_unit": normalize_unit(from_unit),
            "to_unit": normalize_unit(to_unit),
            "precision": 2,
        },
    }
    return {
        "lang": lang,
        "text": text,
        "tool_required": True,
        "expected_tool_call": expected_tool_call,
    }


def _make_no_tool_sample(lang: str) -> dict:
    value = _random_value()
    if lang == "en":
        unit = random.choice(EN_LENGTH + EN_MASS + EN_TEMP)
        template = random.choice(EN_NO_TOOL)
    else:
        unit = random.choice(RU_LENGTH + RU_MASS + RU_TEMP)
        template = random.choice(RU_NO_TOOL)
    text = template.format(value=value, unit=unit)
    return {
        "lang": lang,
        "text": text,
        "tool_required": False,
        "expected_tool_call": None,
    }


def main() -> None:
    samples = []
    for lang in ["en", "ru"]:
        for _ in range(100):
            samples.append(_make_tool_sample(lang))
    for lang in ["en", "ru"]:
        for _ in range(25):
            samples.append(_make_no_tool_sample(lang))

    random.shuffle(samples)
    for idx, sample in enumerate(samples, start=1):
        sample["id"] = idx

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "text_dataset.json"))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
