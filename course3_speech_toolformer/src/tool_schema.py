from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

LENGTH = {"mm", "cm", "m", "km", "inch", "ft", "yd", "mile"}
MASS = {"g", "kg", "oz", "lb"}
TEMP = {"c", "f", "k"}
ALL_UNITS = LENGTH | MASS | TEMP

_RU_MAP = {
    "км": "km",
    "м": "m",
    "см": "cm",
    "мм": "mm",
    "кг": "kg",
    "г": "g",
    "фунт": "lb",
    "фунтов": "lb",
    "цельсий": "c",
    "цельсия": "c",
    "фаренгейт": "f",
    "кельвин": "k",
}


def normalize_unit(u: str) -> str:
    s = u.strip().lower().replace(" ", "_")
    return _RU_MAP.get(s, s)


def unit_group(unit: str) -> str | None:
    if unit in LENGTH:
        return "length"
    if unit in MASS:
        return "mass"
    if unit in TEMP:
        return "temp"
    return None


class Args(BaseModel):
    value: float
    from_unit: str
    to_unit: str
    precision: int = 2

    @field_validator("from_unit", "to_unit", mode="before")
    @classmethod
    def _normalize_units(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("unit must be a string")
        return normalize_unit(v)

    @model_validator(mode="after")
    def _check_groups(self) -> "Args":
        from_group = unit_group(self.from_unit)
        to_group = unit_group(self.to_unit)
        if from_group is None or to_group is None:
            raise ValueError("unknown unit")
        if from_group != to_group:
            raise ValueError("incompatible unit groups")
        return self


class ToolCall(BaseModel):
    name: Literal["unit_convert"]
    arguments: Args
