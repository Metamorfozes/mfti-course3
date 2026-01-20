from __future__ import annotations

from tool_schema import LENGTH, MASS, TEMP, normalize_unit, unit_group

_LENGTH_TO_M = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
    "km": 1000.0,
    "inch": 0.0254,
    "ft": 0.3048,
    "yd": 0.9144,
    "mile": 1609.344,
}

_MASS_TO_KG = {
    "g": 0.001,
    "kg": 1.0,
    "oz": 0.028349523125,
    "lb": 0.45359237,
}


def _temp_to_c(value: float, unit: str) -> float:
    if unit == "c":
        return value
    if unit == "f":
        return (value - 32.0) * 5.0 / 9.0
    if unit == "k":
        return value - 273.15
    raise ValueError("unknown temperature unit")


def _temp_from_c(value_c: float, unit: str) -> float:
    if unit == "c":
        return value_c
    if unit == "f":
        return (value_c * 9.0 / 5.0) + 32.0
    if unit == "k":
        return value_c + 273.15
    raise ValueError("unknown temperature unit")


def unit_convert(value: float, from_unit: str, to_unit: str) -> float:
    from_unit = normalize_unit(from_unit)
    to_unit = normalize_unit(to_unit)

    from_group = unit_group(from_unit)
    to_group = unit_group(to_unit)
    if from_group is None or to_group is None:
        raise ValueError("unknown unit")
    if from_group != to_group:
        raise ValueError("incompatible unit groups")

    if from_group == "length":
        value_m = value * _LENGTH_TO_M[from_unit]
        return value_m / _LENGTH_TO_M[to_unit]
    if from_group == "mass":
        value_kg = value * _MASS_TO_KG[from_unit]
        return value_kg / _MASS_TO_KG[to_unit]
    if from_group == "temp":
        value_c = _temp_to_c(value, from_unit)
        return _temp_from_c(value_c, to_unit)

    raise ValueError("unknown unit group")
