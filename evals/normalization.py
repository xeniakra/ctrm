"""Small deterministic normalisations used only for evaluation comparison."""

from __future__ import annotations

from datetime import date, datetime
from math import isfinite
from typing import Any


def normalize_value(value: Any) -> Any:
    """Canonicalise serialization-only differences without changing semantics."""

    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: normalize_value(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    return value


def values_equal(expected: Any, actual: Any) -> bool:
    return normalize_value(expected) == normalize_value(actual)
