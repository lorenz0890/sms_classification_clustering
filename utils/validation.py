"""Shared validation helpers for configuration objects."""

from __future__ import annotations

from typing import Any


def require_str(value: Any, name: str) -> str:
    """Validate that a value is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def require_bool(value: Any, name: str) -> bool:
    """Validate that a value is a boolean."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


def require_positive_int(value: Any, name: str) -> int:
    """Validate that a value is a positive integer."""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def require_non_negative_int(value: Any, name: str) -> int:
    """Validate that a value is a non-negative integer."""
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def require_ratio(value: Any, name: str) -> float:
    """Validate that a value is a float ratio in (0, 1)."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a float.")
    value_float = float(value)
    if value_float <= 0.0 or value_float >= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return value_float
