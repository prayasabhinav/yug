# Copyright 2026 Prayas Abhinav
# SPDX-License-Identifier: Apache-2.0
"""Which way a series is going: up, down or flat, decided by code from Yug's band.

The model gives only its quantile band. This file turns the band into one of
three answers per step, under a key a person writes::

    {"margin": 0.5, "lower": 0.1, "upper": 0.9}

``margin`` is in the series' own units; ``lower`` and ``upper`` are quantile
levels the checkpoint produces. A step is up (1) when the whole band sits more
than ``margin`` above the last observed value, down (-1) when the whole band
sits more than ``margin`` below it, and flat (0) otherwise, which includes a
band too wide to clear the margin either way.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import QuantileForecast

__all__ = ["UP", "DOWN", "FLAT", "read_key", "direction"]

UP, DOWN, FLAT = 1, -1, 0


def read_key(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def direction(forecast: QuantileForecast, last_value, key: dict) -> np.ndarray:
    """1, -1 or 0 per step: ``(horizon,)``, or ``(n_series, horizon)`` with one
    last value per series."""
    low, high = forecast.quantile(key["lower"]), forecast.quantile(key["upper"])
    last = np.asarray(last_value, dtype=np.float32)
    if last.ndim:
        last = last[:, None]
    out = np.full(low.shape, FLAT, dtype=np.int8)
    out[low > last + key["margin"]] = UP
    out[high < last - key["margin"]] = DOWN
    return out
