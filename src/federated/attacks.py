from __future__ import annotations

import numpy as np


def flip_labels(y: np.ndarray, fraction: float = 1.0, seed: int = 42) -> np.ndarray:
    if not 0 <= fraction <= 1: raise ValueError("fraction must be in [0,1]")
    result = np.asarray(y, dtype=int).copy()
    ids = np.flatnonzero(np.random.default_rng(seed).random(len(result)) < fraction)
    result[ids] = 1 - result[ids]
    return result


def manipulate_update(update: np.ndarray, mode: str, intensity: float = 5.0) -> np.ndarray:
    if not np.isfinite(update).all() or intensity <= 0: raise ValueError("update must be finite and intensity positive")
    if mode == "scale": return update * intensity
    if mode == "sign": return -update * intensity
    if mode == "slow": return -update * min(intensity, 1.25)
    raise ValueError("attack mode must be scale, sign, or slow")
