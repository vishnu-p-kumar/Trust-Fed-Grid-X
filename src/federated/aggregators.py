from __future__ import annotations

import numpy as np


def fedavg(updates: list[np.ndarray], sizes: list[int], weights: np.ndarray | None = None) -> np.ndarray:
    if not updates or len(updates) != len(sizes): raise ValueError("updates and sizes must be non-empty and aligned")
    sample_sizes=np.asarray(sizes,dtype=float)
    if not np.isfinite(sample_sizes).all() or (sample_sizes <= 0).any(): raise ValueError("client sizes must be positive and finite")
    if any(not np.isfinite(u).all() for u in updates): raise ValueError("updates must be finite")
    w = np.asarray(weights if weights is not None else sample_sizes, dtype=float)
    if len(w) != len(updates) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0: raise ValueError("invalid aggregation weights")
    shapes = {u.shape for u in updates}
    if len(shapes) != 1: raise ValueError("update shapes differ")
    w /= w.max()
    w /= w.sum()
    return np.sum(np.stack(updates) * w.reshape((-1,) + (1,) * updates[0].ndim), axis=0)


def trimmed_mean(updates: list[np.ndarray], trim_fraction: float = 0.1) -> np.ndarray:
    if not updates or not 0 <= trim_fraction < 0.5: raise ValueError("invalid updates or trim fraction")
    stack = np.stack(updates)
    if not np.isfinite(stack).all(): raise ValueError("updates must be finite")
    trim = min(int(np.ceil(len(updates) * trim_fraction)), max(0, (len(updates) - 1) // 2))
    ordered = np.sort(stack, axis=0)
    return ordered[trim:len(updates)-trim or None].mean(axis=0)


def trust_weighted(updates: list[np.ndarray], sizes: list[int], trust: list[float]) -> np.ndarray:
    scores = np.clip(np.asarray(trust, dtype=float), 0.0, 1.0)
    sample_sizes=np.asarray(sizes,dtype=float)
    if len(scores) != len(updates) or len(sample_sizes)!=len(updates): raise ValueError("trust scores and sizes must match updates")
    if not np.isfinite(scores).all() or not np.isfinite(sample_sizes).all() or (sample_sizes<=0).any():
        raise ValueError("trust scores must be finite and client sizes positive and finite")
    trusted_sizes = scores * (sample_sizes/sample_sizes.max())
    if trusted_sizes.sum() <= 0:
        return fedavg(updates, sizes)
    return fedavg(updates, sizes, weights=trusted_sizes)
