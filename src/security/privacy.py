"""Research-grade privacy primitives for federated-learning experiments.

These functions are useful for controlled experiments, not a production secure-
aggregation protocol. Production use requires authenticated key exchange, dropout
handling, replay protection, a vetted cryptographic library, and an audited protocol.
"""
from __future__ import annotations

import hashlib
import hmac
import math
from typing import Mapping

import numpy as np


def clip_update(update: np.ndarray, max_norm: float) -> np.ndarray:
    """L2-clip a client update to max_norm; rejects non-finite inputs."""
    values = np.asarray(update, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("update must be non-empty and finite")
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be positive and finite")
    norm = float(np.linalg.norm(values.ravel(), ord=2))
    return values.copy() if norm <= max_norm else values * (max_norm / norm)


def gaussian_dp_aggregate(
    updates: list[np.ndarray], *, max_norm: float, noise_multiplier: float,
    delta: float = 1e-5, seed: int | None = None,
) -> dict[str, object]:
    """Clip client updates and add Gaussian noise to their mean.

    Uses replacement adjacency sensitivity 2*C/n for the mean of n clipped updates.
    The returned epsilon is the classical sufficient bound for the Gaussian mechanism
    (valid in its usual epsilon < 1 regime); it is not a composed multi-round accountant.
    """
    if not updates:
        raise ValueError("at least one client update is required")
    if not math.isfinite(noise_multiplier) or noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive and finite")
    if not 0 < delta < 1:
        raise ValueError("delta must be between zero and one")
    arrays = [np.asarray(item, dtype=float) for item in updates]
    if any(item.shape != arrays[0].shape for item in arrays):
        raise ValueError("client update shapes differ")
    clipped = [clip_update(item, max_norm) for item in arrays]
    mean = np.mean(np.stack(clipped), axis=0)
    sensitivity = 2.0 * max_norm / len(clipped)
    sigma = noise_multiplier * sensitivity
    rng = np.random.default_rng(seed)
    private_mean = mean + rng.normal(0.0, sigma, size=mean.shape)
    epsilon_bound = (sensitivity / sigma) * math.sqrt(2.0 * math.log(1.25 / delta))
    return {
        "aggregate": private_mean,
        "privacy": {
            "mechanism": "client-update clipping + Gaussian noise",
            "adjacency": "replacement of one client update",
            "client_count": len(clipped), "max_l2_norm": float(max_norm),
            "noise_multiplier": float(noise_multiplier), "noise_std": float(sigma),
            "delta": float(delta), "epsilon_single_release_bound": float(epsilon_bound),
            "composed_rounds_accounted": False,
            "warning": "Single-release bound only. Use a vetted RDP/PRV accountant for multi-round training.",
        },
    }


def _pairwise_mask(secret: bytes, context: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Deterministic mask stream for a local protocol simulation; not a crypto library."""
    count = int(np.prod(shape))
    seed = int.from_bytes(hmac.new(secret, context, hashlib.sha256).digest()[:8], "big")
    return np.random.default_rng(seed).normal(0.0, 1.0, size=count).reshape(shape)


def simulated_secure_sum(updates: Mapping[str, np.ndarray], pairwise_secrets: Mapping[tuple[str, str], bytes]) -> np.ndarray:
    """Simulate pairwise additive masks that cancel in the sum.

    All client pairs must have a pre-shared secret. This only demonstrates cancellation;
    it does NOT protect against collusion, network attackers, client dropout or bad key
    management and must not be represented as production secure aggregation.
    """
    names = sorted(updates)
    if len(names) < 2:
        raise ValueError("at least two clients are required")
    arrays = {name: np.asarray(updates[name], dtype=float) for name in names}
    shape = arrays[names[0]].shape
    if any(value.shape != shape or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("updates must be finite and have identical shapes")
    masked = {name: value.copy() for name, value in arrays.items()}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            secret = pairwise_secrets.get((left, right)) or pairwise_secrets.get((right, left))
            if not isinstance(secret, bytes) or len(secret) < 16:
                raise ValueError(f"missing pairwise secret for clients {left} and {right}")
            context = f"{left}|{right}|{shape}".encode()
            mask = _pairwise_mask(secret, context, shape)
            masked[left] += mask
            masked[right] -= mask
    return np.sum(np.stack([masked[name] for name in names]), axis=0)
