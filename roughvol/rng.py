"""Seed handling for a model run.

An int and a ``numpy.random.SeedSequence`` both go through
``np.random.default_rng``. A run spawns children of a copy of the parent
seed. ``SeedSequence.spawn`` advances ``n_children_spawned`` on the object
it is called on, so the copy leaves the caller unmoved: two calls with the
same seed both draw child 0.
"""

from __future__ import annotations

import numpy as np


def as_seed_sequence(seed: int | np.random.SeedSequence) -> np.random.SeedSequence:
    """A fresh ``SeedSequence``. An int becomes one. A sequence is copied."""
    if isinstance(seed, np.random.SeedSequence):
        return np.random.SeedSequence(
            seed.entropy, spawn_key=seed.spawn_key, pool_size=seed.pool_size
        )
    return np.random.SeedSequence(int(seed))


def generator(seed: int | np.random.SeedSequence) -> np.random.Generator:
    """``np.random.default_rng`` on :func:`as_seed_sequence`."""
    return np.random.default_rng(as_seed_sequence(seed))


def spawn(seed: int | np.random.SeedSequence, n: int) -> list[np.random.SeedSequence]:
    """``n`` independent children, in order. One child per stream."""
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    return list(as_seed_sequence(seed).spawn(n))
