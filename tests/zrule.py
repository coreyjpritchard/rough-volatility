"""The z-rule from volsurface ``docs/DESIGN.md`` section 7.

Copied here so a test of ``roughvol`` does not import volsurface. The bounds
are the same three:

- ``max |z|`` below ``Phi^{-1}(1 - 0.001 / (2 n))``.
- The fraction of nodes with ``|z| < 2`` inside the 0.1 and 99.9 percent
  binomial band, when the nodes are independent.
- ``|mean z| < 3.5 / sqrt(n_eff)`` when the nodes are independent.

Nodes that share paths keep the max bound and check the mean on R
independent replications: ``|mean(m)| < 3.5 * sd(m) / sqrt(R)``, ``ddof=1``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom, norm

_WITHIN_2 = float(2.0 * norm.cdf(2.0) - 1.0)


def max_abs_bound(n: int) -> float:
    """``Phi^{-1}(1 - 0.001 / (2 n))``."""
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    return float(norm.ppf(1.0 - 0.001 / (2.0 * n)))


def check_z(
    z: np.ndarray,
    *,
    n_eff: float | None = None,
    mean: bool = True,
    fraction: bool = True,
) -> str | None:
    """None if ``z`` meets the rule, else a sentence naming the failure."""
    scores = np.asarray(z, dtype=float).reshape(-1)
    if scores.size == 0:
        raise ValueError("z is empty")
    if not np.all(np.isfinite(scores)):
        return "z-rule: a node is not finite"
    n = int(scores.size)
    max_abs = float(np.max(np.abs(scores)))
    cap = max_abs_bound(n)
    if max_abs >= cap:
        return f"z-rule max |z| is {max_abs:.4f}, bound {cap:.4f} at n={n}"
    if mean:
        effective = float(n if n_eff is None else n_eff)
        bound = 3.5 / math.sqrt(effective)
        centre = float(np.mean(scores))
        if abs(centre) >= bound:
            return (
                f"z-rule mean z is {centre:.4f}, bound {bound:.4f} at n_eff={effective:g} (n={n})"
            )
    if fraction:
        frac = float(np.mean(np.abs(scores) < 2.0))
        lo = float(binom.ppf(0.001, n, _WITHIN_2)) / n
        hi = float(binom.ppf(0.999, n, _WITHIN_2)) / n
        if frac < lo or frac > hi:
            return f"z-rule fraction |z|<2 is {frac:.4f}, band [{lo:.4f}, {hi:.4f}] at n={n}"
    return None


def check_replication_means(means: np.ndarray) -> str | None:
    """``|mean| < 3.5 * sd / sqrt(R)`` for R >= 2 replications of a shared-path grid."""
    sample = np.asarray(means, dtype=float).reshape(-1)
    if sample.size < 2:
        raise ValueError(f"need at least 2 replications, got {sample.size}")
    if not np.all(np.isfinite(sample)):
        return "replication means are not all finite"
    n_rep = int(sample.size)
    centre = float(np.mean(sample))
    sd = float(np.std(sample, ddof=1))
    if sd == 0.0:
        if centre == 0.0:
            return None
        return f"replication mean is {centre:.4g} with sample sd 0 over R={n_rep}"
    bound = 3.5 * sd / math.sqrt(n_rep)
    if abs(centre) >= bound:
        return (
            f"replication mean is {centre:.4f}, bound {bound:.4f} "
            f"(3.5 * sd/sqrt(R), sd={sd:.4f}, R={n_rep})"
        )
    return None
