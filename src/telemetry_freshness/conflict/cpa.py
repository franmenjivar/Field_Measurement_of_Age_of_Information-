"""Closest point of approach and time-to-conflict geometry.

All quantities are in metres and seconds in a local ENU frame; the horizontal
and vertical channels are treated separately because UTM separation standards
are themselves a cylinder, not a sphere (research plan, section 14).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.schema import State


@dataclass(frozen=True, slots=True)
class CPAResult:
    """Closest-approach geometry for one aircraft pair at one decision time."""

    t_cpa_s: float
    d_cpa_m: float
    d_cpa_horiz_m: float
    d_cpa_vert_m: float
    range_now_m: float
    closing_speed_mps: float
    diverging: bool

    @property
    def is_converging(self) -> bool:
        return not self.diverging


def relative_state(a: State, b: State) -> tuple[np.ndarray, np.ndarray]:
    """Relative position and velocity of ``b`` as seen from ``a``."""
    return b.position - a.position, b.velocity - a.velocity


def compute_cpa(a: State, b: State, *, horizon_s: float = 30.0) -> CPAResult:
    """Constant-velocity closest point of approach within ``horizon_s``.

    ``t_cpa`` is clamped to ``[0, horizon_s]``: a CPA in the past or beyond the
    tactical horizon is not actionable, and letting it run free produces
    nonsense TTC values for diverging pairs.
    """
    r, v = relative_state(a, b)
    vv = float(v @ v)
    range_now = float(np.linalg.norm(r))

    if vv <= 1e-12:
        # Co-moving pair: separation never changes.
        t_star = 0.0
        diverging = True
    else:
        t_star = -float(r @ v) / vv
        diverging = t_star <= 0.0
        t_star = float(np.clip(t_star, 0.0, horizon_s))

    d = r + v * t_star
    closing = -float(r @ v) / range_now if range_now > 1e-9 else 0.0

    return CPAResult(
        t_cpa_s=t_star,
        d_cpa_m=float(np.linalg.norm(d)),
        d_cpa_horiz_m=float(np.hypot(d[0], d[1])),
        d_cpa_vert_m=float(abs(d[2])),
        range_now_m=range_now,
        closing_speed_mps=closing,
        diverging=diverging,
    )


def cpa_vectorised(
    pa: np.ndarray, va: np.ndarray, pb: np.ndarray, vb: np.ndarray, horizon_s: float = 30.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched CPA for arrays of pairs -- used by the density experiments.

    Returns ``(t_cpa, d_cpa_horizontal, d_cpa_vertical)``.
    """
    r = np.asarray(pb, dtype=float) - np.asarray(pa, dtype=float)
    v = np.asarray(vb, dtype=float) - np.asarray(va, dtype=float)
    vv = np.einsum("ij,ij->i", v, v)
    rv = np.einsum("ij,ij->i", r, v)

    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(vv > 1e-12, -rv / vv, 0.0)
    t = np.clip(t, 0.0, horizon_s)

    d = r + v * t[:, None]
    return t, np.hypot(d[:, 0], d[:, 1]), np.abs(d[:, 2])
