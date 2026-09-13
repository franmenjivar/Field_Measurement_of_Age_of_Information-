"""Tactical conflict detector.

One class, two instantiations: one fed ground-truth states, one fed the
surveillance store.  They must be *the same* detector, otherwise a difference
in outcome could come from the detector rather than from information age --
which would invalidate the whole study (research plan, section 14).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from ..core.schema import State
from .cpa import CPAResult, compute_cpa


@dataclass(frozen=True)
class ConflictCriteria:
    """Separation standard and tactical horizon.

    Defaults follow the research plan (30 m horizontal, 30 s horizon).  These
    are *assumptions of the experiment*, not regulatory values, and the paper
    must say so.
    """

    horizontal_separation_m: float = 30.0
    vertical_separation_m: float = 15.0
    horizon_s: float = 30.0
    #: ignore pairs further apart than this (pure speed optimisation)
    screen_range_m: float = 5000.0

    def violated_by(self, d_horiz_m: float, d_vert_m: float) -> bool:
        return (
            d_horiz_m < self.horizontal_separation_m
            and d_vert_m < self.vertical_separation_m
        )


@dataclass(frozen=True, slots=True)
class PairAssessment:
    """Detector output for one pair at one decision time."""

    uas_a: str
    uas_b: str
    t_s: float
    conflict: bool
    ttc_s: float
    d_cpa_horiz_m: float
    d_cpa_vert_m: float
    range_m: float
    closing_speed_mps: float


class ConflictDetector:
    """Pairwise constant-velocity conflict prediction over a set of states."""

    def __init__(self, criteria: ConflictCriteria | None = None) -> None:
        self.criteria = criteria or ConflictCriteria()

    def assess_pair(self, a: State, b: State, t_s: float) -> PairAssessment:
        cpa: CPAResult = compute_cpa(a, b, horizon_s=self.criteria.horizon_s)
        conflict = (
            not cpa.diverging
            and self.criteria.violated_by(cpa.d_cpa_horiz_m, cpa.d_cpa_vert_m)
        )
        return PairAssessment(
            uas_a=a.uas_id,
            uas_b=b.uas_id,
            t_s=float(t_s),
            conflict=conflict,
            # TTC is only meaningful for a predicted conflict; otherwise there
            # is no conflict to be "time to".
            ttc_s=cpa.t_cpa_s if conflict else np.inf,
            d_cpa_horiz_m=cpa.d_cpa_horiz_m,
            d_cpa_vert_m=cpa.d_cpa_vert_m,
            range_m=cpa.range_now_m,
            closing_speed_mps=cpa.closing_speed_mps,
        )

    def assess(self, states: Mapping[str, State], t_s: float) -> list[PairAssessment]:
        """Every aircraft pair in ``states`` at one decision time."""
        out = []
        for a, b in combinations(sorted(states), 2):
            sa, sb = states[a], states[b]
            if (
                abs(sa.x_m - sb.x_m) > self.criteria.screen_range_m
                or abs(sa.y_m - sb.y_m) > self.criteria.screen_range_m
            ):
                continue
            out.append(self.assess_pair(sa, sb, t_s))
        return out

    def conflicts(self, states: Mapping[str, State], t_s: float) -> list[PairAssessment]:
        return [p for p in self.assess(states, t_s) if p.conflict]

    @staticmethod
    def to_frame(assessments: list[PairAssessment]) -> pd.DataFrame:
        if not assessments:
            return pd.DataFrame(
                columns=[
                    "uas_a", "uas_b", "t_s", "conflict", "ttc_s",
                    "d_cpa_horiz_m", "d_cpa_vert_m", "range_m", "closing_speed_mps",
                ]
            )
        return pd.DataFrame([asdict(a) for a in assessments])
