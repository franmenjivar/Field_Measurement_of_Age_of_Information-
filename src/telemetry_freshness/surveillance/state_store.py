"""The UTM's view of the airspace: latest arrived telemetry, nothing more.

This object is what the conflict detector is given.  It holds no reference to
any trajectory, sampler or simulator, so the detector *cannot* read ground
truth even by accident (README.md, "The mechanism"; enforced by tests/test_isolation.py).
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable

import numpy as np

from ..core.schema import State
from ..telemetry.message import TelemetryMessage


class SurveillanceStateStore:
    """Latest-value-wins store keyed by aircraft.

    Parameters
    ----------
    keep_newest_generation
        When true (default) a message that arrives *after* a fresher one is
        discarded rather than overwriting it -- the behaviour of a store that
        timestamps its records.  Set false to model a naive store that simply
        takes whatever arrived last, which is a meaningfully worse system and
        worth reporting as a comparison.
    """

    def __init__(self, *, keep_newest_generation: bool = True) -> None:
        self.keep_newest_generation = keep_newest_generation
        self._arrivals: dict[str, list[TelemetryMessage]] = {}
        self._arrival_times: dict[str, list[float]] = {}
        #: index of the freshest-by-generation message among arrivals[:i+1]
        self._freshest: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ ingest

    def ingest(self, messages: Iterable[TelemetryMessage]) -> None:
        """Load messages into the store, indexed by arrival time.

        A running argmax over generation time is precomputed here so that a
        query is a binary search plus one lookup rather than a scan; the density
        sweeps issue millions of queries.
        """
        for m in messages:
            self._arrivals.setdefault(m.uas_id, []).append(m)
        for uid, msgs in self._arrivals.items():
            msgs.sort(key=lambda m: m.t_arrive_s)
            self._arrival_times[uid] = [m.t_arrive_s for m in msgs]
            best: list[int] = []
            bi = 0
            for i, m in enumerate(msgs):
                if i and m.t_gen_s > msgs[bi].t_gen_s:
                    bi = i
                elif i == 0:
                    bi = 0
                best.append(bi)
            self._freshest[uid] = best

    @property
    def aircraft(self) -> list[str]:
        return sorted(self._arrivals)

    # ------------------------------------------------------------------- query

    def latest(self, uas_id: str, t_s: float) -> TelemetryMessage | None:
        """Freshest message for ``uas_id`` that had arrived by ``t_s``.

        Returns ``None`` when nothing has arrived yet -- the aircraft is simply
        not trackable at that moment.  Callers must treat that as a distinct
        outcome, never as an age of zero.
        """
        msgs = self._arrivals.get(uas_id)
        if not msgs:
            return None
        i = bisect.bisect_right(self._arrival_times[uas_id], t_s)
        if i == 0:
            return None
        if not self.keep_newest_generation:
            return msgs[i - 1]
        # Among everything that has arrived, take the newest state, not the
        # newest arrival: a reordered stale message must not displace it.
        return msgs[self._freshest[uas_id][i - 1]]

    def observed_state(self, uas_id: str, t_s: float) -> State | None:
        """State the UTM believes ``uas_id`` is in, as of ``t_s``."""
        m = self.latest(uas_id, t_s)
        return None if m is None else m.state

    def aoi(self, uas_id: str, t_s: float) -> float:
        """Age of Information at ``t_s``; ``nan`` when nothing has arrived."""
        m = self.latest(uas_id, t_s)
        return np.nan if m is None else t_s - m.t_gen_s

    def tracked(self, t_s: float) -> list[str]:
        """Aircraft with at least one arrived message at ``t_s``."""
        return [u for u in self.aircraft if self.latest(u, t_s) is not None]

    def snapshot(self, t_s: float) -> dict[str, State]:
        """Every observed state the UTM holds at ``t_s``."""
        out = {}
        for u in self.aircraft:
            s = self.observed_state(u, t_s)
            if s is not None:
                out[u] = s
        return out
