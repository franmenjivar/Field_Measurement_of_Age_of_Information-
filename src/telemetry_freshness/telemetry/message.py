"""Telemetry message: the only object that crosses from truth to observation.

Keeping generation and arrival time on the same immutable record is what makes
Age of Information measurable rather than assumed (research plan, section 6).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.schema import State


@dataclass(frozen=True, slots=True)
class TelemetryMessage:
    """One telemetry report in flight between an aircraft and the UTM.

    Attributes
    ----------
    t_gen_s
        When the aircraft sampled this state.  This is the timestamp AoI is
        measured against -- never the arrival time.
    t_arrive_s
        When the UTM could first use it (transport + processing delay).
    state
        The aircraft state as sampled at ``t_gen_s``.
    """

    uas_id: str
    t_gen_s: float
    t_arrive_s: float
    state: State
    seq: int = 0

    @property
    def transit_s(self) -> float:
        return self.t_arrive_s - self.t_gen_s

    def age_at(self, t_s: float) -> float:
        """Age of the information carried by this message at decision time."""
        return t_s - self.t_gen_s
