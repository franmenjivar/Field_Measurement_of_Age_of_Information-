"""Sampling a ground-truth trajectory into a delayed telemetry stream.

The generator is the only component that reads ground truth and writes
messages.  Downstream of it, nothing can see the truth layer -- that separation
is the study's central scientific control (README.md, "The mechanism").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.schema import State
from .message import TelemetryMessage
from .network_model import NetworkModel, TelemetryConfig


class TrajectorySampler:
    """Interpolates one aircraft's ground-truth trajectory at arbitrary times.

    Linear interpolation between recorded samples: the reference streams here
    are 5-62 Hz, so a higher-order scheme would add no information while
    inventing dynamics the aircraft did not fly.
    """

    def __init__(self, traj: pd.DataFrame) -> None:
        if traj.empty:
            raise ValueError("empty trajectory")
        t = traj["t_s"].to_numpy(dtype=float)
        if not np.all(np.diff(t) > 0):
            order = np.argsort(t, kind="stable")
            traj = traj.iloc[order]
            t = traj["t_s"].to_numpy(dtype=float)
        self.uas_id = str(traj["uas_id"].iloc[0])
        self.t = t
        self._cols = {
            c: traj[c].to_numpy(dtype=float)
            for c in ("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps")
        }

    @property
    def t_start(self) -> float:
        return float(self.t[0])

    @property
    def t_end(self) -> float:
        return float(self.t[-1])

    def at(self, t_s: float) -> State:
        """Ground-truth state at ``t_s`` (clamped to the recorded interval)."""
        v = {c: float(np.interp(t_s, self.t, arr)) for c, arr in self._cols.items()}
        return State(uas_id=self.uas_id, t_s=float(t_s), **v)

    def at_many(self, times: np.ndarray) -> pd.DataFrame:
        times = np.asarray(times, dtype=float)
        out = {c: np.interp(times, self.t, arr) for c, arr in self._cols.items()}
        out["t_s"] = times
        out["uas_id"] = self.uas_id
        return pd.DataFrame(out)


def generate_stream(
    sampler: TrajectorySampler,
    config: TelemetryConfig,
    rng: np.random.Generator,
    *,
    t_start: float | None = None,
    t_end: float | None = None,
) -> list[TelemetryMessage]:
    """Sample a trajectory at ``config.update_rate_hz`` and delay each message.

    Returns messages sorted by arrival time -- which is not necessarily
    generation order once jitter is large enough to reorder them.
    """
    t0 = sampler.t_start if t_start is None else float(t_start)
    t1 = sampler.t_end if t_end is None else float(t_end)
    if t1 <= t0:
        return []

    phase = rng.random() * config.interval_s if config.randomise_phase else 0.0
    gen_times = np.arange(t0 + phase, t1, config.interval_s)
    if gen_times.size == 0:
        return []

    net = NetworkModel(config, rng)
    delays = net.delay_s(gen_times.size)
    lost = net.is_lost(gen_times.size)

    messages: list[TelemetryMessage] = []
    for seq, (tg, dl, drop) in enumerate(zip(gen_times, delays, lost, strict=True)):
        if drop:
            continue
        messages.append(
            TelemetryMessage(
                uas_id=sampler.uas_id,
                t_gen_s=float(tg),
                t_arrive_s=float(tg + dl),
                state=sampler.at(float(tg)),
                seq=seq,
            )
        )

    messages.sort(key=lambda m: m.t_arrive_s)
    return messages
