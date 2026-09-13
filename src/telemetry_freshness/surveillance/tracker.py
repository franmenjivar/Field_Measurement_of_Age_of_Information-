"""A constant-velocity Kalman tracker over the surveillance stream.

The extrapolation ablation initially compared "hold the last state" against
plain constant-velocity dead reckoning using the velocity the source reported.
That is a weak comparator: no operational surveillance system extrapolates from
a raw reported velocity.  A real tracker filters, and a filtered velocity is
smoother and less prone to the occasional badly-wrong heading that made naive
dead reckoning fail.

This module supplies that missing arm.  It is a standard constant-velocity
Kalman filter driven by **position measurements only**, so the velocity it
extrapolates along is one the tracker inferred rather than one the aircraft
claimed.  Position-only is the honest choice here: if the filter were fed the
reported velocity it would inherit exactly the error the ablation is testing.

State is ``[x, y, vx, vy]``; the process model is constant velocity with white
acceleration noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.schema import State
from ..telemetry.message import TelemetryMessage


@dataclass(frozen=True)
class TrackerConfig:
    """Filter tuning.

    ``process_accel_noise`` is the standard deviation of the unmodelled
    acceleration the target may exhibit; ``measurement_noise_m`` is the assumed
    position measurement error.  Defaults are set for small UAS: a few m/s^2 of
    manoeuvre and a GNSS-grade position.
    """

    process_accel_noise_mps2: float = 2.0
    measurement_noise_m: float = 5.0
    #: initial velocity uncertainty before the filter has seen motion
    initial_velocity_sigma_mps: float = 10.0
    #: a gap longer than this restarts the track rather than coasting through it
    max_coast_s: float = 10.0
    #: "position" uses position reports only; "position_velocity" also ingests
    #: the reported velocity as a measurement
    measurement_mode: str = "position"
    #: assumed 1-sigma error on the reported velocity, used only when fusing
    velocity_noise_mps: float = 1.5

    def __post_init__(self) -> None:
        if self.measurement_mode not in {"position", "position_velocity"}:
            raise ValueError("measurement_mode must be 'position' or 'position_velocity'")


class ConstantVelocityTracker:
    """Per-aircraft constant-velocity Kalman filter over position reports."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()

    def _predict(self, x, P, dt, q):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        # Continuous white-acceleration process noise, discretised.
        dt2, dt3, dt4 = dt * dt, dt**3, dt**4
        Q = q * np.array(
            [[dt4 / 4, 0, dt3 / 2, 0],
             [0, dt4 / 4, 0, dt3 / 2],
             [dt3 / 2, 0, dt2, 0],
             [0, dt3 / 2, 0, dt2]], float)
        return F @ x, F @ P @ F.T + Q

    def filter_track(self, times_s, x_m, y_m, vx_mps=None, vy_mps=None) -> pd.DataFrame:
        """Run the filter over one aircraft's reports.

        In ``position`` mode only the position is measured and velocity is
        inferred by the filter.  In ``position_velocity`` mode the reported
        velocity is ingested as a measurement too -- the arm that answers
        whether fusing the reported velocity, rather than ignoring it, rescues
        dead-reckoning compensation.

        A gap longer than ``max_coast_s`` reinitialises the track, which is what
        a real tracker does rather than coasting indefinitely through a dropout.
        """
        cfg = self.config
        fuse = cfg.measurement_mode == "position_velocity"
        if fuse and (vx_mps is None or vy_mps is None):
            raise ValueError("position_velocity mode needs vx_mps and vy_mps")

        t = np.asarray(times_s, float)
        zx = np.asarray(x_m, float)
        zy = np.asarray(y_m, float)
        order = np.argsort(t)
        t, zx, zy = t[order], zx[order], zy[order]
        if fuse:
            zvx = np.asarray(vx_mps, float)[order]
            zvy = np.asarray(vy_mps, float)[order]

        q = cfg.process_accel_noise_mps2 ** 2
        if fuse:
            R = np.diag([cfg.measurement_noise_m ** 2, cfg.measurement_noise_m ** 2,
                         cfg.velocity_noise_mps ** 2, cfg.velocity_noise_mps ** 2])
            H = np.eye(4)
        else:
            R = np.eye(2) * cfg.measurement_noise_m ** 2
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)

        out = np.full((t.size, 4), np.nan)
        x = P = None

        for i in range(t.size):
            if x is None or (i > 0 and t[i] - t[i - 1] > cfg.max_coast_s):
                v0 = (zvx[i], zvy[i]) if fuse else (0.0, 0.0)
                x = np.array([zx[i], zy[i], v0[0], v0[1]])
                P = np.diag([cfg.measurement_noise_m ** 2, cfg.measurement_noise_m ** 2,
                             (cfg.velocity_noise_mps if fuse
                              else cfg.initial_velocity_sigma_mps) ** 2,
                             (cfg.velocity_noise_mps if fuse
                              else cfg.initial_velocity_sigma_mps) ** 2])
            else:
                x, P = self._predict(x, P, float(t[i] - t[i - 1]), q)
                z = (np.array([zx[i], zy[i], zvx[i], zvy[i]]) if fuse
                     else np.array([zx[i], zy[i]]))
                y = z - H @ x
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ y
                P = (np.eye(4) - K @ H) @ P
            out[i] = x

        return pd.DataFrame(
            {"t_s": t, "x_m": out[:, 0], "y_m": out[:, 1],
             "vx_mps": out[:, 2], "vy_mps": out[:, 3]}
        )

    def filter_observations(self, observations: pd.DataFrame) -> pd.DataFrame:
        """Filter every aircraft in an observation frame, preserving its columns.

        ``x_m``/``y_m``/``vx_mps``/``vy_mps`` are replaced by the filtered
        estimates; ``t_gen_s`` and the measured ages are left untouched, since
        filtering changes what the tracker believes, not when it last heard.
        """
        parts = []
        for _uid, g in observations.groupby("uas_id", sort=False):
            g = g.sort_values("t_gen_s").copy()
            if self.config.measurement_mode == "position_velocity":
                f = self.filter_track(g["t_gen_s"], g["x_m"], g["y_m"],
                                      g["vx_mps"], g["vy_mps"])
            else:
                f = self.filter_track(g["t_gen_s"], g["x_m"], g["y_m"])
            g["x_m"] = f["x_m"].to_numpy()
            g["y_m"] = f["y_m"].to_numpy()
            g["vx_mps"] = f["vx_mps"].to_numpy()
            g["vy_mps"] = f["vy_mps"].to_numpy()
            parts.append(g)
        return pd.concat(parts).sort_index()


def messages_from_filtered(
    observations: pd.DataFrame, config: TrackerConfig | None = None
) -> list[TelemetryMessage]:
    """Convenience: filtered observation rows as telemetry messages."""
    filt = ConstantVelocityTracker(config).filter_observations(observations)
    return [
        TelemetryMessage(
            uas_id=str(r.uas_id),
            t_gen_s=float(r.t_gen_s),
            t_arrive_s=float(r.t_s),
            state=State(str(r.uas_id), float(r.t_gen_s), float(r.x_m), float(r.y_m),
                        float(r.z_m) if np.isfinite(r.z_m) else 0.0,
                        float(r.vx_mps), float(r.vy_mps), 0.0),
        )
        for r in filt.itertuples()
    ]
