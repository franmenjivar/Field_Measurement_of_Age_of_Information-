"""Synthetic multi-aircraft traffic, calibrated from the real datasets.

BUBBLES caps out at 14 real concurrent aircraft, so the upper end of the
density axis has to be extrapolated.  This generator does that *from measured
parameters* rather than invented ones: speeds are drawn from a distribution
fitted to real flights, and the calibration source is recorded so the paper can
state exactly where the numbers came from.

Any figure built on this must label the synthetic points as extrapolation and
keep the real BUBBLES point visible alongside them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.rng import make_rng
from ..core.schema import coerce_trajectory_frame, derive_kinematics


@dataclass
class SpeedProfile:
    """Empirical speed distribution, sampled by inverse-CDF.

    Built from real telemetry (:meth:`from_trajectories`) so the generated
    traffic inherits real speed structure rather than a uniform guess.
    """

    quantiles: np.ndarray
    values: np.ndarray
    source: str = "unspecified"

    @classmethod
    def from_trajectories(
        cls, traj: pd.DataFrame, *, source: str, min_speed_mps: float = 1.0
    ) -> SpeedProfile:
        s = traj["speed_mps"].dropna()
        s = s[s >= min_speed_mps].to_numpy(float)
        if s.size < 10:
            raise ValueError("not enough airborne samples to fit a speed profile")
        q = np.linspace(0.0, 1.0, 101)
        return cls(quantiles=q, values=np.quantile(s, q), source=source)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.interp(rng.random(n), self.quantiles, self.values)

    def describe(self) -> dict:
        return {
            "source": self.source,
            "p05": float(np.interp(0.05, self.quantiles, self.values)),
            "p50": float(np.interp(0.50, self.quantiles, self.values)),
            "p95": float(np.interp(0.95, self.quantiles, self.values)),
        }


@dataclass
class TrafficConfig:
    """Parameters of a synthetic traffic scenario."""

    n_aircraft: int
    area_km2: float = 1.0
    duration_s: float = 300.0
    sample_rate_hz: float = 10.0
    altitude_band_m: tuple[float, float] = (30.0, 120.0)
    #: fraction of aircraft following straight transits (the rest loiter)
    transit_fraction: float = 0.7
    speed_profile: SpeedProfile | None = None
    seed: int = 42
    metadata: dict = field(default_factory=dict)

    @property
    def side_m(self) -> float:
        return float(np.sqrt(self.area_km2) * 1000.0)

    @property
    def density_per_km2(self) -> float:
        return self.n_aircraft / self.area_km2


def generate_traffic(config: TrafficConfig, scenario_id: str | None = None) -> pd.DataFrame:
    """Generate ``n_aircraft`` trajectories over a square operating area.

    Two behaviours are mixed because a single one distorts the encounter
    statistics: straight transits at random bearings (which produce genuine
    crossing geometries) and loitering orbits (which produce sustained local
    density without crossings).
    """
    sid = scenario_id or f"SIM-N{config.n_aircraft}-A{config.area_km2:g}"
    rng = make_rng(config.seed, sid)
    L = config.side_m
    t = np.arange(0.0, config.duration_s, 1.0 / config.sample_rate_hz)

    profile = config.speed_profile
    speeds = (
        profile.sample(config.n_aircraft, rng)
        if profile is not None
        else rng.uniform(5.0, 20.0, config.n_aircraft)
    )
    alts = rng.uniform(*config.altitude_band_m, config.n_aircraft)
    is_transit = rng.random(config.n_aircraft) < config.transit_fraction

    frames = []
    for i in range(config.n_aircraft):
        v = float(speeds[i])
        if is_transit[i]:
            # Enter from a random edge point on a random bearing across the box.
            bearing = rng.uniform(0, 2 * np.pi)
            start = rng.uniform(0, L, 2) - L / 2
            x = start[0] + v * np.cos(bearing) * t
            y = start[1] + v * np.sin(bearing) * t
            vx = np.full_like(t, v * np.cos(bearing))
            vy = np.full_like(t, v * np.sin(bearing))
        else:
            radius = rng.uniform(50.0, max(80.0, L / 6))
            omega = v / radius
            phase = rng.uniform(0, 2 * np.pi)
            cx, cy = rng.uniform(-L / 2, L / 2, 2)
            x = cx + radius * np.cos(omega * t + phase)
            y = cy + radius * np.sin(omega * t + phase)
            vx = -radius * omega * np.sin(omega * t + phase)
            vy = radius * omega * np.cos(omega * t + phase)

        frames.append(
            pd.DataFrame(
                {
                    "dataset": "sim",
                    "scenario_id": sid,
                    "uas_id": f"SIM-{i:03d}",
                    "t_s": t,
                    "x_m": x,
                    "y_m": y,
                    "z_m": alts[i],
                    "vx_mps": vx,
                    "vy_mps": vy,
                    "vz_mps": 0.0,
                    "source_type": "ground_truth",
                }
            )
        )

    out = derive_kinematics(coerce_trajectory_frame(pd.concat(frames, ignore_index=True)))
    out.attrs["traffic_config"] = {
        "n_aircraft": config.n_aircraft,
        "area_km2": config.area_km2,
        "density_per_km2": config.density_per_km2,
        "transit_fraction": config.transit_fraction,
        "seed": config.seed,
        "speed_profile": profile.describe() if profile else "uniform(5, 20) m/s",
        "synthetic": True,
    }
    return out
