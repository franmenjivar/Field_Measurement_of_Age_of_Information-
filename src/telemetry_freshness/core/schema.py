"""Unified trajectory schema shared by every dataset adapter and simulator.

One row = one state sample of one aircraft at one time.  Every adapter in
``telemetry_freshness.data`` must emit exactly these columns so that the
telemetry emulator, conflict detector and evaluator stay dataset-independent
(dataset-strategy doc, section 32).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PROCESSING_VERSION = "1.0"

#: Column -> numpy dtype.  Order is the canonical column order.
TRAJECTORY_DTYPES: dict[str, str] = {
    "dataset": "string",
    "scenario_id": "string",
    "uas_id": "string",
    "t_s": "float64",
    "lat_deg": "float64",
    "lon_deg": "float64",
    "alt_m_msl": "float64",
    "x_m": "float64",
    "y_m": "float64",
    "z_m": "float64",
    "vx_mps": "float64",
    "vy_mps": "float64",
    "vz_mps": "float64",
    "speed_mps": "float64",
    "heading_deg": "float64",
    "source_type": "string",
    "quality": "float32",
}

TRAJECTORY_COLUMNS: list[str] = list(TRAJECTORY_DTYPES)

#: Columns an adapter must populate; the rest may be NaN and are derived.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "dataset",
    "scenario_id",
    "uas_id",
    "t_s",
    "x_m",
    "y_m",
    "z_m",
)

KEY_COLUMNS: tuple[str, ...] = ("scenario_id", "uas_id")

SOURCE_TYPES: frozenset[str] = frozenset({"ground_truth", "observed", "sensor_track"})


class SchemaError(ValueError):
    """Raised when a frame does not satisfy the unified trajectory schema."""


def empty_trajectory_frame() -> pd.DataFrame:
    """Return a zero-row frame with the canonical columns and dtypes."""
    return pd.DataFrame(
        {c: pd.Series(dtype=d) for c, d in TRAJECTORY_DTYPES.items()}
    )


def coerce_trajectory_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing optional columns, order them canonically and cast dtypes.

    Missing required columns are an error; missing optional ones become NaN.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required column(s): {missing}")

    out = df.copy()
    for col, dtype in TRAJECTORY_DTYPES.items():
        if col not in out.columns:
            out[col] = pd.NA if dtype == "string" else np.nan
        out[col] = out[col].astype(dtype)

    if out["source_type"].isna().all():
        out["source_type"] = "ground_truth"

    return out[TRAJECTORY_COLUMNS]


def derive_kinematics(df: pd.DataFrame, *, overwrite: bool = False) -> pd.DataFrame:
    """Fill ``vx/vy/vz``, ``speed_mps`` and ``heading_deg`` where absent.

    Velocities are obtained by central differences of the ENU position within
    each ``(scenario_id, uas_id)`` group.  Native velocity fields are preferred
    and left untouched unless ``overwrite`` is set -- differentiation injects
    noise, so provenance must record when it was used.
    """
    out = df.copy()
    need_v = overwrite or out[["vx_mps", "vy_mps", "vz_mps"]].isna().all(axis=None)

    if need_v:
        parts = []
        for _, g in out.groupby(list(KEY_COLUMNS), sort=False, dropna=False):
            g = g.sort_values("t_s")
            t = g["t_s"].to_numpy()
            if len(t) < 2:
                g[["vx_mps", "vy_mps", "vz_mps"]] = np.nan
            else:
                for axis, vcol in (("x_m", "vx_mps"), ("y_m", "vy_mps"), ("z_m", "vz_mps")):
                    g[vcol] = np.gradient(g[axis].to_numpy(), t, edge_order=1)
            parts.append(g)
        out = pd.concat(parts).sort_index()

    speed = np.hypot(out["vx_mps"], out["vy_mps"])
    if overwrite or out["speed_mps"].isna().all():
        out["speed_mps"] = speed
    if overwrite or out["heading_deg"].isna().all():
        # ENU: heading measured clockwise from North (+y).
        out["heading_deg"] = np.degrees(np.arctan2(out["vx_mps"], out["vy_mps"])) % 360.0

    return out


def validate_trajectory_frame(df: pd.DataFrame, *, strict_monotonic: bool = True) -> None:
    """Assert the frame is a well-formed trajectory table.

    Checks the invariants the whole pipeline relies on: canonical columns,
    finite positions, known ``source_type`` values, and a strictly increasing
    time base per aircraft (required by the state store's latest-value logic).
    """
    missing = [c for c in TRAJECTORY_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing column(s): {missing}")

    if df.empty:
        return

    bad_source = set(df["source_type"].dropna().unique()) - SOURCE_TYPES
    if bad_source:
        raise SchemaError(f"unknown source_type value(s): {sorted(bad_source)}")

    for col in ("t_s", "x_m", "y_m", "z_m"):
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise SchemaError(f"non-finite values in required column {col!r}")

    if strict_monotonic:
        for key, g in df.groupby(list(KEY_COLUMNS), sort=False, dropna=False):
            t = g["t_s"].to_numpy()
            if not np.all(np.diff(t) > 0):
                raise SchemaError(f"t_s is not strictly increasing for {key}")


@dataclass(frozen=True, slots=True)
class State:
    """A single aircraft state in the local ENU frame.

    Deliberately minimal and immutable: this is the object handed to the
    conflict detector, and it carries no reference back to the simulator.
    """

    uas_id: str
    t_s: float
    x_m: float
    y_m: float
    z_m: float
    vx_mps: float
    vy_mps: float
    vz_mps: float

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x_m, self.y_m, self.z_m], dtype=float)

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx_mps, self.vy_mps, self.vz_mps], dtype=float)

    @property
    def speed_mps(self) -> float:
        return float(np.hypot(self.vx_mps, self.vy_mps))

    def propagate(self, dt_s: float) -> State:
        """Constant-velocity dead reckoning by ``dt_s`` seconds."""
        return State(
            uas_id=self.uas_id,
            t_s=self.t_s + dt_s,
            x_m=self.x_m + self.vx_mps * dt_s,
            y_m=self.y_m + self.vy_mps * dt_s,
            z_m=self.z_m + self.vz_mps * dt_s,
            vx_mps=self.vx_mps,
            vy_mps=self.vy_mps,
            vz_mps=self.vz_mps,
        )
