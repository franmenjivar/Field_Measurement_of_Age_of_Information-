"""U-AGREE strategic airspace-occupancy adapter.

This dataset is **not telemetry** and must never be treated as trajectories.
It describes which 4-D airspace cells each operational intent reserves and
when, from PX4-based U-space scenarios.

Its role here is to supply realistic *demand*: how many operations share the
airspace, how much they overlap in space and time, and therefore how the
conflict-interaction rate scales.  That drives the density axis of E4/E5
without pretending to be aircraft state.
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.provenance import Provenance
from .base import DatasetNotAvailable


def _parse_limits(text: str) -> tuple[float, float]:
    """Parse a ``"[lo, hi]"`` bound string into a pair of floats."""
    lo, hi = str(text).strip().strip("[]").split(",")
    return float(lo), float(hi)


class UAgreeOccupancy:
    """4-D occupancy: operational intents, cells and time windows."""

    name = "uagree"

    def __init__(self, root: str | Path = "data/raw/uagree/17737383") -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise DatasetNotAvailable(
                f"uagree: {self.root} does not exist (see docs/DATASETS.md, U-AGREE)"
            )
        self.times_path = self.root / "times.csv"
        self.cells_path = self.root / "table_cells.csv"
        self.ois_path = self.root / "ois.json"
        for p in (self.times_path, self.cells_path):
            if not p.is_file():
                raise DatasetNotAvailable(f"uagree: missing {p}")

    # ----------------------------------------------------------------- data

    @cached_property
    def times(self) -> pd.DataFrame:
        """One row per (operational intent, cell, time window)."""
        d = pd.read_csv(self.times_path, sep=";", quotechar='"')
        d["timeBegin"] = pd.to_datetime(d["timeBegin"], errors="coerce")
        d["timeEnd"] = pd.to_datetime(d["timeEnd"], errors="coerce")
        d = d.dropna(subset=["timeBegin", "timeEnd"])
        t0 = d["timeBegin"].min()
        d["t_begin_s"] = (d["timeBegin"] - t0).dt.total_seconds()
        d["t_end_s"] = (d["timeEnd"] - t0).dt.total_seconds()
        d["duration_s"] = d["t_end_s"] - d["t_begin_s"]
        return d

    @cached_property
    def cells(self) -> pd.DataFrame:
        """Cell geometry in metres, with centres and volumes."""
        d = pd.read_csv(self.cells_path)
        for axis in ("x", "y", "z"):
            lo_hi = d[f"{axis}lim"].map(_parse_limits)
            d[f"{axis}_lo"] = [v[0] for v in lo_hi]
            d[f"{axis}_hi"] = [v[1] for v in lo_hi]
            d[f"{axis}_c"] = 0.5 * (d[f"{axis}_lo"] + d[f"{axis}_hi"])
        d["volume_m3"] = (
            (d["x_hi"] - d["x_lo"]) * (d["y_hi"] - d["y_lo"]) * (d["z_hi"] - d["z_lo"])
        )
        return d

    @cached_property
    def operational_intents(self) -> pd.DataFrame:
        """Takeoff/landing points and time windows from ``ois.json``.

        The file uses padded keys (``" takeoff "``) and spaced timestamps, so
        keys are stripped and timestamps de-spaced before parsing.
        """
        if not self.ois_path.is_file():
            return pd.DataFrame()
        raw = json.loads(self.ois_path.read_text())
        rows = []
        for i, item in enumerate(raw):
            clean = {str(k).strip(): v for k, v in item.items()}
            row: dict = {"oi_index": i}
            for key in ("takeoff", "landing"):
                geom = clean.get(key) or {}
                coords = geom.get(" coordinates ") or geom.get("coordinates") or []
                if len(coords) >= 3:
                    row[f"{key}_lon"], row[f"{key}_lat"], row[f"{key}_alt"] = coords[:3]
            for key in ("timeBegin", "timeEnd"):
                v = clean.get(key)
                if isinstance(v, str):
                    row[key] = pd.to_datetime(v.replace(" ", ""), errors="coerce")
            rows.append(row)
        return pd.DataFrame(rows)

    def grids(self) -> np.ndarray:
        """Memory-mapped occupancy grids ``(n, t, y, x)``."""
        p = next(self.root.glob("*grids*.npy"), None)
        if p is None:
            raise DatasetNotAvailable(f"uagree: no *grids*.npy under {self.root}")
        return np.load(p, mmap_mode="r")

    # ---------------------------------------------------------- derived

    def concurrency(self, step_s: float = 10.0) -> pd.DataFrame:
        """Number of simultaneously active operational intents over time.

        This is the demand curve the density experiments key off: it says how
        many operations genuinely share the airspace at once, rather than how
        many exist in the dataset.
        """
        t = self.times
        span = t.groupby("oi_id").agg(begin=("t_begin_s", "min"), end=("t_end_s", "max"))
        grid = np.arange(0.0, float(span["end"].max()) + step_s, step_s)
        active = np.zeros(grid.size, dtype=int)
        for _, r in span.iterrows():
            active += ((grid >= r["begin"]) & (grid <= r["end"])).astype(int)
        return pd.DataFrame({"t_s": grid, "active_operations": active})

    def cell_contention(self) -> pd.DataFrame:
        """Per cell: how many distinct intents claim it, and overlap in time.

        A cell claimed by several intents whose windows intersect is where a
        tactical conflict can actually arise, so this is a demand-side proxy
        for interaction density.
        """
        t = self.times
        g = t.groupby("idcell").agg(
            n_claims=("oi_id", "size"),
            n_intents=("oi_id", "nunique"),
            t_first=("t_begin_s", "min"),
            t_last=("t_end_s", "max"),
        )
        overlaps = []
        for cell, grp in t.groupby("idcell"):
            iv = grp[["t_begin_s", "t_end_s"]].sort_values("t_begin_s").to_numpy()
            n = 0
            for i in range(len(iv)):
                for j in range(i + 1, len(iv)):
                    if iv[j][0] < iv[i][1]:
                        n += 1
            overlaps.append({"idcell": cell, "n_time_overlaps": n})
        return g.merge(pd.DataFrame(overlaps).set_index("idcell"), on="idcell").reset_index()

    def summary(self) -> dict:
        t = self.times
        conc = self.concurrency()
        return {
            "n_rows": int(len(t)),
            "n_operational_intents": int(t["oi_id"].nunique()),
            "n_cells_claimed": int(t["idcell"].nunique()),
            "n_cells_defined": int(len(self.cells)),
            "max_concurrent_operations": int(conc["active_operations"].max()),
            "median_concurrent_operations": float(conc["active_operations"].median()),
            "campaign_duration_s": float(t["t_end_s"].max()),
            "median_window_s": float(t["duration_s"].median()),
        }

    def provenance(self) -> Provenance:
        p = Provenance(
            source_dataset="U-AGREE",
            coordinate_frame="local metric grid (dataset-defined)",
            notes=[
                "strategic 4-D occupancy, NOT telemetry",
                "used for demand/interaction density only; no aircraft state is "
                "derived from it",
            ],
        )
        for f in (self.times_path, self.cells_path, self.ois_path):
            if f.is_file():
                p.add_source(str(f))
        p.extra = self.summary()
        return p
