"""Common interface every dataset adapter implements.

The rest of the pipeline (telemetry emulator, detector, evaluator) only ever
sees this interface, so experiments are dataset-independent
(dataset-strategy doc, section 33).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from ..core.geodesy import GeodeticFrame
from ..core.provenance import Provenance


class DatasetNotAvailable(FileNotFoundError):
    """Raised when a dataset's files are not on disk.

    Deliberately loud: an experiment must never silently fall back to
    synthetic data when the real dataset is missing.
    """


class TrajectoryDataset(ABC):
    """A source of ground-truth trajectories in the unified schema."""

    #: short lowercase key, also the value of the ``dataset`` column
    name: str

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise DatasetNotAvailable(
                f"{self.name}: {self.root} does not exist. "
                f"Place the raw dataset there (see docs/DATASETS.md)."
            )

    @abstractmethod
    def scenarios(self) -> list[str]:
        """Identifiers of every scenario/flight this dataset offers."""

    @abstractmethod
    def load_scenario(self, scenario_id: str) -> pd.DataFrame:
        """Return all aircraft of one scenario in the unified schema."""

    @abstractmethod
    def provenance(self, scenario_id: str) -> Provenance:
        """Provenance record for one scenario."""

    def aircraft(self, scenario_id: str) -> list[str]:
        df = self.load_scenario(scenario_id)
        return sorted(df["uas_id"].dropna().unique().tolist())

    def trajectory(self, scenario_id: str, uas_id: str) -> pd.DataFrame:
        df = self.load_scenario(scenario_id)
        return df[df["uas_id"] == uas_id].sort_values("t_s").reset_index(drop=True)

    def frame(self, scenario_id: str) -> GeodeticFrame | None:
        """Local ENU frame used for this scenario, when it is geodetic."""
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(root={self.root!s})"


class ObservationDataset(ABC):
    """A source of *observed* surveillance state (UTM tracks, radar tracks).

    Kept separate from :class:`TrajectoryDataset` on purpose: mixing ground
    truth and observations behind one interface is exactly the error the study
    is designed to avoid.
    """

    name: str

    @abstractmethod
    def load_observations(self, scenario_id: str) -> pd.DataFrame:
        """Observed tracks with at minimum ``t_s``, ``uas_id`` and position.

        Should also carry ``t_gen_s`` (generation time of the state behind the
        observation) whenever the source provides it, since that is what makes
        a real Age-of-Information measurement possible.
        """
