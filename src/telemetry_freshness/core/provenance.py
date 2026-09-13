"""Provenance records written beside every processed artefact.

Dataset-strategy doc section 37: a processed trajectory must always be traceable
back to the exact source file, transformation and code version that produced it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schema import PROCESSING_VERSION


def file_sha256(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (datasets are up to a few GB)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_commit(repo: str | Path = ".") -> str | None:
    """Short commit hash, or ``None`` when not a git repository."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass
class Provenance:
    """Everything needed to explain where a processed artefact came from."""

    source_dataset: str
    source_files: list[str] = field(default_factory=list)
    source_sha256: dict[str, str] = field(default_factory=dict)
    scenario_id: str | None = None
    reference_rate_hz: float | None = None
    coordinate_frame: str = "ENU"
    origin: dict[str, float] = field(default_factory=dict)
    resampling_method: str | None = None
    filters: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    processing_version: str = PROCESSING_VERSION
    git_commit: str | None = field(default_factory=git_commit)
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    created_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    extra: dict[str, Any] = field(default_factory=dict)

    def add_source(self, path: str | Path, *, hash_file: bool = False) -> None:
        """Register a source file; hashing is opt-in because inputs are large."""
        p = str(path)
        if p not in self.source_files:
            self.source_files.append(p)
        if hash_file:
            self.source_sha256[p] = file_sha256(path)

    def to_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(self), indent=2, default=str))
        return out


def write_parquet_with_provenance(
    df, path: str | Path, provenance: Provenance
) -> tuple[Path, Path]:
    """Write ``df`` to Parquet plus a ``<name>.provenance.json`` sidecar."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    side = provenance.to_json(out.with_suffix(".provenance.json"))
    return out, side
