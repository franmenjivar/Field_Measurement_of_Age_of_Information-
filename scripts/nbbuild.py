"""Helper for generating the analysis notebooks from Python source.

Notebooks are generated rather than hand-edited so that they stay in sync with
the package and can be regenerated after a refactor.  Keep algorithm logic in
``src/telemetry_freshness`` -- a notebook should orchestrate and visualise only.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "notebooks"

HEADER = """\
import warnings
warnings.filterwarnings("ignore")

import sys, pathlib
ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import os; os.chdir(ROOT)

import numpy as np
import pandas as pd
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)

from telemetry_freshness.viz import style as S, figures as F
S.use_paper_style()
print("working dir:", ROOT)
"""


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


def build(name: str, cells: list[nbf.NotebookNode]) -> Path:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (telemetry-freshness)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
    }
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOKS / f"{name}.ipynb"
    nbf.write(nb, path)
    return path


def execute(path: Path, timeout: int = 1800) -> tuple[bool, str]:
    """Run a notebook in place; returns (ok, message)."""
    r = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            f"--ExecutePreprocessor.timeout={timeout}",
            str(path),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    tail = (r.stderr or r.stdout).strip().splitlines()
    return r.returncode == 0, "\n".join(tail[-12:])
