r"""Render the three manuscript figures at IEEE two-column print size.

Authored at final physical size -- 3.5 in single column, 7.16 in full width --
so the point sizes in the style are (near enough) the point sizes on the page.

One detail worth knowing when placing these in LaTeX: ``savefig.bbox="tight"``
trims surrounding whitespace, so the emitted PDFs come out slightly narrower
than the nominal width (~3.28 in and ~7.02 in). Placing them with
``\includegraphics[width=\columnwidth]`` therefore scales them up by about 7%,
which lifts the 8 pt type to roughly 8.5 pt on the page -- above IEEE's 8 pt
floor, so this is left as is rather than compensated for.

Output: PDF (vector, for submission) and PNG at 600 dpi (for review).

    python scripts/make_ieee_figures.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import matplotlib

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# The package lives under src/ and every path in the project is repo-relative.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import pandas as pd  # noqa: E402

from telemetry_freshness.data.bubbles import BubblesDataset  # noqa: E402
from telemetry_freshness.evaluation import load as L  # noqa: E402
from telemetry_freshness.viz import figures as F  # noqa: E402
from telemetry_freshness.viz import style as S  # noqa: E402

OUT = "results/figures/ieee"


def figure1() -> None:
    ds = BubblesDataset()
    corr = pd.read_csv("results/processed/E10_load_vs_freshness.csv")
    profiles, ann = {}, {}
    for day in ds.scenarios():
        per = L.cycle_load_table(ds.load_observations(day))
        profiles[day] = L.load_profile(per, min_cycles=150)
        r = corr[(corr.day == day) & (corr.statistic == "P95 AoI")]
        if len(r):
            ann[day] = f"ρ={r.rho.iloc[0]:+.2f}"
    fig, _ = F.fig_load_vs_freshness(
        profiles, annotations=ann, compact=True,
        title="Load inflates the tail, not the centre",
        figsize=S.ieee_size(1, 0.82),
    )
    S.save(fig, "fig1_load_vs_freshness", directory=OUT)
    S.plt.close(fig)


def figure2() -> None:
    tab = pd.read_csv("results/processed/E11_replay_by_aoi.csv")
    fig, _ = F.fig_sense_inversion(
        tab, title="Stale state inverts the encounter sense",
        figsize=S.ieee_size(1, 0.82),
    )
    S.save(fig, "fig2_sense_inversion", directory=OUT)
    S.plt.close(fig)


def figure3() -> None:
    syn = pd.read_parquet("results/raw/E12_synthetic_extrapolation.parquet")
    real = pd.read_parquet("results/raw/E12_four_arm_replay.parquet")
    real["abs_err"] = real.d_cpa_error_m.abs()
    # Three bars keep the figure readable at print size. The filtered arm shown
    # is the *strongest* one -- the tracker that fuses reported velocity with
    # position. The position-only tracker performed no better and is reported
    # in the text rather than plotted.
    two = real[real.arm.isin(["hold_last", "dead_reckon"])].assign(
        extrapolate=lambda d: d.arm == "dead_reckon"
    )
    filt = real[real.arm == "dr_kalman_posvel"]
    filt_inv = (
        100
        * filt.assign(bin=pd.cut(filt.aoi_s, [0, 0.5, 1, 2, 5, 30]))
        .groupby("bin", observed=True)
        .observed_diverging.mean()
    )
    fig, _ = F.fig_extrapolation_ablation(
        syn, two, None,
        filtered_median=float(filt.abs_err.median()),
        filtered_inversion=filt_inv,
        show_mechanism=False,
        title="Dead reckoning fixes the emulator, not the real system",
        figsize=S.ieee_size(2, 0.42),
    )
    S.save(fig, "fig3_extrapolation_ablation", directory=OUT)
    S.plt.close(fig)


if __name__ == "__main__":
    S.use_ieee_style()
    for fn in (figure1, figure2, figure3):
        fn()
        print("rendered", fn.__name__)
    print(f"\nwritten to {OUT}/ (PDF for submission, PNG for review)")
