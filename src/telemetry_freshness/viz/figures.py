"""The manuscript's figures, each callable from a notebook or a script.

Every function takes a tidy DataFrame and returns ``(fig, ax)`` so figures can
be regenerated from stored results without re-running any simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from . import style as S


def _band(ax, x, lo, hi, color, alpha=0.13):
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0, zorder=1)


# --------------------------------------------------------------- F1 / F1b

def _titled(ax, title: str, subtitle: str | None) -> None:
    """Title with an optional subtitle above the axes, without collision."""
    ax.set_title(title, pad=20 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=8,
                color=S.INK["secondary"], va="bottom")


def fig_aoi_vs_position_error(
    df: pd.DataFrame,
    *,
    level_col: str = "update_rate_hz",
    second_col: str | None = "latency_ms",
    aoi_col: str = "aoi_s",
    err_col: str = "horiz_error_m",
    title: str = "Position error is governed by information age, not by update rate",
    subtitle: str = "IDF-DS real fixed-wing flight, 62 Hz reference trajectory",
    figsize=(6.6, 4.2),
):
    """F1 - position error against AoI for every configuration.

    Each (rate, latency) configuration is drawn as its own thin line.  They
    collapse onto a common relationship, which is the point of the figure: a
    second of age costs the same whether it came from slow sampling or from
    network delay.  That is what justifies AoI as the controlling variable
    rather than the configured update rate.
    """
    fig, ax = S.plt.subplots(figsize=figsize)
    levels = sorted(df[level_col].dropna().unique(), reverse=True)
    colors = S.sequential_colors(len(levels))
    group_cols = [level_col] + ([second_col] if second_col and second_col in df else [])

    for lv, c in zip(levels, colors, strict=True):
        sub = df[df[level_col] == lv]
        for _, g in sub.groupby(group_cols, sort=True, observed=True):
            g = g.dropna(subset=[aoi_col, err_col])
            if len(g) < 60:
                continue
            b = (
                g.assign(_b=pd.qcut(g[aoi_col], q=8, duplicates="drop"))
                .groupby("_b", observed=True)
                .agg(aoi=(aoi_col, "median"), p50=(err_col, "median"))
                .dropna()
            )
            if len(b) >= 2:
                ax.plot(b["aoi"], b["p50"], color=c, lw=1.3, alpha=0.75, zorder=3)

    # Pooled relationship across every configuration.
    pooled = df.dropna(subset=[aoi_col, err_col])
    pb = (
        pooled.assign(_b=pd.qcut(pooled[aoi_col], q=18, duplicates="drop"))
        .groupby("_b", observed=True)
        .agg(
            aoi=(aoi_col, "median"),
            p50=(err_col, "median"),
            p95=(err_col, lambda s: np.percentile(s, 95)),
        )
        .dropna()
    )
    _band(ax, pb["aoi"], pb["p50"], pb["p95"], S.INK["secondary"], alpha=0.10)
    ax.plot(pb["aoi"], pb["p50"], color=S.INK["primary"], lw=2.6, zorder=5,
            solid_capstyle="round")
    S.direct_label(ax, pb["aoi"].iloc[-1], pb["p50"].iloc[-1], "all configs", S.INK["primary"])

    ax.set_xlabel("Age of Information (s)")
    ax.set_ylabel("Horizontal position error (m)")
    _titled(ax, title, subtitle)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    handles = [Line2D([], [], color=c, lw=1.6, label=f"{lv:g} Hz")
               for lv, c in zip(levels, colors, strict=True)]
    handles.append(Line2D([], [], color=S.INK["primary"], lw=2.6, label="pooled median"))
    ax.legend(handles=handles, title="Telemetry rate", loc="upper left", ncols=2)
    S.annotate_note(ax, "one line per rate × latency configuration · band: pooled P50-P95")
    S.finish(fig, ax)
    return fig, ax


def fig_error_by_flight_phase(
    df: pd.DataFrame,
    *,
    phase_col: str = "phase",
    err_col: str = "horiz_error_m",
    title: str = "The same information age costs more during manoeuvres",
    figsize=(6.4, 3.6),
):
    """F1b - error distribution by flight phase at comparable AoI."""
    fig, ax = S.plt.subplots(figsize=figsize)
    phases = [p for p in ("cruise", "turn", "climb", "descent", "accel") if p in set(df[phase_col])]
    phases += [p for p in sorted(set(df[phase_col])) if p not in phases]
    colors = S.categorical_colors(min(len(phases), 6))

    data = [df.loc[df[phase_col] == p, err_col].dropna().to_numpy() for p in phases]
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for body, c in zip(parts["bodies"], colors, strict=False):
        body.set_facecolor(c)
        body.set_alpha(0.35)
        body.set_edgecolor(c)
        body.set_linewidth(1.2)

    for i, (vals, c) in enumerate(zip(data, colors, strict=False), start=1):
        if not len(vals):
            continue
        med = float(np.median(vals))
        ax.plot([i], [med], marker="o", ms=6, color=c, zorder=4,
                markeredgecolor=S.INK["surface"], markeredgewidth=1.5)
        ax.annotate(f"{med:.1f} m", (i, med), textcoords="offset points", xytext=(10, 0),
                    fontsize=7.5, color=S.INK["secondary"], va="center")

    ax.set_xticks(range(1, len(phases) + 1))
    ax.set_xticklabels([p.title() for p in phases])
    ax.set_ylabel("Horizontal position error (m)")
    ax.set_title(title)
    S.finish(fig, ax)
    return fig, ax


# --------------------------------------------------------------------- F2

def fig_recall_vs_aoi(
    df: pd.DataFrame,
    *,
    level_col: str = "density_label",
    x_col: str = "aoi_bin_mid",
    y_col: str = "recall",
    lo_col: str | None = "recall_lo",
    hi_col: str | None = "recall_hi",
    requirement: float | None = 0.99,
    title: str = "Conflict recall degrades with information age",
    figsize=(6.4, 4.2),
):
    """F2 - conflict recall vs AoI, one curve per traffic condition."""
    fig, ax = S.plt.subplots(figsize=figsize)
    levels = list(dict.fromkeys(df[level_col]))
    colors = S.sequential_colors(len(levels))

    for lv, c in zip(levels, colors, strict=True):
        g = df[df[level_col] == lv].sort_values(x_col)
        if g.empty:
            continue
        if lo_col and hi_col and lo_col in g:
            _band(ax, g[x_col], g[lo_col], g[hi_col], c)
        ax.plot(g[x_col], g[y_col], color=c, marker="o", ms=4.5, zorder=3,
                markeredgecolor=S.INK["surface"], markeredgewidth=1.0)
        S.direct_label(ax, g[x_col].iloc[-1], g[y_col].iloc[-1], str(lv), c)

    if requirement is not None:
        ax.axhline(requirement, color=S.STATUS["critical"], lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.text(0.01, requirement, f" requirement = {requirement:g}",
                transform=ax.get_yaxis_transform(), va="bottom", fontsize=7,
                color=S.STATUS["critical"])

    ax.set_xscale("symlog", linthresh=0.25)
    ax.set_xlabel("Age of Information (s)")
    ax.set_ylabel("Conflict recall")
    ax.set_ylim(-0.02, 1.04)
    ax.set_title(title)
    ax.legend(
        handles=[Line2D([], [], color=c, lw=2, label=str(lv))
                 for lv, c in zip(levels, colors, strict=True)],
        title=level_col.replace("_", " ").title(), loc="lower left",
    )
    S.annotate_note(ax, "band: Wilson 95% CI")
    S.finish(fig, ax)
    return fig, ax


# --------------------------------------------------------------------- F3

def fig_ttc_error_vs_aoi(
    df: pd.DataFrame,
    *,
    aoi_col: str = "aoi_s",
    signed_col: str = "ttc_error_signed_s",
    level_col: str | None = None,
    title: str = "Stale state distorts the estimated time to conflict",
    figsize=(6.4, 4.0),
):
    """F3 - signed TTC error vs AoI.

    Signed rather than absolute: the dangerous direction is an *overestimated*
    TTC, where the system believes it has more reaction time than it has.
    """
    fig, ax = S.plt.subplots(figsize=figsize)
    levels = list(dict.fromkeys(df[level_col])) if level_col else [None]
    colors = S.sequential_colors(len(levels))

    for lv, c in zip(levels, colors, strict=True):
        g = df if lv is None else df[df[level_col] == lv]
        g = g.dropna(subset=[aoi_col, signed_col])
        if len(g) < 10:
            continue
        b = (
            g.assign(_b=pd.qcut(g[aoi_col], q=min(12, max(2, len(g) // 30)), duplicates="drop"))
            .groupby("_b", observed=True)
            .agg(
                aoi=(aoi_col, "median"),
                mean=(signed_col, "mean"),
                p05=(signed_col, lambda s: np.percentile(s, 5)),
                p95=(signed_col, lambda s: np.percentile(s, 95)),
            )
            .dropna()
        )
        _band(ax, b["aoi"], b["p05"], b["p95"], c)
        ax.plot(b["aoi"], b["mean"], color=c, zorder=3)
        # No direct labels here: the series converge at the right edge, so
        # end-of-line text collides. The legend carries identity instead.

    ax.axhline(0, color=S.INK["muted"], lw=0.9)
    # Spell the direction out: a glyph arrow is not guaranteed in the font.
    ax.text(0.995, -0.02, "  below zero: TTC overestimated (dangerous)",
            transform=ax.get_yaxis_transform(), ha="right", va="top",
            fontsize=7, color=S.STATUS["critical"])
    ax.set_xlabel("Age of Information (s)")
    ax.set_ylabel("Signed TTC error, truth − observed (s)")
    ax.set_title(title)
    if level_col:
        ax.legend(
            handles=[Line2D([], [], color=c, lw=2, label=str(lv))
                     for lv, c in zip(levels, colors, strict=True)],
            title=level_col.replace("_", " ").title(), loc="best",
        )
    S.annotate_note(ax, "line: mean · band: P5-P95")
    S.finish(fig, ax)
    return fig, ax


# --------------------------------------------------------------------- F4

def fig_lead_time_vs_aoi(
    df: pd.DataFrame,
    *,
    aoi_col: str = "aoi_bin_mid",
    lead_col: str = "lead_time_p5_s",
    level_col: str | None = None,
    min_lead_s: float = 3.0,
    title: str = "Alert lead time collapses as surveillance ages",
    figsize=(6.4, 4.0),
):
    """F4 - alert lead time vs AoI, with the unusable region shaded."""
    fig, ax = S.plt.subplots(figsize=figsize)
    levels = list(dict.fromkeys(df[level_col])) if level_col else [None]
    colors = S.sequential_colors(len(levels))

    ymin = float(np.nanmin(df[lead_col])) if df[lead_col].notna().any() else 0.0
    S.shade_region(ax, min(ymin, 0) - 1, min_lead_s,
                   label=f"below minimum manoeuvring time ({min_lead_s:g} s)")

    for lv, c in zip(levels, colors, strict=True):
        g = (df if lv is None else df[df[level_col] == lv]).sort_values(aoi_col)
        g = g.dropna(subset=[aoi_col, lead_col])
        if g.empty:
            continue
        ax.plot(g[aoi_col], g[lead_col], color=c, marker="o", ms=4.5, zorder=3,
                markeredgecolor=S.INK["surface"], markeredgewidth=1.0)
        if lv is not None:
            S.direct_label(ax, g[aoi_col].iloc[-1], g[lead_col].iloc[-1], str(lv), c)

    ax.set_xlabel("Age of Information (s)")
    ax.set_ylabel("Alert lead time, 5th percentile (s)")
    ax.set_title(title)
    if level_col:
        ax.legend(
            handles=[Line2D([], [], color=c, lw=2, label=str(lv))
                     for lv, c in zip(levels, colors, strict=True)],
            title=level_col.replace("_", " ").title(), loc="best",
        )
    S.finish(fig, ax)
    return fig, ax


# --------------------------------------------------------------------- F5

def fig_rate_latency_heatmap(
    df: pd.DataFrame,
    *,
    rate_col: str = "update_rate_hz",
    latency_col: str = "latency_ms",
    value_col: str = "ttc_error_abs_s",
    agg: str = "p95",
    title: str = "P95 TTC error across the rate × latency grid",
    figsize=(6.0, 4.0),
):
    """F5 - heatmap over the controlled sweep, values printed in every cell."""
    f = (lambda s: np.percentile(s.dropna(), 95)) if agg == "p95" else agg
    piv = (
        df.pivot_table(index=rate_col, columns=latency_col, values=value_col, aggfunc=f)
        .sort_index(ascending=False)
    )

    fig, ax = S.plt.subplots(figsize=figsize)
    im = ax.imshow(piv.to_numpy(), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{r:g}" for r in piv.index])
    ax.set_xlabel("Added network latency (ms)")
    ax.set_ylabel("Telemetry update rate (Hz)")
    ax.set_title(title)
    ax.grid(False)

    vals = piv.to_numpy()
    mid = np.nanmean(vals) if np.isfinite(vals).any() else 0.0
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color=S.INK["surface"] if v > mid else S.INK["primary"])

    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(f"{agg.upper()} |TTC error| (s)", fontsize=8)
    cb.outline.set_visible(False)
    S.finish(fig, ax)
    return fig, ax


# --------------------------------------------------------------------- F6

def fig_safe_aoi_vs_density(
    df: pd.DataFrame,
    *,
    x_col: str = "interaction_density",
    y_col: str = "aoi_max_safe_s",
    label_col: str | None = None,
    title: str = "Maximum safe information age falls as interaction density rises",
    xlabel: str = "Local interaction density (neighbours within 500 m)",
    figsize=(6.4, 4.0),
):
    """F6 - the headline result: safe AoI against operating condition."""
    fig, ax = S.plt.subplots(figsize=figsize)
    d = df.dropna(subset=[x_col, y_col]).sort_values(x_col)
    c = S.CATEGORICAL[0]

    ax.plot(d[x_col], d[y_col], color=c, marker="o", ms=7, lw=2, zorder=3,
            markeredgecolor=S.INK["surface"], markeredgewidth=1.5)

    if len(d) >= 3:
        # Monotonicity is not assumed (research plan, Figure 2): the fit is a
        # visual guide only and is drawn recessively.
        z = np.polyfit(d[x_col], d[y_col], 1)
        xs = np.linspace(d[x_col].min(), d[x_col].max(), 50)
        ax.plot(xs, np.polyval(z, xs), color=c, lw=1.0, ls=(0, (5, 4)), alpha=0.5, zorder=2)

    if label_col:
        for _, r in d.iterrows():
            ax.annotate(str(r[label_col]), (r[x_col], r[y_col]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=7.5, color=S.INK["secondary"])

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Maximum safe AoI (s)")
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    S.finish(fig, ax)
    return fig, ax


# --------------------------------------------------------------------- F8

def fig_rate_vs_realised_aoi(
    df: pd.DataFrame,
    *,
    rate_col: str = "update_rate_hz",
    aoi_col: str = "aoi_s",
    title: str = "Configured update rate does not determine information freshness",
    figsize=(6.4, 4.0),
):
    """F8 - realised AoI distribution per configured rate.

    Makes the paper's framing point: two systems at the same nominal rate can
    present very different information ages to the conflict detector.
    """
    fig, ax = S.plt.subplots(figsize=figsize)
    rates = sorted(df[rate_col].dropna().unique())
    colors = S.sequential_colors(len(rates))

    for i, (r, c) in enumerate(zip(rates, colors, strict=True)):
        vals = df.loc[df[rate_col] == r, aoi_col].dropna().to_numpy()
        if not vals.size:
            continue
        bp = ax.boxplot(
            [vals], positions=[i], widths=0.55, vert=True, showfliers=False,
            patch_artist=True, medianprops=dict(color=S.INK["surface"], lw=1.6),
            whiskerprops=dict(color=c, lw=1.2), capprops=dict(color=c, lw=1.2),
        )
        for box in bp["boxes"]:
            box.set_facecolor(c)
            box.set_edgecolor(c)
            box.set_alpha(0.85)
        # The nominal expectation is half the sampling interval.
        ax.plot([i], [0.5 / r], marker="_", ms=18, color=S.STATUS["critical"], zorder=5)

    ax.plot([], [], marker="_", ls="none", ms=12, color=S.STATUS["critical"],
            label="nominal expectation (1/2 sampling interval)")
    ax.set_xticks(range(len(rates)), [f"{r:g} Hz" for r in rates])
    ax.set_xlabel("Configured telemetry update rate")
    ax.set_ylabel("Realised Age of Information (s)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    S.finish(fig, ax)
    return fig, ax


# ------------------------------------------------------------- diagnostics

def fig_aoi_distribution(
    aoi_s, *, title: str = "Measured Age of Information", label: str = "",
    figsize=(6.4, 3.6), reference: dict | None = None,
):
    """Empirical AoI CDF on a log axis - the shape of a real link's tail."""
    a = np.sort(pd.Series(aoi_s).dropna().to_numpy(dtype=float))
    a = a[a > 0]
    fig, ax = S.plt.subplots(figsize=figsize)
    c = S.CATEGORICAL[0]
    ax.plot(a, np.linspace(0, 1, a.size), color=c, lw=2)

    for q, name in ((0.5, "P50"), (0.95, "P95"), (0.99, "P99")):
        v = float(np.quantile(a, q))
        ax.plot([v], [q], marker="o", ms=6, color=c,
                markeredgecolor=S.INK["surface"], markeredgewidth=1.5, zorder=4)
        # Near the top of the CDF the curve is almost flat, so a label placed
        # level with the marker would sit on it; drop those below instead.
        dy = -13 if q >= 0.95 else 0
        S.direct_label(ax, v, q, f"{name} = {v:.2f}s", S.INK["secondary"], dx=9, dy=dy)

    if reference:
        for name, v in reference.items():
            ax.axvline(v, color=S.INK["muted"], lw=0.9, ls=(0, (3, 3)))
            ax.text(v, 0.02, f" {name}", rotation=90, fontsize=7,
                    color=S.INK["muted"], va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("Age of Information (s, log scale)")
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    if label:
        S.annotate_note(ax, f"{label} · n = {a.size:,}", loc="lower right")
    S.finish(fig, ax)
    return fig, ax


def fig_trajectory_map(
    traj: pd.DataFrame, *, title: str = "Scenario trajectories",
    max_aircraft: int = 6, figsize=(6.0, 5.4),
):
    """Plan view of a scenario, for QC and for the methods section."""
    fig, ax = S.plt.subplots(figsize=figsize)
    ids = list(traj["uas_id"].dropna().unique())[:max_aircraft]
    colors = S.categorical_colors(min(len(ids), 6))

    for uid, c in zip(ids, colors, strict=False):
        g = traj[traj["uas_id"] == uid].sort_values("t_s")
        ax.plot(g["x_m"], g["y_m"], color=c, lw=1.5, alpha=0.9, label=str(uid))
        ax.plot(g["x_m"].iloc[0], g["y_m"].iloc[0], marker="o", ms=6, color=c,
                markeredgecolor=S.INK["surface"], markeredgewidth=1.4, zorder=4)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(title)
    ax.legend(loc="best", ncols=2, fontsize=7.5)
    S.annotate_note(ax, "● start of track")
    S.finish(fig, ax)
    return fig, ax


# --------------------------------------------------------------------- F9

def fig_load_vs_freshness(
    profiles: dict[str, pd.DataFrame],
    *,
    load_col: str = "n_tracked",
    annotations: dict[str, str] | None = None,
    compact: bool = False,
    title: str = "Traffic load inflates the tail of information age, not its centre",
    subtitle: str | None = None,
    figsize=(6.8, 4.3),
):
    """F9 / paper figure 1 - measured AoI against concurrently tracked aircraft.

    Median and P95 are drawn together because the finding is the *divergence*
    between them: the centre of the distribution is flat under load while the
    tail climbs.

    ``compact`` targets a single IEEE column.  At 3.5 in there is no room for
    end-of-line labels, so series identity moves into a short legend and the
    two statistics are annotated once inside the axes.

    No dispersion band is drawn: the per-cycle P95 has occasional multi-minute
    outliers, so an inter-quartile band spans most of the axis and hides the
    trend the figure exists to show.  The correlation statistics carry the
    uncertainty instead.
    """
    fig, ax = S.plt.subplots(figsize=figsize)
    colors = S.categorical_colors(max(len(profiles), 2))
    handles, top_curve = [], None

    for i, ((label, d), c) in enumerate(zip(profiles.items(), colors, strict=False)):
        d = d.sort_values(load_col)
        lw = 1.6 if compact else 2.4
        ax.plot(d[load_col], d["aoi_p95_s"], color=c, marker="o",
                ms=3.2 if compact else 5.5, lw=lw, zorder=4,
                markeredgecolor=S.INK["surface"], markeredgewidth=0.8)
        ax.plot(d[load_col], d["aoi_median_s"], color=c, marker="o",
                ms=2.4 if compact else 3.5, lw=lw * 0.65, ls=(0, (3, 2)),
                zorder=3, alpha=0.9)
        note = f" ({annotations[label]})" if annotations and label in annotations else ""
        if compact:
            short = f"Day {i + 1}{note}"
            handles.append(Line2D([], [], color=c, lw=1.6, label=short))
        else:
            S.direct_label(ax, d[load_col].iloc[-1], d["aoi_p95_s"].iloc[-1],
                           f"{label}{note}", c)
        if top_curve is None or d["aoi_p95_s"].max() > top_curve[1]:
            top_curve = (d[load_col].iloc[-1], d["aoi_p95_s"].iloc[-1])

    ax.set_xlabel("Aircraft tracked concurrently")
    ax.set_ylabel("Age of Information (s)")
    ax.set_ylim(bottom=0)

    if compact:
        # Integer aircraft counts only -- 2.5 aircraft is not a thing.
        lo = int(min(d[load_col].min() for d in profiles.values()))
        hi = int(max(d[load_col].max() for d in profiles.values()))
        ax.set_xticks(range(lo, hi + 1, 2))
        ax.set_xlim(lo - 0.5, hi + 0.5)
        ax.set_title(title)
        ax.legend(handles=handles, loc="upper left", handlelength=1.4,
                  borderpad=0.3, labelspacing=0.25)
        # Identify the two statistics once, on the curves themselves.
        ax.annotate("P95", xy=top_curve, xytext=(-2, 4),
                    textcoords="offset points", ha="right", fontsize=7,
                    fontweight="bold", color=S.INK["primary"])
        med_last = list(profiles.values())[0].sort_values(load_col).iloc[-1]
        ax.annotate("median", xy=(med_last[load_col], med_last["aoi_median_s"]),
                    xytext=(-2, -10), textcoords="offset points", ha="right",
                    fontsize=7, color=S.INK["secondary"])
    else:
        _titled(ax, title, subtitle)
        ax.legend(
            handles=[
                Line2D([], [], color=S.INK["secondary"], lw=2.4, label="P95 of cycle (tail)"),
                Line2D([], [], color=S.INK["secondary"], lw=1.5, ls=(0, (4, 3)),
                       label="median of cycle (centre)"),
            ],
            loc="upper left",
        )
    S.finish(fig, ax)
    return fig, ax


# -------------------------------------------------------------------- F10

def fig_replay_error_decomposition(
    df: pd.DataFrame,
    *,
    aoi_col: str = "aoi_s",
    err_col: str = "d_cpa_error_m",
    inversion_col: str = "observed_diverging",
    bins=(0.0, 0.5, 1.0, 2.0, 5.0, 30.0),
    title: str = "Real surveillance stream: information age on top of the sensor floor",
    subtitle: str | None = None,
    figsize=(6.6, 8.4),
):
    """F10 - two stacked panels from the fully-real replay.

    Top: closest-approach error against measured age, decomposed by quantile
    regression into an age-independent floor (the UTM's own estimation error)
    and an age-driven slope.  Median and P90 are fitted separately because they
    behave differently -- age inflates the tail of the error several times
    faster than its centre.  Binned points and fitted lines use the same
    statistic, so they cannot disagree.

    Bottom: how often the surveillance picture inverts the *sense* of the
    encounter -- reporting a converging pair as diverging.

    Both panels read against measured age, but not on a common scale: the top
    is continuous and capped at the P98 age, the bottom is binned by ``bins``
    with unequal widths.  They are stacked for column width, not for reading
    one off the other vertically.
    """
    import statsmodels.api as sm

    fig, axes = S.plt.subplots(2, 1, figsize=figsize)

    # ---- top: error vs age, median and P90 decomposition
    ax = axes[0]
    d = df.dropna(subset=[aoi_col, err_col]).copy()
    d["abs_err"] = d[err_col].abs()

    sample = d.sample(min(4000, len(d)), random_state=0)
    ax.scatter(sample[aoi_col], sample["abs_err"], s=5, alpha=0.08,
               color=S.INK["muted"], edgecolors="none", zorder=1)

    binned = (d.assign(_b=pd.cut(d[aoi_col], list(bins)))
                .groupby("_b", observed=True)
                .agg(aoi=(aoi_col, "median"),
                     p50=("abs_err", "median"),
                     p90=("abs_err", lambda s: np.percentile(s, 90)))
                .dropna())

    X = sm.add_constant(d[[aoi_col]])
    # Axes are capped near the 98th percentile of age: beyond it the sample is
    # a handful of dropped-track epochs, and letting them set the scale would
    # compress the range where almost all the data actually lies.
    x_max = float(d[aoi_col].quantile(0.98))
    xs = np.linspace(0, x_max, 50)

    y_top = 0.0
    for q, col, label, c in ((0.9, "p90", "P90", S.CATEGORICAL[1]),
                             (0.5, "p50", "median", S.CATEGORICAL[0])):
        fit = sm.QuantReg(d["abs_err"], X).fit(q=q)
        floor, slope = float(fit.params.iloc[0]), float(fit.params.iloc[1])
        ax.plot(xs, floor + slope * xs, color=c, lw=1.5, ls=(0, (5, 3)), zorder=4)
        vis = binned[binned["aoi"] <= x_max]
        ax.plot(vis["aoi"], vis[col], color=c, marker="o", ms=6, lw=2.4, zorder=5,
                markeredgecolor=S.INK["surface"], markeredgewidth=1.3)
        ax.plot([0], [floor], marker="_", ms=14, color=c, zorder=6)
        ax.annotate(f"{label}:  {floor:.0f} m  +  {slope:.1f} m per s of age",
                    xy=(x_max * 0.52, floor + slope * x_max * 0.52),
                    xytext=(0, 11), textcoords="offset points",
                    fontsize=8, fontweight="medium", color=c,
                    path_effects=[__import__("matplotlib").patheffects.withStroke(
                        linewidth=2.5, foreground=S.INK["surface"])])
        y_top = max(y_top, floor + slope * x_max)

    ax.set_ylim(0, y_top * 1.18)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Measured Age of Information (s)")
    ax.set_ylabel("|closest-approach error| (m)")
    ax.set_title("Age adds to a floor — and hits the tail hardest")
    S.annotate_note(ax, f"quantile regression · intercept = age-independent floor · "
                        f"x capped at P98 age ({x_max:.1f} s)", loc="upper left")

    # ---- bottom: sense inversion rate
    ax = axes[1]
    inv = (df.assign(_b=pd.cut(df[aoi_col], list(bins)))
             .groupby("_b", observed=True)
             .agg(rate=(inversion_col, "mean"), n=(inversion_col, "size"))
             .dropna())
    labels = [f"{iv.left:g}–{iv.right:g}" for iv in inv.index]
    ax.bar(range(len(inv)), inv["rate"] * 100, color=S.CATEGORICAL[1], width=0.66, zorder=3)
    for i, (r, n) in enumerate(zip(inv["rate"], inv["n"], strict=True)):
        ax.text(i, r * 100, f"{r*100:.1f}%\nn={n:,}", ha="center", va="bottom",
                fontsize=7.5, color=S.INK["secondary"])
    ax.set_xticks(range(len(inv)), labels)
    ax.set_xlabel("Measured Age of Information (s)")
    ax.set_ylabel("Converging pairs reported as diverging (%)")
    ax.set_ylim(0, float(inv["rate"].max()) * 100 * 1.35)
    ax.set_title("Stale state inverts the sense of the encounter")

    S.finish(fig, axes)
    # The header sits a fixed distance above the axes, not a fixed fraction of
    # the figure, so a tall stacked layout keeps the same spacing as a wide one.
    h = fig.get_size_inches()[1]
    fig.suptitle(title, x=0.006, y=1 + 0.46 / h, ha="left", fontsize=11, fontweight="bold")
    if subtitle:
        fig.text(0.006, 1 + 0.15 / h, subtitle, ha="left", fontsize=8,
                 color=S.INK["secondary"])
    return fig, axes


# -------------------------------------------------------------------- F11

def fig_extrapolation_ablation(
    synthetic: pd.DataFrame,
    real: pd.DataFrame,
    heading_error_deg,
    *,
    filtered_median: float | None = None,
    filtered_inversion: pd.Series | None = None,
    show_mechanism: bool = True,
    bins=(0.0, 0.5, 1.0, 2.0, 5.0, 30.0),
    title: str = "Dead reckoning fixes the emulator and not the real system",
    subtitle: str | None = None,
    figsize=(12.4, 3.9),
):
    """F11 - three panels on whether compensation removes the freshness effect.

    Left: error with and without dead reckoning, emulated stream versus real
    stream.  Middle: sense inversion by age on the real stream, showing that
    compensation makes it worse where age is largest.  Right: the mechanism --
    the distribution of the heading error in the velocity the compensation
    relies on.

    ``synthetic`` needs ``extrapolate`` and ``horiz_error_m``; ``real`` needs
    ``extrapolate``, ``aoi_s``, ``d_cpa_error_m`` and ``observed_diverging``.
    """
    # The mechanism panel is a heading-error CDF; in print it reduces to three
    # numbers in the text, so it is optional.
    n_panels = 3 if show_mechanism else 2
    fig, axes = S.plt.subplots(1, n_panels, figsize=figsize)
    c_hold, c_dr = S.CATEGORICAL[0], S.CATEGORICAL[1]

    # ---- left: paired error, emulated vs real
    ax = axes[0]
    pairs = []
    s50 = synthetic.groupby("extrapolate")["horiz_error_m"].median()
    r = real.assign(abs_err=real["d_cpa_error_m"].abs())
    r50 = r.groupby("extrapolate")["abs_err"].median()
    pairs.append(("Emulated stream\n(exact velocity)", s50.get(False), s50.get(True)))
    pairs.append(("Real stream\n(reported velocity)", r50.get(False), r50.get(True)))

    xs = np.arange(len(pairs))
    w = 0.34
    ax.bar(xs - w / 2, [p[1] for p in pairs], w, color=c_hold, label="hold last state", zorder=3)
    ax.bar(xs + w / 2, [p[2] for p in pairs], w, color=c_dr, label="dead reckon", zorder=3)
    # Optional third arm: a filtered-velocity tracker, drawn only where supplied.
    if filtered_median is not None and np.isfinite(filtered_median):
        ax.bar([1 + 1.5 * w], [filtered_median], w * 0.8, color=S.CATEGORICAL[2],
               label="dead reckon (Kalman, +vel)", zorder=3)
        ax.text(1 + 1.5 * w, filtered_median, f"{filtered_median:.1f}", ha="center",
                va="bottom", fontsize=8, color=S.INK["secondary"])
    for i, (_, a, b) in enumerate(pairs):
        for off, v in ((-w / 2, a), (w / 2, b)):
            ax.text(i + off, v, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=8, color=S.INK["secondary"])
        drop = 100 * (1 - b / a) if a else np.nan
        ax.text(i, max(a, b) * 1.16, f"{drop:.0f}% of error removed", ha="center",
                fontsize=8.5, fontweight="bold",
                color=S.STATUS["good"] if drop > 20 else S.STATUS["critical"])
    ax.set_xticks(xs, [p[0] for p in pairs])
    ax.set_ylabel("median error (m)")
    ax.set_ylim(0, max(max(p[1], p[2]) for p in pairs) * 1.42)
    ax.set_title("Compensation works only on the emulator")
    ax.legend(loc="upper left")

    # ---- middle: sense inversion by age, real stream
    ax = axes[1]
    inv = (real.assign(_b=pd.cut(real["aoi_s"], list(bins)))
               .groupby(["_b", "extrapolate"], observed=True)["observed_diverging"]
               .mean().unstack() * 100).dropna(how="all")
    labels = [f"{iv.left:g}–{iv.right:g}" for iv in inv.index]
    xs = np.arange(len(inv))
    if filtered_inversion is not None:
        w3 = 0.26
        ax.bar(xs - w3, inv.get(False), w3, color=c_hold, label="hold last state", zorder=3)
        ax.bar(xs, inv.get(True), w3, color=c_dr, label="dead reckon", zorder=3)
        ax.bar(xs + w3, filtered_inversion.reindex(inv.index).to_numpy() , w3,
               color=S.CATEGORICAL[2], label="dead reckon (Kalman, +vel)", zorder=3)
    else:
        ax.bar(xs - w / 2, inv.get(False), w, color=c_hold, label="hold last state", zorder=3)
        ax.bar(xs + w / 2, inv.get(True), w, color=c_dr, label="dead reckon", zorder=3)
    ax.set_xticks(xs, labels)
    ax.set_xlabel("Measured Age of Information (s)")
    ax.set_ylabel("converging pairs read as diverging (%)")
    ax.set_title("…and makes it worse where age is largest")
    # One legend for the figure: the arms are identical across panels, so a
    # second copy only costs column width.

    if not show_mechanism:
        S.finish(fig, axes)
        fig.suptitle(title, x=0.006, y=1.11, ha="left", fontweight="bold")
        if subtitle:
            fig.text(0.006, 1.035, subtitle, ha="left", fontsize=7,
                     color=S.INK["secondary"])
        return fig, axes

    # ---- right: the mechanism
    ax = axes[2]
    h = np.sort(pd.Series(heading_error_deg).dropna().to_numpy(dtype=float))
    ax.plot(h, np.linspace(0, 1, h.size), color=S.CATEGORICAL[2], lw=2.2)
    for q, name in ((0.5, "P50"), (0.9, "P90"), (0.95, "P95")):
        v = float(np.quantile(h, q))
        ax.plot([v], [q], marker="o", ms=6, color=S.CATEGORICAL[2], zorder=4,
                markeredgecolor=S.INK["surface"], markeredgewidth=1.4)
        S.direct_label(ax, v, q, f"{name} = {v:.0f}°", S.INK["secondary"], dx=8,
                       dy=-11 if q >= 0.9 else 0)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("heading error of the reported velocity (°)")
    ax.set_ylabel("cumulative probability")
    ax.set_ylim(0, 1.03)
    ax.set_title("Why: the velocity it extrapolates along")
    S.annotate_note(ax, "measured at generation time — staleness excluded", loc="lower right")

    S.finish(fig, axes)
    fig.suptitle(title, x=0.004, y=1.11, ha="left", fontsize=11, fontweight="bold")
    if subtitle:
        fig.text(0.004, 1.035, subtitle, ha="left", fontsize=8, color=S.INK["secondary"])
    return fig, axes


# ------------------------------------------------------- paper figure 2

def fig_sense_inversion(
    bins_table: pd.DataFrame,
    *,
    bin_col: str = "bin",
    rate_col: str = "rate_%",
    lo_col: str = "ci_lo_%",
    hi_col: str = "ci_hi_%",
    n_col: str = "n",
    title: str = "Stale state inverts the sense of the encounter",
    figsize=None,
):
    """Paper figure 2 - converging pairs reported as diverging, by age.

    Standalone version of what was the right-hand panel of F10, with the
    cluster-robust intervals shown as error bars.  Splitting it out lets the
    error decomposition be reported as text (two regression lines are four
    numbers) while this, the concrete failure mode, keeps a figure to itself.
    """
    d = bins_table.copy()
    fig, ax = S.plt.subplots(figsize=figsize or S.ieee_size(1, 0.78))

    x = np.arange(len(d))
    rate = d[rate_col].to_numpy(float)
    err = np.vstack([rate - d[lo_col].to_numpy(float), d[hi_col].to_numpy(float) - rate])

    ax.bar(x, rate, width=0.62, color=S.CATEGORICAL[1], zorder=3)
    ax.errorbar(x, rate, yerr=err, fmt="none", ecolor=S.INK["primary"],
                elinewidth=0.8, capsize=2.2, zorder=4)

    # Sample size goes under the tick, not inside the bar: the low bins are too
    # short to hold rotated text, and the top bin being thin is exactly what the
    # reader needs to see.
    labels = [
        str(b).replace("(", "").replace("]", "").replace(", ", "–") + f"\n$n$={n:,}"
        for b, n in zip(d[bin_col], d[n_col], strict=True)
    ]
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", pad=2)
    ax.set_xlabel("Age of Information (s)")
    ax.set_ylabel("Converging pairs read\nas diverging (%)")
    ax.set_ylim(0, float(d[hi_col].max()) * 1.10)
    ax.set_title(title)

    S.finish(fig, ax)
    return fig, ax
