"""Generate notebooks 00-02: environment, BUBBLES freshness, IDF-DS downsampling."""

from __future__ import annotations

from nbbuild import HEADER, build, code, md

# ---------------------------------------------------------------- 00

nb00 = [
    md("""
# 00 · Environment and dataset check

Confirms the toolchain and that all four datasets parse into the unified schema.
Run this first; every other notebook assumes it passes.

See `docs/DATASETS.md` for what each dataset is and what it may be used to claim.
"""),
    code(HEADER),
    code("""
import platform, importlib
print("python  :", platform.python_version())
for m in ["numpy","scipy","pandas","pyarrow","matplotlib","seaborn","pyproj","statsmodels"]:
    print(f"{m:12s}:", importlib.import_module(m).__version__)
"""),
    md("## Geodesy check\n\nThe ENU transform must be metrically exact — conflict geometry depends on it."),
    code("""
from telemetry_freshness.core.geodesy import GeodeticFrame
f = GeodeticFrame(39.5970, -0.2764, 30.0)
lat = np.array([39.5970, 39.5980, 39.5970]); lon = np.array([-0.2764, -0.2764, -0.2754])
alt = np.array([30.0, 30.0, 80.0])
e, n, u = f.to_enu(lat, lon, alt)
back = f.to_geodetic(e, n, u)
print("0.001 deg north ->", round(float(n[1]), 2), "m   (expect ~111.0)")
print("0.001 deg east  ->", round(float(e[2]), 2), "m   (expect ~85.9 at 39.6N)")
print("50 m up         ->", round(float(u[2]), 2), "m")
print("round-trip error (deg):", float(np.max(np.abs(back[0] - lat))))
"""),
    md("## Dataset availability\n\nEach adapter is exercised on one scenario."),
    code("""
from telemetry_freshness.data.bubbles import BubblesDataset
from telemetry_freshness.data.idf_ds import IdfDsDataset
from telemetry_freshness.data.icmcis import IcmcisDataset
from telemetry_freshness.data.uagree import UAgreeOccupancy
from telemetry_freshness.core.schema import validate_trajectory_frame

rows = []

b = BubblesDataset(); bs = b.scenarios()[0]; bgt = b.load_scenario(bs)
validate_trajectory_frame(bgt)
rows.append(["BUBBLES", bs, len(bgt), bgt.uas_id.nunique(), round(bgt.t_s.max()-bgt.t_s.min()), "ground truth + UTM tracks"])

i = IdfDsDataset(); isc = i.scenarios()[0]; igt = i.load_scenario(isc)
validate_trajectory_frame(igt)
rows.append(["IDF-DS", isc, len(igt), igt.uas_id.nunique(), round(igt.t_s.max()), f"{i.reference_rate_hz(isc):.0f} Hz reference"])

c = IcmcisDataset(); csc = "train/Scenario_1_1"; cgt = c.load_scenario(csc)
validate_trajectory_frame(cgt)
rows.append(["ICMCIS", csc, len(cgt), cgt.uas_id.nunique(), round(cgt.t_s.max()), "truth + radar"])

u = UAgreeOccupancy(); s = u.summary()
rows.append(["U-AGREE", "occupancy", s["n_rows"], s["n_operational_intents"], round(s["campaign_duration_s"]), "4-D occupancy, NOT telemetry"])

display(pd.DataFrame(rows, columns=["dataset","scenario","rows","aircraft/intents","duration_s","role"]))
"""),
    md("## Mechanism self-test\n\nThe analytic invariants the study rests on."),
    code("""
import subprocess, sys
r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"], capture_output=True, text=True)
print(r.stdout[-1500:] or r.stderr[-1500:])
"""),
]

# ---------------------------------------------------------------- 01

nb01 = [
    md("""
# 01 · Measured information freshness in a real U-space system (BUBBLES)

This notebook does something the research plan assumed would have to be
*emulated*: it **measures** Age of Information directly, in an operational
U-space validation environment, on **two independent campaign days**.

**Age of Information** is read directly from the UTM track records as
`t − t_last`, where the dataset documentation defines `t_last` as *"Timestamp
from the last message received from the drone"*. Crucially this is a
**single-clock** difference, computed inside the UTM's own time base, so it is
immune to synchronisation error between aircraft and ground.

> **Retraction.** An earlier version of this notebook also reported a measured
> one-way latency as `time_ms − message.time`. The dataset documentation
> describes `time_ms` as *"Arrival time to the MQTT server (**ignore**)"*, and
> the data bears that out: day 2 yields **negative** one-way delays for every
> aircraft. That result has been withdrawn and is not used anywhere in this
> study. The evidence for the withdrawal is retained below, because the failure
> mode is itself worth reporting.

> **Claim discipline.** BUBBLES is a real campaign, but with 13–14 aircraft over
> several km² it is *not* a high-density scenario. It bounds the low end of the
> density axis; it does not stand in for dense operations.
"""),
    code(HEADER),
    code("""
from telemetry_freshness.data.bubbles import BubblesDataset
from telemetry_freshness.evaluation.density import airborne_only, density_timeline

ds = BubblesDataset()
DAYS = ds.scenarios()
truth, obs = {}, {}
for d in DAYS:
    truth[d] = ds.load_scenario(d)
    o = ds.load_observations(d)
    obs[d] = o[~o.is_probe]

display(pd.DataFrame([{
    "day": d,
    "telemetry_messages": len(truth[d]),
    "aircraft": truth[d].uas_id.nunique(),
    "duration_h": round((truth[d].t_s.max()-truth[d].t_s.min())/3600, 2),
    "utm_track_rows": len(obs[d]),
    "utm_conflict_alerts": int(obs[d].tactical_conflict.sum()),
} for d in DAYS]))
"""),
    md("""
### Data-integrity note

Day 1's telemetry logs contain **injected system-test probe messages** mixed in
with real flight data: a placeholder payload (attitude 4/4/4, ground speed
exactly 20, `pilot_alerts` as the string `"none"`), an uptime rather than epoch
clock, and a position ~40 km NW of the site. The adapter excludes them by
default. Day 2 contains none.
"""),
    code("""
probe = BubblesDataset(include_probe_messages=True)
for d in DAYS:
    g = probe.load_scenario(d)
    n = int(g.is_probe.sum())
    print(f"{d}: {n:,} probe messages excluded from ground truth, "
          f"{int(ds.load_observations(d).is_probe.sum()):,} probe tracks flagged")
"""),
    md("## Measured Age of Information — the headline quantity"),
    code("""
rows = []
for d in DAYS:
    a = obs[d].aoi_s.dropna()
    rows.append({"day": d, "n": len(a), "mean": a.mean(), "p50": a.median(),
                 "p75": a.quantile(.75), "p90": a.quantile(.90), "p95": a.quantile(.95),
                 "p99": a.quantile(.99), "max": a.max()})
AOI = pd.DataFrame(rows).round(3)
display(AOI)
AOI.to_csv("results/processed/C1_aoi_by_day.csv", index=False)
"""),
    md("""
**The centre replicates; the tail does not.** Median AoI is 0.597 s and 0.567 s
on the two days — a 5% difference. The P99 is 18.3 s and 98.8 s — a factor of
five. Both days are heavy-tailed, but *how* heavy is not a stable property of
the system.

This is the first of three places in this study where a distribution's centre is
well-behaved and its tail is not, and it is why every requirement in this work
is stated against a tail statistic.
"""),
    code("""
for d in DAYS:
    fig, ax = F.fig_aoi_distribution(
        obs[d].aoi_s,
        title=f"Measured Age of Information — {d.replace('_','-')}",
        label=f"BUBBLES · {truth[d].uas_id.nunique()} aircraft",
        reference={"1 Hz nominal": 1.0})
    S.save(fig, f"F0_bubbles_aoi_cdf_{d}"); S.plt.show()
"""),
    code("""
per_ac = []
for d in DAYS:
    t = (obs[d].groupby("uas_id").aoi_s
         .agg(n="size", p50="median", p95=lambda s: s.quantile(.95),
              p99=lambda s: s.quantile(.99)).round(3))
    t["day"] = d
    per_ac.append(t)
display(pd.concat(per_ac).sort_values(["day","p95"], ascending=[True,False]))
"""),
    md("""
## Why the cross-clock latency measurement was withdrawn

Retained as evidence for the retraction above, and because it is a caution for
anyone attempting to measure information age in the field.
"""),
    code("""
rows = []
for d in DAYS:
    l = truth[d].latency_s.dropna()
    # Jitter = deviation from each aircraft's own median, so a constant
    # per-aircraft clock offset cancels and only the variable part survives.
    j = (truth[d].dropna(subset=["latency_s"])
         .groupby("uas_id").latency_s.transform(lambda s: s - s.median()).abs())
    rows.append({"day": d, "n": len(l),
                 "latency_p50": l.median(), "latency_p95": l.quantile(.95),
                 "latency_max": l.max(),
                 "jitter_p50": j.median(), "jitter_p95": j.quantile(.95)})
LAT = pd.DataFrame(rows).round(3)
display(LAT)
LAT.to_csv("results/processed/C1_latency_by_day.csv", index=False)
"""),
    code("""
for d in DAYS:
    t = (truth[d].dropna(subset=["latency_s"]).groupby("uas_id").latency_s
         .agg(n="size", p50="median", iqr=lambda s: s.quantile(.75)-s.quantile(.25)).round(3))
    print(f"=== {d} — per-aircraft median latency (s) ===")
    print(f"  all medians {'POSITIVE' if (t.p50>0).all() else 'NEGATIVE'}; "
          f"range {t.p50.min():+.3f} to {t.p50.max():+.3f}; median IQR {t.iqr.median():.3f}")
    display(t)
"""),
    md("""
**Day 2's latencies are all negative** — every aircraft, clustered tightly
(median per-aircraft IQR 0.05 s) around roughly −1.6 s, with one at −2.88 s. A
negative one-way delay is not physical. The tight per-aircraft spread shows this
is a **constant clock offset per airframe**, not variable transport delay, and
the documentation's instruction to ignore `time_ms` explains why.

Note what is *not* affected. The variable component — jitter, measured as
deviation from each aircraft's own median, so a constant offset cancels —
replicates across both days (P95 0.292 s versus 0.430 s). And **AoI is
untouched on either day**, because it never crosses a clock boundary.

**Methodological finding.** Measuring information age by cross-clock
differencing is fragile in a real multi-operator deployment: one of two days was
unusable for it, and nothing in the data itself would have flagged that had the
documentation not been consulted. The single-clock `t − t_last` construction is
the robust instrument, and field measurement of AoI should prefer it.
"""),
    code("""
d = DAYS[0]
lat = truth[d].latency_s.dropna()
lat = lat[lat.between(-1, 30)]
fig, ax = S.plt.subplots(figsize=(6.4, 3.8))
ax.hist(lat, bins=120, color=S.CATEGORICAL[0], alpha=0.85, edgecolor="none")
for q, lbl in ((.5, "P50"), (.95, "P95")):
    v = lat.quantile(q)
    ax.axvline(v, color=S.STATUS["critical"], lw=1.2, ls=(0, (4, 3)))
    ax.text(v, ax.get_ylim()[1]*0.92, f" {lbl}={v:.2f}s", fontsize=7.5,
            color=S.STATUS["critical"])
ax.set_xlabel("One-way telemetry latency (s)"); ax.set_ylabel("Messages")
ax.set_title("Cross-clock delay, day 1 — retained as evidence, not used as a result")
S.annotate_note(ax, f"BUBBLES {d} · n = {len(lat):,} · WITHDRAWN: `time_ms` is documented as 'ignore'")
S.finish(fig, ax); S.save(fig, "S3_withdrawn_crossclock_delay"); S.plt.show()
"""),
    md("## Traffic density actually achieved"),
    code("""
rows = []
for d in DAYS:
    air = airborne_only(truth[d])
    dt = density_timeline(air, step_s=60)
    dt = dt[dt.n_aircraft >= 2]
    rows.append({"day": d, "windows": len(dt),
                 "aircraft_max": int(dt.n_aircraft.max()),
                 "density_min": dt.density_per_km2.min(),
                 "density_median": dt.density_per_km2.median(),
                 "density_max": dt.density_per_km2.max()})
display(pd.DataFrame(rows).round(2))
print("Real density is ~0.4-3 UAS/km2 on both days: a real campaign, but not a")
print("dense one. High-density conditions must be extrapolated, and labelled so.")
"""),
    code("""
fig, ax = F.fig_trajectory_map(airborne_only(truth[DAYS[0]]),
                               title=f"BUBBLES campaign {DAYS[0]} (6 of 14 aircraft)")
S.save(fig, "S2_bubbles_tracks"); S.plt.show()
"""),
    md("""
## Link-delay pool for the validity check

Exported for the link-model validity check (notebook 12). **Day 1 only.**

Because it derives from the withdrawn `time_ms` field, this pool is used
**only** to test whether the *shape* of a delay distribution changes conclusions
relative to a constant of the same mean — a question about modelling choices,
not a claim about this deployment's true latency. No result in the paper depends
on its absolute level.
"""),
    code("""
from pathlib import Path
src = truth[DAYS[0]].latency_s.dropna()
pool = src[src.between(0, 10)].to_numpy()
Path("data/processed").mkdir(parents=True, exist_ok=True)
np.save("data/processed/bubbles_latency_pool_s.npy", pool)
prov = ds.provenance(DAYS[0])
prov.notes.append("latency pool: day 1 only; day 2 excluded for per-aircraft clock offsets")
prov.to_json("data/processed/bubbles_latency_pool.provenance.json")
print(f"saved {pool.size:,} measured latency samples from {DAYS[0]}")
print(f"  mean {pool.mean():.3f}  p50 {np.median(pool):.3f}  "
      f"p95 {np.percentile(pool,95):.3f}  p99 {np.percentile(pool,99):.3f}  max {pool.max():.2f}")
"""),
]

# ---------------------------------------------------------------- 02

nb02 = [
    md("""
# 02 · Experiment E2 — telemetry downsampling on real flights (IDF-DS)

**Question (RQ1).** How does surveillance Age of Information translate into
error in the aircraft state the UTM is working from?

**Ground truth.** The IDF-DS SpeedyBee blackbox `navPos`/`navVel` stream at
~62 Hz. That rate matters: it resolves the 100–200 ms information-age regime,
which the original dataset plan had written off as unmeasurable.

**Method.** Resample the reference trajectory at 10 / 5 / 2 / 1 / 0.5 Hz, apply
added network latency, hold the latest received state, and compare against the
reference at a dense grid of decision times.

Two displacement quantities are reported and they are *not* the same:
`stale displacement` is the measured truth-vs-observed distance, while
`v · AoI` is the constant-velocity bound the research plan writes down (§7).
Comparing them shows where that bound breaks.
"""),
    code(HEADER),
    code("""
from telemetry_freshness.data.idf_ds import IdfDsDataset
from telemetry_freshness.telemetry.generator import TrajectorySampler, generate_stream
from telemetry_freshness.telemetry.network_model import TelemetryConfig
from telemetry_freshness.surveillance.state_store import SurveillanceStateStore
from telemetry_freshness.surveillance.age_of_information import freshness_errors
from telemetry_freshness.core.rng import make_rng

ds = IdfDsDataset()
LAPS = ds.scenarios()[:12]          # 12 real flights
RATES_HZ  = [10, 5, 2, 1, 0.5]
LATENCY_MS = [0, 100, 250, 500, 1000]
print(f"{len(LAPS)} laps · reference rate {ds.reference_rate_hz(LAPS[0]):.1f} Hz")
"""),
    code("""
records = []
for lap in LAPS:
    traj = ds.load_scenario(lap)
    s = TrajectorySampler(traj)
    t_eval = np.arange(s.t_start + 10, s.t_end - 5, 0.2)
    for rate in RATES_HZ:
        for lat_ms in LATENCY_MS:
            cfg = TelemetryConfig(update_rate_hz=rate, latency_ms=lat_ms,
                                  randomise_phase=True)
            store = SurveillanceStateStore()
            store.ingest(generate_stream(s, cfg, make_rng(2026, f"{lap}|{rate}|{lat_ms}")))
            f = freshness_errors(store, s, t_eval)
            f = f[f.tracked].copy()
            f["scenario_id"] = lap
            f["update_rate_hz"] = rate
            f["latency_ms"] = lat_ms
            records.append(f)

E2 = pd.concat(records, ignore_index=True)
print(f"{len(E2):,} decision-time observations")
E2.to_parquet("results/raw/E2_idfds_downsampling.parquet", index=False)
"""),
    md("### Summary across the sweep"),
    code("""
summary = (E2.groupby(["update_rate_hz","latency_ms"])
    .agg(aoi_p50=("aoi_s","median"), aoi_p95=("aoi_s", lambda s: s.quantile(.95)),
         err_p50=("horiz_error_m","median"), err_p95=("horiz_error_m", lambda s: s.quantile(.95)),
         err_max=("horiz_error_m","max"),
         pred_p95=("predicted_displacement_m", lambda s: s.quantile(.95)),
         hdg_p95=("heading_error_deg", lambda s: s.quantile(.95)))
    .round(2).reset_index())
display(summary)
summary.to_csv("results/processed/E2_summary.csv", index=False)
"""),
    md("## Figure F1 — Age of Information vs position error"),
    code("""
fig, ax = F.fig_aoi_vs_position_error(
    E2, level_col="update_rate_hz", aoi_col="aoi_s", err_col="horiz_error_m",
    subtitle=f"IDF-DS · {len(LAPS)} real fixed-wing flights · {ds.reference_rate_hz(LAPS[0]):.0f} Hz reference",
)
S.save(fig, "F1_aoi_vs_position_error"); S.plt.show()
"""),
    md("""
## Does `v · AoI` bound the real displacement?

The research plan proposes `D_stale = v · AoI` as an interpretable measure. On a
straight leg it is exact; the question is what happens during real manoeuvres.
"""),
    code("""
cmp = E2.dropna(subset=["predicted_displacement_m","horiz_error_m"]).copy()
cmp["ratio"] = cmp["horiz_error_m"] / cmp["predicted_displacement_m"].replace(0, np.nan)
r = cmp["ratio"].replace([np.inf,-np.inf], np.nan).dropna()
display(pd.Series({"p05": r.quantile(.05), "p50": r.median(), "p95": r.quantile(.95),
                   "p99": r.quantile(.99), "frac_exceeding_bound": float((r > 1).mean())
                  }).round(3).to_frame("measured error / (v · AoI)"))

fig, ax = S.plt.subplots(figsize=(6.4, 4.0))
sub = cmp.sample(min(20000, len(cmp)), random_state=0)
ax.scatter(sub.predicted_displacement_m, sub.horiz_error_m, s=3, alpha=0.12,
           color=S.CATEGORICAL[0], edgecolors="none")
lim = float(np.nanpercentile(sub.predicted_displacement_m, 99.5))
ax.plot([0, lim], [0, lim], color=S.INK["secondary"], lw=1.2, ls=(0,(4,3)))
ax.text(lim, lim, "  y = x", fontsize=7.5, color=S.INK["secondary"], va="center")
ax.set_xlim(0, lim); ax.set_ylim(0, lim*1.4)
ax.set_xlabel("Predicted displacement  v · AoI  (m)")
ax.set_ylabel("Measured position error (m)")
ax.set_title("The constant-velocity bound is not conservative during manoeuvres")
S.finish(fig, ax); S.save(fig, "S5_bound_vs_measured"); S.plt.show()
"""),
    md("## Velocity and heading error"),
    code("""
fig, axes = S.plt.subplots(1, 2, figsize=(9.2, 3.8))
for ax, col, lbl in ((axes[0], "vel_error_mps", "Velocity error (m/s)"),
                     (axes[1], "heading_error_deg", "Heading error (deg)")):
    rates = sorted(E2.update_rate_hz.unique(), reverse=True)
    for rate, c in zip(rates, S.sequential_colors(len(rates))):
        g = E2[E2.update_rate_hz == rate]
        b = (g.assign(_b=pd.qcut(g.aoi_s, 10, duplicates="drop"))
               .groupby("_b", observed=True)
               .agg(aoi=("aoi_s","median"), v=(col,"median")).dropna())
        ax.plot(b.aoi, b.v, color=c, marker="o", ms=3.5)
        S.direct_label(ax, b.aoi.iloc[-1], b.v.iloc[-1], f"{rate:g} Hz", c)
    ax.set_xlabel("Age of Information (s)"); ax.set_ylabel(lbl)
axes[0].set_title("Velocity error"); axes[1].set_title("Heading error")
S.finish(fig, axes); S.save(fig, "S6_velocity_heading_error"); S.plt.show()
"""),
]

if __name__ == "__main__":
    for name, cells in (
        ("00_environment_and_data_check", nb00),
        ("01_bubbles_measured_freshness", nb01),
        ("02_idfds_downsampling", nb02),
    ):
        print("wrote", build(name, cells))
