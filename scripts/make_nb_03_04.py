"""Generate notebooks 03-04: manoeuvre sensitivity and pairwise encounters."""

from __future__ import annotations

from nbbuild import HEADER, build, code, md

nb03 = [
    md("""
# 03 · Experiment E3 — does the cost of stale data depend on what the aircraft is doing?

**Question (RQ3, H2).** Constant-velocity reasoning makes stale state look
harmless. Real aircraft turn, climb and accelerate. Notebook 02 showed that
error is a clean function of Age of Information *when pooled*; this notebook
asks whether that function has the same slope in every flight phase.

Phases are derived from the 62 Hz reference trajectory itself (turn rate,
vertical speed, along-track acceleration) rather than from autopilot mode
flags, so the same rule can be applied to any dataset.
"""),
    code(HEADER),
    code("""
from telemetry_freshness.data.idf_ds import IdfDsDataset
from telemetry_freshness.evaluation.phases import (
    PhaseThresholds, label_phases, attach_phase, phase_summary)
from telemetry_freshness.telemetry.generator import TrajectorySampler, generate_stream
from telemetry_freshness.telemetry.network_model import TelemetryConfig
from telemetry_freshness.surveillance.state_store import SurveillanceStateStore
from telemetry_freshness.surveillance.age_of_information import freshness_errors
from telemetry_freshness.core.rng import make_rng

ds = IdfDsDataset()
LAPS = ds.scenarios()[:12]
TH = PhaseThresholds(turn_rate_deg_s=3.0, vertical_speed_mps=1.0, accel_mps2=0.5)
print("phase thresholds:", TH)
"""),
    code("""
rows = []
for lap in LAPS:
    traj = label_phases(ds.load_scenario(lap), TH)
    s = TrajectorySampler(traj)
    t_eval = np.arange(s.t_start + 10, s.t_end - 5, 0.2)
    for rate in (5, 2, 1):
        for lat_ms in (0, 250, 500):
            cfg = TelemetryConfig(update_rate_hz=rate, latency_ms=lat_ms)
            store = SurveillanceStateStore()
            store.ingest(generate_stream(s, cfg, make_rng(2026, f"{lap}|{rate}|{lat_ms}")))
            f = freshness_errors(store, s, t_eval)
            f = f[f.tracked]
            f = attach_phase(f, traj)
            f["scenario_id"] = lap; f["update_rate_hz"] = rate; f["latency_ms"] = lat_ms
            rows.append(f)

E3 = pd.concat(rows, ignore_index=True)
E3.to_parquet("results/raw/E3_maneuver_sensitivity.parquet", index=False)
print(f"{len(E3):,} observations")
display(E3.phase.value_counts(normalize=True).round(3).to_frame("share of flight time"))
"""),
    md("""
### Controlling for age

Phases differ in how much time they occupy *and* in the ages they experience.
Comparing raw error across phases would confound the two, so the comparison is
made inside a narrow AoI band.
"""),
    code("""
BAND = (0.9, 1.1)
band = E3[E3.aoi_s.between(*BAND)].copy()
print(f"AoI band {BAND[0]}-{BAND[1]} s   n = {len(band):,}")
summary = phase_summary(band)
display(summary.round(2))
summary.to_csv("results/processed/E3_phase_summary.csv", index=False)
"""),
    code("""
fig, ax = F.fig_error_by_flight_phase(
    band, phase_col="phase", err_col="horiz_error_m",
    title=f"Position error by flight phase at matched information age ({BAND[0]}-{BAND[1]} s)")
S.save(fig, "F1b_error_by_flight_phase"); S.plt.show()
"""),
    md("""
### The naive expectation does not survive contact with the data

The research plan anticipated that a turn would inflate position error. It does
not. At matched age, turning shows the *lowest* median position error of any
phase — because a turning aircraft's chord is shorter than the straight-line
distance a constant-velocity prediction assumes, so the stale position is
actually closer to the truth.

What a turn destroys is the **direction**, and two confounds have to be removed
before that is visible:

1. phases fly at different speeds (descent here is ~50% faster), so raw metres
   are not comparable — normalise error by speed, giving an *effective time lag*;
2. heading error must be reported separately from position error.
"""),
    code("""
band = band.copy()
# Error normalised by true speed: metres of error per metre/second of speed,
# i.e. the effective lag in seconds. This makes phases at different speeds
# directly comparable.
band["effective_lag_s"] = band["horiz_error_m"] / band["truth_speed_mps"].replace(0, np.nan)

cmp = (band.groupby("phase", observed=True)
       .agg(n=("horiz_error_m","size"),
            speed_mps=("truth_speed_mps","median"),
            pos_err_p50=("horiz_error_m","median"),
            pos_err_p95=("horiz_error_m", lambda s: s.quantile(.95)),
            eff_lag_p50=("effective_lag_s","median"),
            hdg_err_p50=("heading_error_deg","median"),
            hdg_err_p95=("heading_error_deg", lambda s: s.quantile(.95)))
       .round(2))
ref = cmp.loc["cruise"] if "cruise" in cmp.index else cmp.iloc[0]
cmp["pos_vs_cruise"] = (cmp.pos_err_p50 / ref.pos_err_p50).round(2)
cmp["hdg_vs_cruise"] = (cmp.hdg_err_p50 / ref.hdg_err_p50).round(2)
display(cmp)
cmp.to_csv("results/processed/E3_phase_summary.csv")
"""),
    code("""
fig, axes = S.plt.subplots(1, 2, figsize=(10.0, 3.8))
phases = [p for p in ("cruise","accel","climb","descent","turn") if p in set(band.phase.dropna())]
colors = S.categorical_colors(len(phases))

for ax, col, lbl, ttl in (
    (axes[0], "effective_lag_s", "Effective lag: error / speed (s)", "Position error, speed-normalised"),
    (axes[1], "heading_error_deg", "Heading error (deg)", "Heading error"),
):
    data = [band.loc[band.phase == p, col].dropna().to_numpy() for p in phases]
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c); body.set_alpha(0.35)
        body.set_edgecolor(c); body.set_linewidth(1.2)
    for i, (v, c) in enumerate(zip(data, colors), start=1):
        if not len(v): continue
        m = float(np.median(v))
        ax.plot([i], [m], marker="o", ms=6, color=c, zorder=4,
                markeredgecolor=S.INK["surface"], markeredgewidth=1.5)
        ax.annotate(f"{m:.2f}", (i, m), textcoords="offset points", xytext=(9, 0),
                    fontsize=7.5, color=S.INK["secondary"], va="center")
    ax.set_xticks(range(1, len(phases)+1)); ax.set_xticklabels([p.title() for p in phases])
    ax.set_ylabel(lbl); ax.set_title(ttl)

S.finish(fig, axes); S.save(fig, "F1b_phase_position_vs_heading"); S.plt.show()
"""),
    md("""
Once speed is controlled for, the phases sit within a narrow band on position
error, while heading error separates them sharply. The operational consequence
is that manoeuvring degrades the *predicted track* rather than the current
position — which is why the conflict-detection experiments key on TTC error
(notebook 04) rather than on position error alone.
"""),
    md("## Error growth with age, per phase"),
    code("""
fig, ax = S.plt.subplots(figsize=(6.6, 4.2))
phases = [p for p in ("cruise","accel","climb","descent","turn") if p in set(E3.phase.dropna())]
colors = S.categorical_colors(len(phases))
slopes = {}
for p, c in zip(phases, colors):
    g = E3[E3.phase == p].dropna(subset=["aoi_s","heading_error_deg"])
    if len(g) < 200: continue
    b = (g.assign(_b=pd.qcut(g.aoi_s, 12, duplicates="drop"))
           .groupby("_b", observed=True)
           .agg(aoi=("aoi_s","median"), err=("heading_error_deg","median")).dropna())
    ax.plot(b.aoi, b.err, color=c, marker="o", ms=3.5, zorder=3)
    S.direct_label(ax, b.aoi.iloc[-1], b.err.iloc[-1], p, c)
    slopes[p] = float(np.polyfit(b.aoi, b.err, 1)[0])
ax.set_xlabel("Age of Information (s)"); ax.set_ylabel("Heading error, median (deg)")
ax.set_title("Heading knowledge decays far faster in a turn")
ax.legend(handles=[S.plt.Line2D([], [], color=c, lw=2, label=p) for p, c in zip(phases, colors)],
          title="Flight phase", loc="upper left")
S.finish(fig, ax); S.save(fig, "S7_heading_error_growth_by_phase"); S.plt.show()

display(pd.Series(slopes, name="heading-error growth (deg per second of age)").round(2).to_frame())
"""),
    md("""
> **Finding (revises H2).** The research plan's hypothesis was that manoeuvring
> amplifies the position error caused by stale data. On these real fixed-wing
> flights it does not: at matched information age, turning produces the
> *smallest* median position error. What manoeuvring destroys is heading
> knowledge, and therefore the extrapolated track. H2 should be restated in
> terms of predicted-track error rather than position error.
"""),
]

nb04 = [
    md("""
# 04 · Experiment E1 — pairwise encounters: rate × latency × geometry

**Questions.** RQ1 (how does AoI affect TTC estimation and alert timing) and
RQ2 (do update rates that suffice for tracking still suffice for *tactical*
conflict detection).

**Design.** Controlled two-aircraft encounters so the truth is known exactly.
Every configuration varies one factor family at a time against a fixed
geometry, per research plan §22. Monte-Carlo repetitions vary the seed only.

The detector is run twice over identical criteria — once on ground truth, once
on the surveillance store — so any difference is attributable to information
age and nothing else.
"""),
    code(HEADER),
    code("""
from telemetry_freshness.simulation.encounters import EncounterSpec, synthetic_encounter
from telemetry_freshness.experiments.runner import ScenarioConfig, run_scenario
from telemetry_freshness.telemetry.network_model import TelemetryConfig
from telemetry_freshness.conflict.detector import ConflictCriteria

CRITERIA = ConflictCriteria(horizontal_separation_m=30.0, vertical_separation_m=15.0, horizon_s=30.0)
GEOMETRIES = ["head_on", "crossing", "converging", "overtaking"]
RATES_HZ   = [10, 5, 2, 1, 0.5]
LATENCY_MS = [0, 100, 250, 500, 1000]
N_SEEDS    = 12

print("separation standard:", CRITERIA)
print(f"{len(GEOMETRIES)}×{len(RATES_HZ)}×{len(LATENCY_MS)}×{N_SEEDS} = "
      f"{len(GEOMETRIES)*len(RATES_HZ)*len(LATENCY_MS)*N_SEEDS} runs")
"""),
    md("""
### Encounter geometries

All four are designed to conflict (0 m nominal miss distance) so that recall is
measurable; the aircraft speeds differ per geometry to give a realistic spread
of closing speeds.
"""),
    code("""
specs = {
    "head_on":    EncounterSpec("head_on",    speed_a_mps=15, speed_b_mps=15, lead_in_s=45, duration_s=90),
    "crossing":   EncounterSpec("crossing",   speed_a_mps=15, speed_b_mps=12, lead_in_s=45, duration_s=90),
    "converging": EncounterSpec("converging", speed_a_mps=14, speed_b_mps=14, lead_in_s=45, duration_s=90),
    "overtaking": EncounterSpec("overtaking", speed_a_mps=18, speed_b_mps=9,  lead_in_s=45, duration_s=90),
}
fig, axes = S.plt.subplots(1, 4, figsize=(11.5, 3.0), sharey=True)
for ax, (name, sp) in zip(axes, specs.items()):
    tr = synthetic_encounter(sp)
    for uid, c in zip(sorted(tr.uas_id.unique()), S.categorical_colors(2)):
        g = tr[tr.uas_id == uid]
        ax.plot(g.x_m, g.y_m, color=c, lw=1.8)
        ax.plot(g.x_m.iloc[0], g.y_m.iloc[0], marker="o", ms=5, color=c,
                markeredgecolor=S.INK["surface"], markeredgewidth=1.2)
    ax.set_title(name.replace("_", " ")); ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("East (m)")
axes[0].set_ylabel("North (m)")
S.finish(fig, axes); S.save(fig, "S8_encounter_geometries"); S.plt.show()
"""),
    code("""
from itertools import product
runs = []
for geom, rate, lat_ms, seed in product(GEOMETRIES, RATES_HZ, LATENCY_MS, range(N_SEEDS)):
    traj = synthetic_encounter(specs[geom])
    cfg = ScenarioConfig(
        scenario_id=f"{geom}",
        telemetry=TelemetryConfig(update_rate_hz=rate, latency_ms=lat_ms,
                                  jitter_ms=0.1*lat_ms, latency_model="fixed"),
        criteria=CRITERIA, decision_rate_hz=5.0, seed=1000 + seed,
    )
    r = run_scenario(traj, cfg)
    if r.empty: continue
    r["encounter_type"] = geom
    r["seed_index"] = seed
    runs.append(r)

E1 = pd.concat(runs, ignore_index=True)
E1.to_parquet("results/raw/E1_pairwise_sweep.parquet", index=False)
print(f"{len(E1):,} (decision time, pair) records over {E1.groupby(['encounter_type','update_rate_hz','latency_ms','seed_index']).ngroups} runs")
"""),
    md("## Realised information age versus the configured rate — Figure F8"),
    code("""
fig, ax = F.fig_rate_vs_realised_aoi(E1)
S.save(fig, "F8_rate_vs_realised_aoi"); S.plt.show()

display(E1.groupby(["update_rate_hz","latency_ms"]).aoi_s
        .agg(p50="median", p95=lambda s: s.quantile(.95)).round(3).unstack().round(2))
"""),
    md("""
The nominal expectation (half the sampling interval) only holds at zero added
latency. Once transport delay is present the realised age is dominated by it,
which is precisely why the paper reports AoI rather than update rate.
"""),
    md("## Signed TTC error — Figure F3"),
    code("""
fig, ax = F.fig_ttc_error_vs_aoi(E1, level_col="encounter_type")
S.save(fig, "F3_ttc_error_vs_aoi"); S.plt.show()

ttc = E1.dropna(subset=["ttc_error_signed_s"])
# signed = truth - observed, so NEGATIVE is the dangerous direction: the
# observed TTC exceeds reality and the system believes it has more time.
display(ttc.groupby("encounter_type").ttc_error_signed_s.agg(
    n="size", mean="mean", p05=lambda s: s.quantile(.05),
    frac_ttc_overestimated=lambda s: float((s < 0).mean())).round(3))
print(f"Overall: {(ttc.ttc_error_signed_s < 0).mean():.1%} of comparable estimates "
      f"overestimate TTC (mean {ttc.ttc_error_signed_s.mean():.2f} s).")
"""),
    md("## Rate × latency grid — Figure F5"),
    code("""
fig, ax = F.fig_rate_latency_heatmap(E1, value_col="ttc_error_abs_s", agg="p95")
S.save(fig, "F5_rate_latency_heatmap"); S.plt.show()
"""),
    md("## Detection quality and alert timing"),
    code("""
from telemetry_freshness.evaluation.metrics import recall_by_aoi, detection_counts
from telemetry_freshness.evaluation.timing import timing_table

overall = detection_counts(E1.conflict_truth, E1.conflict_observed)
display(pd.Series(overall.as_dict()).to_frame("all configurations").T)

rec = recall_by_aoi(E1)
display(rec[["aoi_bin","n","n_positive","recall","precision"]].round(3))
rec.to_csv("results/processed/E1_recall_by_aoi.csv", index=False)
"""),
    code("""
timing = timing_table(
    E1, group_cols=("encounter_type","update_rate_hz","latency_ms","seed_index","uas_a","uas_b"),
    separation_threshold_m=CRITERIA.horizontal_separation_m)
timing.to_parquet("results/raw/E1_timing.parquet", index=False)

t = timing.dropna(subset=["alert_lead_time_s"])
display(t.groupby(["update_rate_hz","latency_ms"]).agg(
    n=("alert_lead_time_s","size"),
    lead_p50=("alert_lead_time_s","median"),
    lead_p05=("alert_lead_time_s", lambda s: s.quantile(.05)),
    delay_p50=("detection_delay_s","median"),
    missed=("missed","sum")).round(2).unstack(fill_value=np.nan)["lead_p05"].round(2))
print("\\nTable: 5th-percentile alert lead time (s) by rate (rows) and latency (cols).")
"""),
    md("## Alert lead time against age — Figure F4"),
    code("""
from telemetry_freshness.evaluation.metrics import bin_by_aoi

# Attach each encounter's realised median AoI to its timing outcome.
aoi_per_run = (E1.groupby(["encounter_type","update_rate_hz","latency_ms","seed_index"])
                 .aoi_s.median().rename("aoi_s").reset_index())
tm = timing.merge(aoi_per_run, on=["encounter_type","update_rate_hz","latency_ms","seed_index"])
binned = bin_by_aoi(tm.dropna(subset=["alert_lead_time_s"]))
curve = (binned.groupby(["encounter_type","aoi_bin"], observed=True)
         .agg(aoi_bin_mid=("aoi_bin_mid","first"),
              lead_time_p5_s=("alert_lead_time_s", lambda s: np.percentile(s, 5)),
              n=("alert_lead_time_s","size")).reset_index())
curve = curve[curve.n >= 5]

fig, ax = F.fig_lead_time_vs_aoi(curve, level_col="encounter_type", min_lead_s=3.0)
S.save(fig, "F4_lead_time_vs_aoi"); S.plt.show()
curve.to_csv("results/processed/E1_lead_time_by_aoi.csv", index=False)
"""),
    md("""
> **Assumption.** The 3 s minimum manoeuvring time is an experiment parameter,
> not a regulatory value. It is carried explicitly so the threshold in
> notebook 06 can be re-derived under any other choice.
"""),
]

if __name__ == "__main__":
    for name, cells in (
        ("03_maneuver_sensitivity", nb03),
        ("04_pairwise_encounters", nb04),
    ):
        print("wrote", build(name, cells))
