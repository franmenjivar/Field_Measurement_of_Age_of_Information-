"""Generate notebooks 05-08: density, safe-AoI threshold, ICMCIS, paper figures."""

from __future__ import annotations

from nbbuild import HEADER, build, code, md

nb05 = [
    md("""
# 05 · Experiments E4/E5 — density scaling and interaction density

**Questions.** RQ2 and RQ3: do the update rates that suffice at low traffic
still suffice as density rises, and is the freshness requirement better
predicted by raw density or by *interaction* density?

**Where the traffic comes from.** BUBBLES is the anchor — 14 real aircraft, and
notebook 01 measured its realised density at only **0.4–3.1 UAS/km²**. That is a
real U-space campaign but it is *not* dense. Everything above it is generated
by the traffic simulator, with speeds drawn from the BUBBLES empirical
distribution so the extrapolation inherits real speed structure.

> Figures in this notebook must keep the real anchor point visually distinct
> from the synthetic extrapolation. The paper may not describe the high-density
> points as observed.
"""),
    code(HEADER),
    code("""
from telemetry_freshness.data.bubbles import BubblesDataset
from telemetry_freshness.evaluation.density import (
    airborne_only, density_timeline, local_density, interaction_rate, LOCAL_RADII_M)
from telemetry_freshness.simulation.traffic_generator import (
    SpeedProfile, TrafficConfig, generate_traffic)
from telemetry_freshness.experiments.runner import ScenarioConfig
from telemetry_freshness.experiments.fast_runner import run_scenario_fast
from telemetry_freshness.telemetry.network_model import TelemetryConfig
from telemetry_freshness.conflict.detector import ConflictCriteria

CRITERIA = ConflictCriteria(horizontal_separation_m=30.0, vertical_separation_m=15.0, horizon_s=30.0)

bub = BubblesDataset(); DAY = "2022_04_11"
real = airborne_only(bub.load_scenario(DAY))
profile = SpeedProfile.from_trajectories(real, source=f"BUBBLES {DAY} (14 real UAS)")
print("speed profile from real flights:", {k: round(v,2) if isinstance(v,float) else v
                                           for k, v in profile.describe().items()})
"""),
    md("## The real anchor: density actually achieved in BUBBLES"),
    code("""
real_dens = density_timeline(real, step_s=60)
real_dens = real_dens[real_dens.n_aircraft >= 2]
display(real_dens[["n_aircraft","area_km2","density_per_km2"] +
                  [f"local_mean_{int(r)}m" for r in LOCAL_RADII_M]]
        .describe().loc[["min","50%","max"]].round(2))
"""),
    md("""
## Synthetic density sweep

Aircraft count is varied over a fixed 1 km² area, holding geometry mix, speed
distribution, duration and conflict criteria constant so density is the only
factor that moves (research plan §22).
"""),
    code("""
DENSITIES = [10, 25, 50, 75, 100]
LATENCY_MS = [0, 250, 500, 1000, 2000]
RATE_HZ = 1.0
# Conflict events scale with density, so low-density scenarios need more
# repetitions to populate the AoI bins. Budget is roughly constant per density.
SEEDS_FOR = {10: 48, 25: 24, 50: 12, 75: 8, 100: 6}
DURATION_S = 180

# record="relevant" keeps only pairs either layer called a conflict and counts
# the rest; recall, precision and TTC error are unchanged by this (verified in
# tests/test_fast_runner.py) and 100-aircraft runs become tractable.
runs, dens_rows, tn_counts = [], [], []
for n in DENSITIES:
    for seed in range(SEEDS_FOR[n]):
        tcfg = TrafficConfig(n_aircraft=n, area_km2=1.0, duration_s=DURATION_S,
                             sample_rate_hz=5.0, speed_profile=profile, seed=500+seed)
        traj = generate_traffic(tcfg)

        # Interaction density is a property of the traffic, measured once.
        ld = local_density(traj, DURATION_S/2)
        dens_rows.append({
            "density_per_km2": n, "seed": seed,
            **{f"local_mean_{int(r)}m": float(ld[f"n_within_{int(r)}m"].mean())
               for r in LOCAL_RADII_M}})

        for lat_ms in LATENCY_MS:
            cfg = ScenarioConfig(
                scenario_id=f"D{n}", criteria=CRITERIA, decision_rate_hz=2.0, seed=7000+seed,
                telemetry=TelemetryConfig(update_rate_hz=RATE_HZ, latency_ms=lat_ms,
                                          jitter_ms=0.1*lat_ms))
            r = run_scenario_fast(traj, cfg, record="relevant")
            tn_counts.append({"density_per_km2": n, "seed": seed, "latency_ms": lat_ms,
                              "n_pair_observations": r.attrs.get("n_pair_observations", 0),
                              "n_true_negative": r.attrs.get("n_true_negative", 0)})
            if r.empty: continue
            r["density_per_km2"] = n
            r["seed_index"] = seed
            runs.append(r)

E4 = pd.concat(runs, ignore_index=True)
E4.to_parquet("results/raw/E4_density_sweep.parquet", index=False)
DENS = pd.DataFrame(dens_rows)
TN = pd.DataFrame(tn_counts)
TN.to_csv("results/processed/E4_pair_counts.csv", index=False)
print(f"{len(E4):,} conflict-relevant rows retained")
print(f"{TN.n_pair_observations.sum():,} pair-observations evaluated "
      f"({TN.n_true_negative.sum():,} true negatives counted, not stored)")
display(DENS.groupby("density_per_km2").mean().round(2))
"""),
    md("## Figure F2 — conflict recall against information age, per density"),
    code("""
from telemetry_freshness.evaluation.metrics import recall_by_aoi

curves = []
for n, g in E4.groupby("density_per_km2"):
    c = recall_by_aoi(g)
    c["density_per_km2"] = n
    c["density_label"] = f"{n}/km²"
    curves.append(c)
REC = pd.concat(curves, ignore_index=True)
REC = REC[REC.n_positive >= 30]
REC.to_csv("results/processed/E4_recall_by_aoi_density.csv", index=False)

from telemetry_freshness.evaluation.statistics import wilson_interval
lo_hi = [wilson_interval(int(r.tp), int(r.n_positive)) for _, r in REC.iterrows()]
REC["recall_lo"] = [x[0] for x in lo_hi]; REC["recall_hi"] = [x[1] for x in lo_hi]

fig, ax = F.fig_recall_vs_aoi(REC, level_col="density_label", requirement=0.99)
S.save(fig, "F2_recall_vs_aoi_by_density"); S.plt.show()
display(REC.pivot_table(index="aoi_bin", columns="density_label", values="recall").round(4))
"""),
    md("## E5 — which predictor actually governs the requirement?"),
    code("""
inter = []
for (n, seed), g in E4.groupby(["density_per_km2","seed_index"]):
    ir = interaction_rate(g, window_s=60)
    inter.append({"density_per_km2": n, "seed_index": seed,
                  "conflict_pairs_per_min": ir.conflict_pairs_per_min.mean(),
                  "converging_pairs_per_min": ir.converging_pairs_per_min.mean()})
INTER = pd.DataFrame(inter)
PRED = (INTER.merge(DENS, left_on=["density_per_km2","seed_index"],
                    right_on=["density_per_km2","seed"], how="left"))
display(PRED.groupby("density_per_km2").mean(numeric_only=True).round(2))
PRED.to_csv("results/processed/E5_predictors.csv", index=False)
"""),
]

nb06 = [
    md("""
# 06 · Experiment E6 — the safe Age-of-Information threshold

The paper's headline quantity (research plan §16):

$$\\mathrm{AoI}^{\\text{safe}}_{\\max} = \\max\\{\\,\\mathrm{AoI} : \\text{stated performance constraints hold}\\,\\}$$

**The constraints are inputs, not findings.** They are stated here, carried into
every result, and varied in a sensitivity analysis. Nothing in this notebook
should be read as a regulatory requirement.

A bin only counts as satisfying a constraint if it does so at the **lower
Wilson confidence bound** and has enough samples — otherwise a thinly-populated
bin with a lucky recall of 1.0 would set the threshold.
"""),
    code(HEADER),
    code("""
from telemetry_freshness.evaluation.threshold import (
    SafetyConstraints, derive_safe_aoi, safe_aoi_by_condition, evaluate_bins)

E1 = pd.read_parquet("results/raw/E1_pairwise_sweep.parquet")
E4 = pd.read_parquet("results/raw/E4_density_sweep.parquet")

CONSTRAINTS = SafetyConstraints(
    min_recall=0.99,
    min_lead_time_p95_s=3.0,
    max_ttc_error_p95_s=None,
    min_samples=200,
    use_confidence_bound=True,
    confidence=0.95,
)
display(pd.Series(CONSTRAINTS.describe()).to_frame("stated constraint"))
"""),
    md("## Threshold from the controlled pairwise encounters"),
    code("""
res = derive_safe_aoi(E1, CONSTRAINTS)
print(f"AoI_max^safe = {res.aoi_max_safe_s} s      limiting constraint: {res.limiting_constraint}")
display(res.bins[["aoi_bin","n","n_positive","recall","recall_lo","ok_recall","satisfies"]].round(4))
res.bins.to_csv("results/processed/E6_threshold_bins_E1.csv", index=False)
"""),
    md("## Threshold per encounter geometry"),
    code("""
by_geom = safe_aoi_by_condition(E1, ["encounter_type"], CONSTRAINTS)
display(by_geom)

closing = (E1.groupby("encounter_type").closing_speed_truth_mps
           .median().rename("median_closing_speed_mps"))
by_geom = by_geom.merge(closing, on="encounter_type")
by_geom.to_csv("results/processed/E6_safe_aoi_by_geometry.csv", index=False)
display(by_geom.round(2))
"""),
    md("""
## Threshold per traffic density — Figure F6

Deriving a per-density threshold first, then examining what actually moves with
density. The threshold is a step function of the bin edges, so it is reported
alongside the underlying recall curves rather than on its own.
"""),
    code("""
CONSTRAINTS_DENSITY = SafetyConstraints(
    min_recall=CONSTRAINTS.min_recall,
    min_lead_time_p95_s=CONSTRAINTS.min_lead_time_p95_s,
    min_samples=40,
    use_confidence_bound=True,
)

RECALL_LEVELS = [0.90, 0.95, 0.99]
rows = []
for req in RECALL_LEVELS:
    c = SafetyConstraints(min_recall=req,
                          min_lead_time_p95_s=CONSTRAINTS.min_lead_time_p95_s,
                          min_samples=40, use_confidence_bound=True)
    d = safe_aoi_by_condition(E4, ["density_per_km2"], c)
    d["min_recall"] = req
    rows.append(d)
BYDENS = pd.concat(rows, ignore_index=True)
BYDENS.to_csv("results/processed/E6_safe_aoi_by_density.csv", index=False)

display(BYDENS.pivot(index="density_per_km2", columns="min_recall", values="aoi_max_safe_s"))
print("NaN = no information age satisfies that recall requirement at that density.")
"""),
    md("""
### The per-density thresholds do not form a trend — and the reason matters

The table above is not monotonic in density. Before reading anything into that,
look at the quantity underneath it.
"""),
    code("""
from telemetry_freshness.evaluation.metrics import recall_by_aoi

curves = []
for n, g in E4.groupby("density_per_km2"):
    c = recall_by_aoi(g); c["density_per_km2"] = n
    curves.append(c)
CURVES = pd.concat(curves, ignore_index=True)
CURVES = CURVES[CURVES.n_positive >= 30]

piv = CURVES.pivot_table(index="aoi_bin", columns="density_per_km2", values="recall")
order = ["[0.0, 0.1)","[0.1, 0.25)","[0.25, 0.5)","[0.5, 0.75)","[0.75, 1.0)",
         "[1.0, 1.5)","[1.5, 2.0)","[2.0, 3.0)","[3.0, 5.0)"]
piv = piv.reindex([o for o in order if o in piv.index])
display(piv.round(4))

spread = (piv.max(axis=1) - piv.min(axis=1))
print(f"Across a 10x change in density, recall within an AoI bin varies by at most "
      f"{spread.max():.3f} (median {spread.median():.3f}).")
"""),
    md("""
> **Result (RQ2/RQ3).** Conflict recall is a function of information age and is
> **essentially independent of traffic density**: across a tenfold change in
> density, recall inside a given AoI bin moves by only a few points, with no
> consistent ordering. Density does not change how detectable an individual
> encounter is.
>
> The per-density thresholds in the previous table are therefore step-function
> noise around one common curve, not a density trend. Reporting them as a trend
> would be an error.

### What density does change

If per-encounter detectability is fixed, the density effect must be in the
*rate* of encounters. The operationally meaningful quantity is the number of
conflicts missed per minute:

$$\\text{missed conflicts/min} = \\lambda_c(\\rho) \\times \\bigl(1 - \\text{recall}(\\mathrm{AoI})\\bigr)$$

where the first term scales with density and the second does not.
"""),
    code("""
DECISION_HZ = 2.0
rowsm = []
for (n, lat), g in E4.groupby(["density_per_km2","latency_ms"]):
    span_s = g.t_s.max() - g.t_s.min()
    n_runs = g.seed_index.nunique()
    if span_s <= 0 or n_runs == 0: continue
    truth = g[g.conflict_truth]
    missed = truth[~truth.conflict_observed]
    # Convert conflict-observations into per-minute event rates.
    per_min = 60.0 / (span_s * n_runs * DECISION_HZ)
    rowsm.append({
        "density_per_km2": n, "latency_ms": lat,
        "aoi_p50_s": g.aoi_s.median(),
        "recall": 1 - len(missed)/max(len(truth), 1),
        "conflict_obs_per_min": len(truth) * per_min * DECISION_HZ,
        "missed_per_min": len(missed) * per_min * DECISION_HZ,
    })
MISS = pd.DataFrame(rowsm)
MISS.to_csv("results/processed/E6_missed_conflict_rate.csv", index=False)
display(MISS.pivot(index="density_per_km2", columns="latency_ms", values="missed_per_min").round(2))
print("Missed true conflicts per minute, by density (rows) and added latency (cols).")
"""),
    code("""
fig, ax = S.plt.subplots(figsize=(6.6, 4.2))
lats = sorted(MISS.latency_ms.unique())
colors = S.sequential_colors(len(lats))
for lat, c in zip(lats, colors):
    g = MISS[MISS.latency_ms == lat].sort_values("density_per_km2")
    ax.plot(g.density_per_km2, g.missed_per_min, color=c, marker="o", ms=6, lw=2,
            markeredgecolor=S.INK["surface"], markeredgewidth=1.3, zorder=3)
    S.direct_label(ax, g.density_per_km2.iloc[-1], g.missed_per_min.iloc[-1], f"{lat:g} ms", c)
ax.set_xlabel("Traffic density (UAS/km²)  —  synthetic extrapolation")
ax.set_ylabel("Missed true conflicts per minute")
ax.set_title("Density does not change detectability — it multiplies the consequence")
ax.legend(handles=[S.plt.Line2D([], [], color=c, lw=2, label=f"{lat:g} ms")
                   for lat, c in zip(lats, colors)],
          title="Added latency", loc="upper left")
S.annotate_note(ax, "1 Hz telemetry · per-encounter recall is density-invariant (table above)")
S.finish(fig, ax); S.save(fig, "F6_missed_conflicts_vs_density"); S.plt.show()
"""),
    code("""
# The density-invariant curve itself, pooled across densities.
pooled = recall_by_aoi(E4)
pooled = pooled[pooled.n_positive >= 30]
from telemetry_freshness.evaluation.statistics import wilson_interval
lh = [wilson_interval(int(r.tp), int(r.n_positive)) for _, r in pooled.iterrows()]
pooled["recall_lo"] = [x[0] for x in lh]; pooled["recall_hi"] = [x[1] for x in lh]
pooled["density_label"] = "all densities pooled"
pooled.to_csv("results/processed/E6_pooled_recall_curve.csv", index=False)

fig, ax = F.fig_recall_vs_aoi(pooled, level_col="density_label", requirement=0.99,
    title="Conflict recall against information age, pooled over 10–100 UAS/km²")
S.annotate_note(ax, "curves for individual densities are within a few points of this one",
                loc="lower left")
S.save(fig, "F6b_pooled_recall_curve"); S.plt.show()
"""),
    md("""
## Which predictor explains the threshold best? (RQ3)

Rank correlation, because the relationships are expected to be monotonic but
not linear.
"""),
    code("""
from telemetry_freshness.evaluation.statistics import spearman_predictors

try:
    PRED = pd.read_csv("results/processed/E5_predictors.csv").groupby("density_per_km2").mean(numeric_only=True).reset_index()
    BYDENS = BYDENS.merge(PRED, on="density_per_km2", how="left")
except FileNotFoundError:
    pass
cands = [c for c in ("density_per_km2","local_mean_100m","local_mean_250m","local_mean_500m",
                     "local_mean_1000m","conflict_pairs_per_min","converging_pairs_per_min")
         if c in BYDENS.columns]

# Correlate within each recall requirement: pooling across requirements would
# mix two different quantities on the y-axis.
for req, g in BYDENS.groupby("min_recall"):
    finite = g.dropna(subset=["aoi_max_safe_s"])
    print(f"--- recall requirement {req:g}  ({len(finite)} finite thresholds) ---")
    if len(finite) >= 3:
        display(spearman_predictors(finite, "aoi_max_safe_s", cands).round(4))
    else:
        print("   too few finite thresholds for a rank correlation; reported descriptively.")
        display(g[["density_per_km2","aoi_max_safe_s","limiting_constraint"]])
"""),
    md("## Sensitivity of the threshold to the stated constraints — Figure F7"),
    code("""
rows = []
for recall_req in (0.95, 0.98, 0.99, 0.995, 0.999):
    for lead_req in (0.0, 2.0, 3.0, 5.0):
        c = SafetyConstraints(min_recall=recall_req, min_lead_time_p95_s=lead_req,
                              min_samples=200, use_confidence_bound=True)
        r = derive_safe_aoi(E1, c)
        rows.append({"min_recall": recall_req, "min_lead_s": lead_req,
                     "aoi_max_safe_s": r.aoi_max_safe_s,
                     "limiting": r.limiting_constraint})
SENS = pd.DataFrame(rows)
SENS.to_csv("results/processed/E6_sensitivity.csv", index=False)

piv = SENS.pivot(index="min_recall", columns="min_lead_s", values="aoi_max_safe_s")
fig, ax = S.plt.subplots(figsize=(6.2, 4.0))
im = ax.imshow(piv.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto")
ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
ax.set_yticks(range(len(piv.index)), [f"{r:g}" for r in piv.index])
ax.set_xlabel("Minimum alert lead time (s)"); ax.set_ylabel("Minimum conflict recall")
ax.set_title("The derived threshold is a function of the constraints you state")
ax.grid(False)
vals = piv.to_numpy(dtype=float)
mid = np.nanmean(vals)
for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        v = vals[i, j]
        ax.text(j, i, "—" if not np.isfinite(v) else f"{v:g}", ha="center", va="center",
                fontsize=8, color=S.INK["surface"] if np.isfinite(v) and v > mid else S.INK["primary"])
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("AoI_max^safe (s)", fontsize=8)
cb.outline.set_visible(False)
S.finish(fig, ax); S.save(fig, "F7_threshold_sensitivity"); S.plt.show()
display(SENS)
"""),
    md("""
> **How to report this.** The threshold is meaningless without its constraints
> attached. Every figure and table above carries them, and F7 shows the whole
> surface rather than a single number, so a reader can locate the value that
> corresponds to *their* operational requirement.
"""),
]

nb07 = [
    md("""
# 07 · External validation — real surveillance observations (ICMCIS)

BUBBLES and IDF-DS both observe aircraft through **cooperative telemetry**. If
the freshness effects reported so far were an artefact of that one mechanism,
this notebook would be where it shows: ICMCIS observes the same kind of target
with **radar**, an entirely independent sensing chain.

| Layer | Source |
|---|---|
| ground truth | onboard drone log (10 Hz), geodetic position + speed + yaw |
| observed | ALVIRA and ARCUS radar tracks, with the sensor's own timestamps |

> **Claim discipline.** ICMCIS is a counter-UAS detection and tracking research
> dataset, not an operational UTM. It validates the truth-versus-observation
> mechanism; it does not describe UTM performance.
"""),
    code(HEADER),
    code("""
from telemetry_freshness.data.icmcis import IcmcisDataset
ds = IcmcisDataset()
SCENARIO = "train/Scenario_1_1"
gt = ds.load_scenario(SCENARIO)
dt = np.median(np.diff(gt[gt.uas_id == gt.uas_id.iloc[0]].t_s.to_numpy()))
print(f"scenarios available : {len(ds.scenarios())}")
print(f"ground truth        : {len(gt):,} rows, {gt.uas_id.nunique()} aircraft, "
      f"{gt.t_s.max():.0f} s, {1/dt:.1f} Hz")
"""),
    code("""
obs = {}
for sensor in ("ALVIRA", "ARCUS"):
    try:
        o = ds.load_observations(SCENARIO, sensor=sensor)
        obs[sensor] = o
        iv = o.track_interval_s.dropna(); iv = iv[(iv > 0) & (iv < 60)]
        print(f"{sensor:7s}: {len(o):6,} track rows | "
              f"update interval p50 {iv.median():.3f}s ({1/iv.median():.2f} Hz), p95 {iv.quantile(.95):.3f}s")
    except Exception as e:
        print(f"{sensor}: {type(e).__name__}: {e}")
"""),
    md("""
### A caveat on the age measurement

The sensor track timestamp has millisecond resolution, but the row's record
clock is integer seconds. The observation age recoverable from ICMCIS is
therefore quantised to ~1 s and is a coarse figure. The **track update
interval** keeps full resolution and is the more trustworthy measure of how
often this surveillance chain refreshed a track.
"""),
    code("""
fig, ax = S.plt.subplots(figsize=(6.6, 4.0))
for (sensor, o), c in zip(obs.items(), S.categorical_colors(max(len(obs), 2))):
    iv = o.track_interval_s.dropna(); iv = iv[(iv > 0) & (iv < 30)].sort_values()
    if iv.empty: continue
    ax.plot(iv.to_numpy(), np.linspace(0, 1, iv.size), color=c, lw=2)
    S.direct_label(ax, iv.iloc[-1], 1.0, sensor, c, dy=-8)
ax.set_xscale("log"); ax.set_xlabel("Track update interval (s, log scale)")
ax.set_ylabel("Cumulative probability")
ax.set_title("Real radar track refresh intervals")
S.annotate_note(ax, f"ICMCIS {SCENARIO}")
S.finish(fig, ax); S.save(fig, "S9_icmcis_track_intervals"); S.plt.show()
"""),
    md("## Truth versus observation in the plan view"),
    code("""
fig, ax = S.plt.subplots(figsize=(6.4, 5.6))
g = gt[gt.uas_id == gt.uas_id.iloc[0]]
ax.plot(g.x_m, g.y_m, color=S.INK["primary"], lw=2.0, label="onboard log (truth)", zorder=4)
for (sensor, o), c in zip(obs.items(), S.categorical_colors(max(len(obs), 2))):
    sub = o[(o.x_m.abs() < 3000) & (o.y_m.abs() < 3000)]
    ax.scatter(sub.x_m, sub.y_m, s=6, alpha=0.35, color=c, edgecolors="none", label=f"{sensor} track")
ax.set_aspect("equal", adjustable="datalim")
ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
ax.set_title("Onboard truth against independent radar observation")
ax.legend(loc="best")
S.finish(fig, ax); S.save(fig, "S10_icmcis_truth_vs_radar"); S.plt.show()
"""),
    md("""
## Applying the same freshness machinery to a real sensor

The radar track stream is fed through the identical Age-of-Information
measurement used for the emulated links, so the numbers are directly
comparable with notebooks 02 and 04.
"""),
    code("""
from telemetry_freshness.telemetry.message import TelemetryMessage
from telemetry_freshness.surveillance.state_store import SurveillanceStateStore
from telemetry_freshness.surveillance.age_of_information import freshness_errors
from telemetry_freshness.telemetry.generator import TrajectorySampler
from telemetry_freshness.core.schema import State

rows = []
sampler = TrajectorySampler(gt[gt.uas_id == gt.uas_id.iloc[0]])
for sensor, o in obs.items():
    sub = o.dropna(subset=["t_gen_s","x_m","y_m"]).sort_values("t_gen_s")
    sub = sub[sub.t_gen_s.between(sampler.t_start, sampler.t_end)]
    if len(sub) < 20:
        print(f"{sensor}: too few overlapping track rows"); continue
    store = SurveillanceStateStore()
    store.ingest([
        TelemetryMessage(
            uas_id=sampler.uas_id, t_gen_s=float(r.t_gen_s), t_arrive_s=float(r.t_gen_s),
            state=State(sampler.uas_id, float(r.t_gen_s), float(r.x_m), float(r.y_m),
                        float(r.z_m), float(r.vx_mps or 0), float(r.vy_mps or 0), 0.0))
        for r in sub.itertuples()
    ])
    t_eval = np.arange(sub.t_gen_s.min() + 2, sub.t_gen_s.max() - 2, 0.5)
    f = freshness_errors(store, sampler, t_eval)
    f = f[f.tracked].copy(); f["sensor"] = sensor
    rows.append(f)

if rows:
    E9 = pd.concat(rows, ignore_index=True)
    E9.to_parquet("results/raw/E9_icmcis_validation.parquet", index=False)
    display(E9.groupby("sensor").agg(
        n=("aoi_s","size"), aoi_p50=("aoi_s","median"), aoi_p95=("aoi_s", lambda s: s.quantile(.95)),
        err_p50=("horiz_error_m","median"), err_p95=("horiz_error_m", lambda s: s.quantile(.95))
    ).round(2))
else:
    print("No overlapping truth/track window in this scenario.")
"""),
    md("""
> The absolute error here mixes information age with the radar's own
> **measurement** error, which the cooperative-telemetry experiments do not
> have. The two cannot be separated from this data, so ICMCIS is used to
> confirm the *direction and mechanism* — observation age degrades the observed
> state — and not to produce a threshold.
"""),
]

nb08 = [
    md("""
# 08 · Paper figures

Regenerates every manuscript figure from the stored result tables. **No
simulation runs here** — if a figure changes, it is because the analysis
changed, not because a random seed moved.

Outputs land in `results/figures/` as vector PDF plus 300-dpi PNG.
"""),
    code(HEADER),
    code("""
from pathlib import Path
RAW = Path("results/raw")
loaded = {}
for key, fn in [("E1","E1_pairwise_sweep.parquet"), ("E2","E2_idfds_downsampling.parquet"),
                ("E3","E3_maneuver_sensitivity.parquet"), ("E4","E4_density_sweep.parquet")]:
    p = RAW / fn
    if p.exists():
        loaded[key] = pd.read_parquet(p)
        print(f"{key}: {len(loaded[key]):,} rows")
    else:
        print(f"{key}: MISSING ({fn}) — run the upstream notebook first")
"""),
    code("""
built = []

if "E2" in loaded:
    fig, _ = F.fig_aoi_vs_position_error(
        loaded["E2"], subtitle="IDF-DS · 12 real fixed-wing flights · 62 Hz reference")
    built += S.save(fig, "F1_aoi_vs_position_error"); S.plt.close(fig)

if "E1" in loaded:
    fig, _ = F.fig_ttc_error_vs_aoi(loaded["E1"], level_col="encounter_type")
    built += S.save(fig, "F3_ttc_error_vs_aoi"); S.plt.close(fig)

    fig, _ = F.fig_rate_latency_heatmap(loaded["E1"], value_col="ttc_error_abs_s", agg="p95")
    built += S.save(fig, "F5_rate_latency_heatmap"); S.plt.close(fig)

    fig, _ = F.fig_rate_vs_realised_aoi(loaded["E1"])
    built += S.save(fig, "F8_rate_vs_realised_aoi"); S.plt.close(fig)

for name, fn, kwargs in [
    ("F2_recall_vs_aoi_by_density", "results/processed/E4_recall_by_aoi_density.csv",
     dict(level_col="density_label", requirement=0.99)),
]:
    p = Path(fn)
    if p.exists():
        d = pd.read_csv(p)
        from telemetry_freshness.evaluation.statistics import wilson_interval
        lh = [wilson_interval(int(r.tp), int(r.n_positive)) for _, r in d.iterrows()]
        d["recall_lo"] = [x[0] for x in lh]; d["recall_hi"] = [x[1] for x in lh]
        fig, _ = F.fig_recall_vs_aoi(d, **kwargs)
        built += S.save(fig, name); S.plt.close(fig)

p = Path("results/processed/E1_lead_time_by_aoi.csv")
if p.exists():
    fig, _ = F.fig_lead_time_vs_aoi(pd.read_csv(p), level_col="encounter_type", min_lead_s=3.0)
    built += S.save(fig, "F4_lead_time_vs_aoi"); S.plt.close(fig)

print(f"{len(built)} files written")
for b in built: print("  ", b)
"""),
    md("## Manifest"),
    code("""
figs = sorted(Path("results/figures").glob("*.pdf"))
manifest = pd.DataFrame({
    "figure": [f.stem for f in figs],
    "pdf_kb": [round(f.stat().st_size/1024, 1) for f in figs],
    "png_exists": [f.with_suffix(".png").exists() for f in figs],
})
display(manifest)
manifest.to_csv("results/figures/manifest.csv", index=False)
"""),
]

if __name__ == "__main__":
    for name, cells in (
        ("05_density_scaling", nb05),
        ("06_safe_aoi_threshold", nb06),
        ("07_icmcis_validation", nb07),
        ("08_paper_figures", nb08),
    ):
        print("wrote", build(name, cells))
