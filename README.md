# Field Measurement of Age of Information

**Age-of-Information requirements for tactical conflict detection in high-density UAS traffic management.**

This repository contains everything needed to reproduce the experiments: the
`telemetry_freshness` Python package, the thirteen analysis notebooks that run
the experiment catalogue, the test suite that enforces the study's scientific
controls, and the stored experiment outputs and figures.

The four source datasets are third-party and are **not** redistributed here.
[`docs/DATASETS.md`](docs/DATASETS.md) says where to obtain each one and exactly
where to place it on disk.

---

## 1. What is being measured

```
ground truth ──► telemetry generator ──► network model ──► surveillance store
                  (rate)                  (latency,          (latest-value-wins)
                                           jitter, loss)            │
                                                     decision time ─┤
                                                                    ▼
                                       AoI = t − generation time of freshest state
                                                                    │
                              ┌─────────────────────────────────────┴────┐
                        detector(truth)                          detector(observed)
                              └──────────────► evaluator ◄─────────────────┘
```

Age of Information (AoI) is the elapsed time between the moment an aircraft
state was *generated* on board and the moment a conflict detector *uses* it.
Two detectors run over the same encounter: one on ground truth, one on what the
surveillance system actually held. The difference between them is the cost of
staleness.

**The central control:** the detector is constructed with a state store and holds
no reference to any trajectory, so it cannot read ground truth even accidentally.
`tests/test_isolation.py` fails if that separation is ever broken.

---

## 2. Quick start

Python 3.11 or newer is required (3.13 was used for the published runs).

```bash
git clone https://github.com/franmenjivar/Field_Measurement_of_Age_of_Information-.git
cd Field_Measurement_of_Age_of_Information-

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[logs,notebooks,dev]"

.venv/bin/python -m pytest tests/ -q      # 24 tests, no datasets needed
```

All 24 tests must pass before you go further. They exercise the mechanism
analytically (closed-form AoI, seeded-RNG reproducibility) and assert the
truth/observation isolation; none of them touch the raw datasets.

Then obtain the datasets ([`docs/DATASETS.md`](docs/DATASETS.md)) and launch
Jupyter:

```bash
.venv/bin/python -m jupyter lab
```

Run the notebooks **in numerical order**, starting with
`00_environment_and_data_check.ipynb`. Every later notebook assumes 00 passes.

---

## 3. Repository layout

```text
src/telemetry_freshness/   installable package — all algorithm logic lives here
  core/          ENU geodesy, seeded RNG, unified trajectory schema, provenance
  data/          one adapter per dataset (bubbles, idf_ds, icmcis, uagree)
  telemetry/     message model, generator (rate), network model (latency/jitter/loss)
  surveillance/  latest-value-wins state store, AoI accounting, tracker
  simulation/    synthetic traffic generator, encounter geometry construction
  conflict/      CPA solver and the conflict detector
  evaluation/    metrics, timing, phases, density, thresholds, statistics, inference
  viz/           paper figure builders and the single shared style
  experiments/   runners (full + fast), real-stream replay

notebooks/                 00–12; orchestration and figures only, no algorithm logic
scripts/                   generators that build the notebooks (see §6)
tests/                     analytic, mechanism and isolation tests
data/raw/                  immutable third-party datasets — you provide these
data/interim/              unpacked archives — you provide these
data/processed/            small derived artefacts, committed here
results/raw/               experiment outputs (Parquet), committed here
results/processed/         aggregated tables (CSV), committed here
results/figures/           paper figures (PDF + 300-dpi PNG), committed here
```

`data/raw/` is never written to by any code path.

---

## 4. Reproducing the experiments

Each notebook runs one or more experiments from the catalogue, writes Parquet to
`results/raw/`, aggregated CSV to `results/processed/`, and figures to
`results/figures/`.

| # | Notebook | Experiment | Source data | Produces |
|---|---|---|---|---|
| 00 | `00_environment_and_data_check` | toolchain + dataset parse check | all four | — |
| 01 | `01_bubbles_measured_freshness` | measured freshness in a real U-space system | BUBBLES | F0, S2–S4 |
| 02 | `02_idfds_downsampling` | **E2** telemetry downsampling on real flights | IDF-DS | **F1**, S5, S6 |
| 03 | `03_maneuver_sensitivity` | **E3** manoeuvre sensitivity | IDF-DS | **F1b**, S7 |
| 04 | `04_pairwise_encounters` | **E1** pairwise sweep (rate × latency × geometry) | IDF-DS motion | **F3**, **F4**, **F5**, **F8**, S8 |
| 05 | `05_density_scaling` | **E4/E5** density scaling and interaction density | simulated traffic | **F2** |
| 06 | `06_safe_aoi_threshold` | **E6** safe-AoI threshold and sensitivity | E1 + E4 + E5 | **F6**, F6b, **F7** |
| 07 | `07_icmcis_validation` | **E9** truth-vs-radar external validation | ICMCIS | S9, S10 |
| 08 | `08_paper_figures` | regenerate every figure from stored results | `results/` | all |
| 09 | `09_load_vs_freshness` | **E10** does traffic load degrade freshness? | BUBBLES | F9 |
| 10 | `10_real_stream_replay` | **E11** replay of a real surveillance stream | BUBBLES | F10 |
| 11 | `11_extrapolation_ablation` | **E12** does dead reckoning remove the effect? | BUBBLES + sim | F11 |
| 12 | `12_empirical_latency` | **E13** measured link vs modelled link | BUBBLES pool | F12 |

### Experiment catalogue

| ID | Name | Independent variables |
|---|---|---|
| E1 | Pairwise rate × latency sweep | rate {0.5, 1, 2, 5, 10} Hz × latency {0, 100, 250, 500, 1000} ms × geometry {head-on, crossing, overtaking, converging} × 12 Monte-Carlo seeds (1000–1011) |
| E2 | IDF-DS downsampling | rate {ref, 10, 5, 2, 1, 0.5} Hz × added delay {0, 50, 100, 250, 500, 1000} ms |
| E3 | Manoeuvre sensitivity | flight phase {cruise, turn, climb, descent, high-accel} × AoI |
| E4 | Density scaling | ρ {10, 25, 50, 75, 100} UAS/km² × AoI bins |
| E5 | Local interaction density | `N_i(r)`, r ∈ {100, 250, 500, 1000} m; conflict rate λ_c |
| E6 | Safe AoI threshold | constraints: recall ≥ 0.99, P95 lead time ≥ `T_min` |
| E7 | Update rate vs realised AoI | derived from E1 |
| E9 | ICMCIS truth-vs-sensor | — |
| E10 | Surveillance load vs freshness | tracks held per UTM cycle |
| E11 | Real-stream replay | real UTM tracks vs onboard truth |
| E12 | Extrapolation ablation | {hold-last, dead-reckon, reported-velocity, truth-velocity} arms |
| E13 | Measured vs modelled link | empirical latency pool vs matched constant latency |

Every sweep varies exactly one factor family; geometry, speeds, thresholds and
duration are held fixed and asserted equal across the sweep by the runner.

### Regenerating only the figures

If you want the figures without re-running the sweeps, the committed
`results/raw/*.parquet` and `results/processed/*.csv` are sufficient — run
notebook `08_paper_figures` alone. It reads only from `results/` and needs no
raw dataset.

### Reproducibility contract

- Every stochastic component draws from a seeded `numpy.random.Generator`; a run
  is reproducible from its config alone, and `tests/test_mechanism.py` asserts it.
- Processed artefacts carry a `*.provenance.json` sidecar recording source file,
  transformation, code version, library versions and platform.
- `data/raw/` is read-only to the pipeline.

---

## 5. Results these experiments produce

- **Real measured AoI** in an operational U-space system: median 0.60 s,
  P95 3.75 s, **P99 18.3 s** — strongly heavy-tailed.
- **Error is governed by age, not update rate**: 25 rate × latency configurations
  collapse onto one curve (< 2 % spread above 0.75 s AoI). A 1 Hz link with no
  delay beats a 10 Hz link with 1 s delay.
- **Manoeuvres break heading, not position**: at matched age, effective lag is
  0.96–0.97 s in *every* flight phase, while heading error is 5.9× worse in a turn.
- **Safe AoI = 0.50 s** at recall ≥ 0.99 for isolated encounters; overtaking is
  the most demanding geometry, at 0.25 s.
- **Density does not change detectability — it multiplies consequence**:
  per-encounter recall is density-invariant across a 10× density change, while
  missed conflicts per minute scale as density^1.87.

### Interpreting the results honestly

The following claims are **not** supported by these data and should not be made:

- IDF-DS is a high-density UTM dataset — it is **single real flights**.
- ICMCIS is an operational UTM — it is a **detection/tracking research set**.
- U-AGREE supplies concurrent traffic — its operational intents are **strictly
  sequential** (max concurrent operations = 1, zero cell contention).
- Any AoI value is "universally safe" — every threshold here is conditional on
  the stated recall and lead-time constraints.

Encounters constructed from real IDF-DS segments are **experimentally
constructed geometry with real motion characteristics**, not observed encounters.

---

## 6. Notebooks are generated, not hand-edited

The notebooks under `notebooks/` are built from the generators in `scripts/`:

```bash
.venv/bin/python scripts/make_nb_00_02.py
.venv/bin/python scripts/make_nb_03_04.py
.venv/bin/python scripts/make_nb_05_08.py
.venv/bin/python scripts/make_nb_09.py
.venv/bin/python scripts/make_nb_10.py
.venv/bin/python scripts/make_nb_11.py
.venv/bin/python scripts/make_nb_12.py
```

This keeps them in sync with the package after a refactor. If you edit a
notebook by hand, your change is lost the next time a generator runs — put
algorithm changes in `src/telemetry_freshness/` instead.

`scripts/make_ieee_figures.py` rebuilds the publication figure set from
`results/`.

---

## 7. Troubleshooting

**`DatasetNotAvailable`** — an adapter could not find its dataset root. This is
deliberately loud: an experiment must never silently fall back to synthetic data.
Check the path against [`docs/DATASETS.md`](docs/DATASETS.md).

**Notebook imports fail** — every notebook resolves the repository root itself
and `chdir`s to it, so run Jupyter from anywhere inside the repository, and make
sure you installed the package with `pip install -e .` into the same interpreter
Jupyter is using.

**`pyproj` or `pyarrow` wheel errors** — both ship binary wheels for CPython
3.11–3.13 on macOS and Linux; on an unsupported interpreter pip will try to build
from source. Use a supported Python version rather than building.

---

## 8. License

Code in this repository: MIT (see [`LICENSE`](LICENSE)).

The four source datasets are **not** covered by that license. Each carries its
own terms, and you must obtain them from their original providers and comply
with those terms. See [`docs/DATASETS.md`](docs/DATASETS.md).
