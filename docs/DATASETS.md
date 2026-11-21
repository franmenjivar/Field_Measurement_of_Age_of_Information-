# Datasets — obtaining and placing them

Four third-party datasets are used. None are redistributed in this repository:
each has its own licence and provider. Obtain them yourself, place them at the
paths below, then run `notebooks/00_environment_and_data_check.ipynb` — it
exercises every adapter on one scenario and tells you which datasets are missing.

Total on-disk size once all four are in place is roughly **3.8 GB**.

Every adapter takes a `root=` argument, so if you keep the data elsewhere you can
pass your own path instead of using the defaults below.

---

## Target layout

```text
data/
├── raw/                               # immutable; never written to
│   ├── bubbles/DroneFlightDataset/
│   │   ├── 2022_04_11/
│   │   │   ├── input/telemetry/<drone_id>/logNNN.txt
│   │   │   └── utm_tracks/logNNN.txt
│   │   └── 2022_04_12/
│   ├── icmcis/drone_detection/
│   │   ├── train/<scenario>/
│   │   └── test/<scenario>/
│   ├── idf_ds/Speedybee_DataSet.zip
│   └── uagree/17737383/
│       ├── ois.json
│       ├── times.csv
│       ├── table_cells.csv
│       └── real_grids_21x28x194_n5.npy
├── interim/
│   └── idf_ds/Speedybee DataSet/       # the unpacked archive
│       ├── processed/GPS/LapNNN_gps.csv
│       └── processed/telemetry/LapNNN.csv
└── processed/                          # committed in this repo
    ├── bubbles_latency_pool_s.npy
    └── bubbles_latency_pool.provenance.json
```

Default roots, as coded in the adapters:

| Dataset | Adapter | Default root |
|---|---|---|
| BUBBLES | `data/bubbles.py` | `data/raw/bubbles/DroneFlightDataset` |
| IDF-DS | `data/idf_ds.py` | `data/interim/idf_ds/Speedybee DataSet` |
| ICMCIS | `data/icmcis.py` | `data/raw/icmcis/drone_detection` |
| U-AGREE | `data/uagree.py` | `data/raw/uagree/17737383` |

---

## 1. BUBBLES — primary

A real U-space validation campaign flown near Valencia, Spain (≈39.60 N,
−0.28 E) on 2022-04-11 and 2022-04-12, with **14 drones airborne
simultaneously**: `ADS-Phanthom`, `ASD-Madv-4`, `BV-Madv-{1,2}`, `PLB-Madv-3`,
`PLB-Mzoom-3`, `PLV-Mdual-{1,2}`, `PLV-Mzoom-{1,2}`, `UAW-Valaq`, `UPV-Matrice`,
`UPV-Mavic-{1,2}`. ≈217 MB.

**How to obtain it.** The campaign data was produced by the BUBBLES SESAR project
with the Air Navigation Systems research group at Universitat Politècnica de
València. It is not on a public mirror this repository can link to directly;
request it from the BUBBLES / SNA-UPV project authors. Announcement thread:
<https://www.linkedin.com/posts/sna-upv_dataset-de-trayectorias-uas-para-la-validaci%C3%B3n-activity-7426565739358236672-p1qb>

**Verifying your copy.** [`bubbles_manifest.sha256`](bubbles_manifest.sha256)
lists the SHA-256 of every file used in the published runs (8,739 files: the
telemetry and UTM track logs of both days plus `Dataset Ejercicio5_Fase2.docx`;
paths relative to `DroneFlightDataset/`). The SHA-256 of the manifest itself is
`d63f1d5bf4a7d84d81b255bd8a9387f973b3a0bf9c79ad8be2853f7f789d67a7`.

```bash
cd data/raw/bubbles/DroneFlightDataset
shasum -a 256 -c --quiet ../../../../docs/bubbles_manifest.sha256   # sha256sum -c on Linux
```

No output means every file matches; any mismatch or missing file is listed.

This is the only dataset that carries **measured** latency and Age of
Information, so notebooks 01, 09, 10, 11 and 12 cannot run without it.

### Why it matters here

The archive has two layers that map exactly onto this study's two-state model.

**Ground truth / generation layer** — `input/telemetry/<drone_id>/logNNN.txt`,
JSONL, one telemetry message per line:

```json
{"time_ms": 1649675595106,
 "time": "2022-04-11_13:13:15",
 "message": {"position": {"latitude": 0, "longitude": 0, "altitude": 0},
             "attitude": {"pitch": 0, "yaw": 0, "roll": 0},
             "speed":    {"GS": 0, "track_angle": 0, "vert_speed": 0},
             "battery": 0, "pilot_alerts": [],
             "time": 1649675594587}}
```

`time_ms − message.time` is a **directly measured one-way telemetry latency**.

**Observed layer** — `utm_tracks/logNNN.txt`, JSONL, one UTM surveillance cycle
per line, with `t` (cycle time) and `t_last` (last update for that aircraft).
**`t − t_last` is Age of Information, measured in a real operational U-space
system** — not modelled.

### Gotchas the adapter handles (all verified)

- `t` is offset **+7199.3 s (≈ +2 h)** from `time_ms`: the UTM writes local CEST
  wall-clock as if it were epoch. The two layers must be normalised before joining.
- `message.time` units are **not uniform across drones** — epoch-ms for
  `ADS-Phanthom`, `ASD-Madv-4`, `BV-Madv-{1,2}`, `PLB-Madv-3`, `PLB-Mzoom-3`,
  `PLV-Mzoom-{1,2}` and `UAW-Valaq` (latency directly measurable), but a
  monotonic **uptime counter** for `PLV-Mdual-{1,2}`, `UPV-Matrice` and
  `UPV-Mavic-1` (mixed for `UPV-Mavic-2`), where only relative jitter is
  recoverable after per-file offset estimation. The two groups are reported
  separately.
- ~3 % of `utm_tracks` lines carry `message` as a dict rather than a list; these
  are skipped and counted.
- Some raw latencies are slightly negative (min −208 ms) from sender/receiver
  clock skew. Skew is estimated and reported, never silently clipped.

---

## 2. IDF-DS (SpeedyBee) — primary validation

Real autonomous fixed-wing flights over a 3 981 m circuit. ≈554 MB.

- Paper: <https://www.nature.com/articles/s41597-026-06716-3>
- Data: <https://doi.org/10.5281/zenodo.16992975>

Download the SpeedyBee F405 / INAV portion, keep the archive at
`data/raw/idf_ds/Speedybee_DataSet.zip`, and unpack it to
`data/interim/idf_ds/`. The archive used for the published runs has MD5
`fa6210078d8f615118027a5310dfc99c`.

```bash
mkdir -p data/interim/idf_ds
unzip data/raw/idf_ds/Speedybee_DataSet.zip -d data/interim/idf_ds/
md5 data/raw/idf_ds/Speedybee_DataSet.zip        # md5sum on Linux
```

What the adapter reads:

- `processed/GPS/LapNNN_gps.csv` — 112 laps at exactly 5.00 Hz, ~460 s each.
  `GPS_coord[0]` = lat, `[1]` = lon (decimal degrees), `GPS_altitude` (m),
  `GPS_speed` (m/s), `GPS_velned[0..2]` = [N, E, D] in **cm/s**,
  `GPS_ground_course` (deg), `GPS_hdop`.
- `processed/telemetry/LapNNN.csv` — 110 laps, 126 columns, **~62 Hz**:
  `navPos[0..2]` (EKF position, **cm**, local NEU from home), `navVel[0..2]`
  (**cm/s**), `attitude[0..2]` (**decidegrees**, yaw 0–3596), `BaroAlt` (cm),
  `navState`.

**The 62 Hz `navPos`/`navVel` stream is the reference trajectory, not the 5 Hz
GPS.** That rate is what makes the 100–200 ms AoI regime measurable on real data.

---

## 3. ICMCIS — external validation

Counter-UAS detection challenge data: 7 train + 7 test scenarios, drone onboard
logs alongside independent radar and RF sensor tracks. ≈366 MB.

- <https://www.kaggle.com/c/icmcis-drone-tracking/data>

A Kaggle account and acceptance of the competition rules are required. Unpack so
that scenario directories sit under `data/raw/icmcis/drone_detection/{train,test}/`.

Per scenario:

- `YYYY-MM-DD_HH-MM-SS_v2.csv` — drone onboard log = **ground truth**
  (lat, lon, `altitude(m)`, `speed(mps)`, `velocityX/Y/Z(mps)`,
  `pitch/roll/yaw(deg)`, `time(millisecond)`, `datetime(utc)`, epoch
  `timestamp`; ~7 980 rows). Some scenarios have two drone logs.
- `ALVIRA_scenario.csv`, `ARCUS_scenario.csv` — radar tracks = **observed**.
- `DIANA_scenario.csv` — RF detection (bearing, range, SNR, classification).
- `VENUS_scenario.csv` — RF direction finding.

Its role is narrow: truth-vs-sensor observation-age validation against an
*independent* surveillance technology (radar rather than cooperative telemetry).

---

## 4. U-AGREE — operation-footprint geometry only

Strategic 4-D airspace occupancy from PX4-based U-space scenarios. ≈105 MB.

- <https://zenodo.org/records/17737383>

Contents: `ois.json` (operational intents), `times.csv` (28 117 rows,
`;`-separated: `idtime; timeBegin; timeEnd; idcell; oi_id; ordinal`),
`table_cells.csv` (114 072 cells with `xlim/ylim/zlim` in metres),
`real_grids_21x28x194_n5.npy` (`(100, 194, 21, 28)` int64 occupancy grids,
~6.8 % occupied).

**Verified limitation — read before planning any experiment on it.** This is
**not telemetry**; do not treat it as trajectories. Its 8 operational intents
are strictly sequential: each spans only ~19–21 s, none overlap in time
(**max concurrent operations = 1**), and no two intents claim the same cell
(**zero cell contention**). U-AGREE therefore **cannot** supply concurrent
traffic, conflict density or congestion scenarios. Its usable role is limited to
realistic single-operation 4-D footprint geometry (cells ≈1 392 m³, corridor
x 4 734–5 714 m, y −454…527 m, z −103…48 m) and operation-duration statistics.

---

## Datasets deliberately not used

- **CREATE U-space** — superseded by BUBBLES, which is real rather than
  simulated and carries native AoI fields.
- **AUVSI USRD** — licensed and unavailable. Platform-class speed distributions
  are instead derived empirically from the 14 real BUBBLES airframes and from
  IDF-DS, and are labelled as such.

Neither may be cited as a source for any result in this repository.
