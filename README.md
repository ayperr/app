# Citi Bike NYC — Rebalancing Dashboard

A Streamlit app for a Citi Bike operations/dispatch audience: predicts each
station's net bike flow a few hours ahead (separately for classic and
electric bikes), flags stations at risk of overfilling or running empty,
and recommends a minimum-cost set of truck moves to fix it -- including
proactive moves from stations that are merely comfortably stocked, not
just ones that are literally overfull (see **Modeling summary** below).

Two modes, switchable from the sidebar at any time:

- **📼 Historical Playback** — replays a real 5-week window exactly as the
  models saw it at training/eval time. Fully reproducible, no network
  needed, always available.
- **🔴 Live** — the same two trained models, scoring real-time GBFS station
  status, live weather, and live NYC event data, auto-refreshing every 2
  minutes (or on demand via the in-app **Refresh now** button). See
  **Live mode** below before demoing this one.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. First load takes ~30-40 seconds in Historical
mode (loading both models and simulating two months of station occupancy);
it's cached after that. Live mode's first load is faster (no occupancy
simulation to run) but still needs a few seconds to load both models and
poll three live data sources.

## Live mode

Live mode trains offline (same pipeline, same models — nothing about the
models themselves changes) and serves online: each refresh cycle pulls
real-time data from three public sources and reshapes it into the exact
feature layout the models were trained on. In the app, the **Live data**
tab breaks down exactly which features are real-time right now vs. a
documented stand-in, with live coverage numbers; `live_data.py`'s module
docstring has the full mechanics (reference-hour convention, the hybrid
lag-feature strategy, and a real bug found and fixed along the way).

**Needs outbound internet access to:**
- `gbfs.citibikenyc.com` — live station status (bike/dock counts)
- `api.open-meteo.com` — live weather
- `data.cityofnewyork.us` — live NYC permitted-events feed

If a deploy host blocks any of these, Live mode degrades gracefully (a
clear in-app message, auto-retrying every cycle) rather than crashing —
Historical Playback keeps working regardless, since it's fully self-
contained. No API keys are needed for any of the three.

**Known gap:** the live GBFS feed checked during development covers what
looks like all five NYC boroughs but not Hoboken/Jersey City — roughly 200
stations that are real to the system (and in the historical training data)
just won't appear in Live mode. Historical Playback still covers them.

**Self-accumulating history:** Live mode logs one snapshot per station per
hour to `data/live_history/snapshot_log.csv` (gitignored — this is runtime
state, not something to commit). That log is what lets the flow-lag
features graduate from a historical seasonal proxy to real self-observed
values over the app's first running week; delete the file to reset it.

## Deploy to Streamlit Community Cloud

1. Push this `app/` folder to a GitHub repo (everything it needs — code,
   models, demo data — is self-contained inside it; nothing here needs
   Git LFS, the two model files are ~40MB each).
2. On [share.streamlit.io](https://share.streamlit.io), point a new app at
   the repo with `app.py` as the entry point.
3. No secrets or API keys are required to run either mode — the map's
   basemap tiles come from Carto's free tier, Historical Playback runs
   entirely on the bundled demo data, and Live mode's three data sources
   (above) are all public/keyless. If the host has outbound internet
   access, both modes work out of the box.

## Project structure

```
app.py    main UI: sidebar controls, shared KPIs/map/tabs, mode switch
data_utils.py         Historical Playback data layer + occupancy simulation
live_data.py            Live mode: snapshot logging, hybrid lag features, live snapshot assembly
gbfs.py                   live GBFS/weather/events fetch layer used by live_data.py
inference.py           loads the two XGBoost models, predicts net flow
optimizer.py          min-cost-flow rebalancing solver
data/                  station list, ~2 months of demo feature data, station ID crosswalk,
                       seasonal baseline lookup (Live mode's fallback), live history log (runtime)
models/               trained XGBoost models + metadata (feature order, categories, metrics)
```

## Modeling summary

Two independent XGBoost regressors (classic bikes, electric bikes) predict
net flow (arrivals − departures) 3 hours ahead per station, trained on
~3 years of NYC Citi Bike trip history plus historical weather and permitted-
events data. Validated with **5-fold time series cross-validation**
(expanding-window, chronological — never random k-fold, which would leak
near-identical neighboring-in-time rows into training) on the ~2 years
before a final held-out test window, then scored once against that
untouched, chronologically-later ~6-month test set (see the app's
**Model performance** tab for both the per-fold CV numbers and the test
score).

The rebalancing optimizer is **proactive**, not just reactive: a station
doesn't need to be predicted literally over capacity to act as a bike
source for a truck move. Any station predicted to stay comfortably
stocked — above its own empty-risk threshold plus a safety margin it
keeps in reserve — even after giving bikes away is a valid donor for a
nearby station predicted to run low. Previously the optimizer could only
relieve literal overfills, so a system with plenty of empty risk but zero
overfull docks had nothing to recommend at all; the **Rebalancing moves**
tab now labels each move as either relieving an overfill or proactively
freeing up a comfortable station.

Full methodology, every simplifying assumption made, and one real
data-quality bug found and fixed along the way are documented in the
app's **About this demo** tab and in the module docstrings
(`data_utils.py` and `optimizer.py` especially) — worth reading before
presenting this, not just decoration.

## Regenerating the models

The scripts that built everything under `data/` and `models/` — ETL
(`01_densify_and_join_duckdb.py`), leak-safe feature engineering
(`02_features.py`), model training (`03_train_xgboost.py`), and the
seasonal-baseline lookup Live mode falls back on (`04_seasonal_baseline.py`)
— ship alongside this `app/` folder in the sibling `pipeline/` folder. You
don't need to re-run them to use this app; they're here for reference/
writeup purposes and to reproduce the pipeline from scratch if the
underlying trip data is refreshed. See `pipeline/README.md` for what each
stage needs.
