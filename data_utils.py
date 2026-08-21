"""
Historical "playback" data layer -- one of the app's two modes (see
live_data.py for the other: real live GBFS/weather/events, scored through
the same trained models). Playback replays a real 5-week historical window
exactly as the models saw it at training/eval time; useful as a
fully-reproducible, network-free demo/fallback even once Live mode exists.

Why playback exists at all instead of only live: GBFS only ever exposes
*current* station state, never history, so on a fresh deploy there's no
live source yet for the lag/rolling-average features the models need
(net_flow 1h/24h/168h ago, a same-weekday-hour rolling average, ...).
live_data.py fills that gap with a historical-seasonal-average proxy that
gets replaced by real accumulated live history over the following days/
week -- see its module docstring. Playback mode sidesteps the question
entirely by replaying hours where the real values are already known.

Station *occupancy* was never in the training data either -- Citi Bike
doesn't publish historical GBFS snapshots, only current state, and the
trip-history archive only records flow (arrivals/departures), never
absolute dock counts. It's simulated here: every station is seeded at half
its (estimated) capacity at the start of the demo window, then real
historical net_flow is cumulatively summed hour by hour, clipped to
[0, capacity]. That makes the first few simulated days less trustworthy
than later ones -- the UI only lets you scrub the window's back 60 days,
excluding a 7-day warm-up run at the front.

Station capacity itself is an estimate too (see prep notes in
data/stations.csv) -- live GBFS station_information.json is the correct
source, and Live mode (live_data.py) overwrites it with real dock counts
for every station it can reach.

One more real data-quality issue surfaced during testing and is corrected
here rather than swept under the rug: Citi Bike's own trip-history exports
represent some stations' station_id inconsistently across months (e.g.
"6098.1" in some files, "6098.10" in others, for the exact same dock at
the same lat/lng) -- 123 stations (~4.6% of the system) are affected. Left
alone, the two ID variants are trained/predicted as if they were different
stations, and the optimizer would happily recommend moving bikes from a
station to itself. `station_id_map.csv` (built by grouping stations with
identical lat/lng) collapses each such pair to one canonical id at the
snapshot layer, below. The trained models themselves still saw fragmented,
gappier history for these ~123 stations during training, since fixing that
requires re-running the ETL with canonicalized IDs from the start -- so
predictions for merged stations are the sum of both variants' forecasts, a
mitigation rather than a full fix, and are probably somewhat noisier than
the system average.
"""
import numpy as np
import pandas as pd
import streamlit as st

from inference import predict_net_flow

DATA = "data"
BIKE_TYPES = ["classic_bike", "electric_bike"]
WARMUP_DAYS = 7
HORIZON_HOURS = 3


@st.cache_data(show_spinner="Loading historical demo data...")
def load_demo_features(bike_type):
    df = pd.read_parquet(f"{DATA}/demo_features_{bike_type}.parquet")
    df["station_id"] = df["station_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    # This table is ~2.4M rows x 29 columns and stays cached (st.cache_data)
    # for the life of the process -- loaded at float64/int64 precision from
    # parquet it runs ~475MB in memory *per bike type*, ~950MB for both,
    # which alone is enough to OOM-kill the process on a constrained host
    # (e.g. Streamlit Community Cloud's free ~1GB tier) -- seen in practice
    # as a bare "Oh no."/502-503 with no Python traceback, since the process
    # dies before it can log one. None of these columns need 64-bit
    # precision (weather/flow figures, not financial data), so downcasting
    # to the smallest safe dtype roughly halves the footprint for free.
    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    int_cols = df.select_dtypes(include="int64").columns
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    df["borough"] = df["borough"].astype("category")
    return df


@st.cache_data(show_spinner="Loading station list...")
def load_stations():
    df = pd.read_csv(f"{DATA}/stations.csv")
    df["station_id"] = df["station_id"].astype(str)
    return df


@st.cache_data(show_spinner="Loading station ID crosswalk...")
def load_station_id_map():
    """raw station_id (as it appears in the feature tables) -> canonical
    station_id (as it appears in stations.csv). See module docstring."""
    df = pd.read_csv(f"{DATA}/station_id_map.csv", dtype=str)
    return df.set_index("station_id")["canonical_station_id"]


def _canonicalize(df, id_col="station_id"):
    """Replaces df[id_col] (raw) with its canonical id in place; ids with
    no map entry (shouldn't happen, but be defensive) pass through as-is."""
    id_map = load_station_id_map()
    df = df.copy()
    df[id_col] = df[id_col].map(id_map).fillna(df[id_col])
    return df


@st.cache_data(show_spinner="Computing available time range...")
def available_timestamps():
    feats = load_demo_features("classic_bike")
    ts = pd.to_datetime(feats["date"]) + pd.to_timedelta(feats["hour"], unit="h")
    ts = pd.Series(ts.unique()).sort_values().reset_index(drop=True)
    cutoff = ts.min() + pd.Timedelta(days=WARMUP_DAYS)
    return ts[ts >= cutoff].reset_index(drop=True)


@st.cache_data(show_spinner="Simulating station occupancy from historical flow...")
def simulate_occupancy_total():
    """
    Combined (classic + electric) simulated bike count per station, for
    every hour in the demo window (including the warm-up days -- callers
    should only expose timestamps from available_timestamps() in the UI).
    Returns a wide DataFrame: index=station_id, columns=pd.Timestamp, values
    = simulated bike count (float).
    """
    parts = []
    for bt in BIKE_TYPES:
        f = load_demo_features(bt)[["station_id", "date", "hour", "net_flow"]]
        parts.append(_canonicalize(f))
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.groupby(["station_id", "date", "hour"], as_index=False)["net_flow"].sum()
    combined["ts"] = combined["date"] + pd.to_timedelta(combined["hour"], unit="h")

    stations = load_stations()[["station_id", "capacity"]]
    wide = combined.pivot(index="station_id", columns="ts", values="net_flow")
    wide = wide.reindex(stations["station_id"]).fillna(0.0)
    wide = wide.reindex(sorted(wide.columns), axis=1)

    cap = stations.set_index("station_id")["capacity"].reindex(wide.index).fillna(25).to_numpy(dtype="float32")
    flow = wide.to_numpy(dtype="float32")
    occ = np.empty_like(flow)
    cur = cap / 2.0
    for t in range(flow.shape[1]):
        cur = np.clip(cur + flow[:, t], 0, cap)
        occ[:, t] = cur
    return pd.DataFrame(occ, index=wide.index, columns=wide.columns)


def get_snapshot(selected_ts):
    """
    selected_ts: pandas.Timestamp at hour resolution (must come from
    available_timestamps()).

    Returns a per-station DataFrame: station_id, name, lat, lng, borough,
    capacity, current_bikes_total (simulated occupancy as of this hour),
    pred_net_flow_classic, pred_net_flow_electric, pred_net_flow_total
    (all HORIZON_HOURS ahead of selected_ts).
    """
    stations = load_stations()
    occ = simulate_occupancy_total()
    current_bikes = occ[selected_ts] if selected_ts in occ.columns else pd.Series(0.0, index=occ.index)

    preds = {}
    for bt in BIKE_TYPES:
        feats = load_demo_features(bt)
        snap = feats[(feats["date"] == selected_ts.normalize()) & (feats["hour"] == selected_ts.hour)]
        raw_pred = predict_net_flow(bt, snap)
        # raw_pred is indexed by the *raw* station_id from the feature table;
        # roll up to canonical ids (sums the ~123 split-history stations'
        # two forecasts into one) before it ever meets stations.csv
        canon_id = raw_pred.index.to_series().map(load_station_id_map()).fillna(raw_pred.index.to_series())
        preds[bt] = raw_pred.groupby(canon_id.to_numpy()).sum()

    out = stations.set_index("station_id").copy()
    out["current_bikes_total"] = current_bikes.reindex(out.index).fillna(out["capacity"] / 2.0)
    out["pred_net_flow_classic"] = preds["classic_bike"].reindex(out.index).fillna(0.0)
    out["pred_net_flow_electric"] = preds["electric_bike"].reindex(out.index).fillna(0.0)
    out["pred_net_flow_total"] = out["pred_net_flow_classic"] + out["pred_net_flow_electric"]
    return out.reset_index()
