"""
Live snapshot assembly -- turns right-now GBFS/weather/events into the same
shaped feature row the trained models saw during training, using a hybrid
source per feature (see the table in streamlit_app.py's Live tab and the
About section for the full breakdown). This is the module that makes
"train offline on history, score online on live data" actually work.

Reference-hour convention: a training row at hour H holds flow that
happened *during* [H, H+1) and predicts [H+3, H+4). Live serving mirrors
that exactly by using the most recently *completed* hour as "the row's
hour" (e.g. at 14:37 wall-clock, that's 13:00) -- the in-progress hour
[14:00, 14:37) isn't finished yet, so it can't be a real net_flow value.
Weather/events are fetched for right-now and used as a stand-in for
conditions during that just-completed hour -- fine, since both change
slowly relative to an hour (same "weather-as-of-time" simplification
documented for historical mode).

Real bug caught while building this: DuckDB's EXTRACT(dow) -- what every
day_of_week value in the trained models is in -- numbers Sunday=0..
Saturday=6. pandas' .dayofweek numbers Monday=0..Sunday=6. They disagree on
every single day. Feeding a live pandas weekday straight into the model
would have silently mislabeled every day of the week (e.g. a live
Wednesday encoded as pandas' "2" when the model was trained to read "2" as
Tuesday) -- exactly the kind of bug that doesn't crash, just quietly
degrades one of the model's most important features. _duckdb_dow() below
is the one place that conversion happens; nothing else in this file should
call pandas .dayofweek directly.

A second subtlety, caught by checking rather than assuming: station_id is
itself a trained categorical feature (see 03_train_xgboost.py), and for the
~126 phantom-pair stations (data_utils.py's docstring), the "canonical" id
station_id_map.csv designates is picked by nearest-lat/lng grouping, with no
regard for which raw id variant the model actually saw much history under.
Checked empirically: for 82 of those 126 stations, the "canonical" variant
has *less* training history than the other raw variant, and for a handful
it has exactly zero (e.g. every historical row for that physical dock was
recorded under the *other* raw id) -- meaning the model never learned any
real split behavior for that exact category value. Naively summing both raw
variants' predictions (what data_utils.get_snapshot() does for playback,
where each variant genuinely has its own distinct historical lag/flow
values) does NOT transfer here: live mode only has one combined signal per
physical dock, so feeding that same combined signal to both raw ids and
summing would double-count it and roughly 2x the prediction for these
stations. Instead, _best_raw_station_id() below picks whichever raw variant
the model actually has the *most* training history for, per bike type (the
better variant can differ between classic and electric -- e.g. if the two
id variants happen to fall on different sides of the e-bike rollout), and
uses only that one id string as the station_id feature -- one prediction,
no double-counting, but grounded in a category the trees actually learned
something from rather than an arbitrary pick.
"""
import os
import time

import numpy as np
import pandas as pd
import streamlit as st

import gbfs
from data_utils import HORIZON_HOURS, load_station_id_map, load_stations

DATA = "data"
LIVE_HISTORY_DIR = f"{DATA}/live_history"
SNAPSHOT_LOG_PATH = f"{LIVE_HISTORY_DIR}/snapshot_log.csv"
NY_TZ = "America/New_York"

LAG_HOURS_NEEDED = {"net_flow_lag_1h": 2, "net_flow_lag_24h": 25, "net_flow_lag_168h": 169,
                     "net_flow_rollavg_24h": 25, "net_flow_avg_last4_sameslot": 169}


def _duckdb_dow(pandas_dayofweek):
    """pandas .dayofweek (Mon=0..Sun=6) -> DuckDB EXTRACT(dow) (Sun=0..Sat=6),
    the convention every trained model's day_of_week feature is in. See
    module docstring -- do not use pandas .dayofweek directly anywhere else
    in this file."""
    return (pandas_dayofweek + 1) % 7


@st.cache_data(show_spinner=False)
def _load_holidays():
    """
    Set of US-holiday dates (pandas.Timestamp, midnight) used to compute
    is_holiday live, the same list 02_features.py joined against at training
    time (copied to data/holidays.csv unchanged). Covers through end of 2026
    -- like any fixed holiday calendar, this needs a refreshed file once that
    runs out; until then, dates past its coverage just fall back to
    is_holiday=0, same as any non-holiday date.
    """
    df = pd.read_csv(f"{DATA}/holidays.csv")
    return set(pd.to_datetime(df["date"]))


def get_reference_hour(now=None):
    """The most recently *completed* hour, tz-naive, matching the training
    data's timestamp convention (local NYC time, no tz attached)."""
    if now is None:
        now = pd.Timestamp.now(tz=NY_TZ)
    elif now.tzinfo is None:
        now = now.tz_localize(NY_TZ)
    return now.tz_localize(None).floor("h") - pd.Timedelta(hours=1)


# ------------------------------------------------------------- seasonal baseline
@st.cache_data(show_spinner="Loading seasonal baseline table...")
def load_seasonal_baseline(bike_type):
    """
    Canonicalized (station_id, hour, day_of_week) -> typical value lookup.
    Phantom-pair stations (see data_utils.py) are merged with an
    n_obs-weighted average, not a plain mean, so a station that only had
    history under one of its two raw ids isn't diluted by an all-zero
    "observation" from the other.
    """
    df = pd.read_parquet(f"{DATA}/seasonal_baseline_{bike_type}.parquet")
    df["station_id"] = df["station_id"].astype(str)
    id_map = load_station_id_map()
    df["station_id"] = df["station_id"].map(id_map).fillna(df["station_id"])

    value_cols = [c for c in df.columns if c.startswith("avg_")]
    for c in value_cols:
        df[c] = df[c] * df["n_obs"]  # un-average -> weighted sum, so groupby-sum is a correct weighted merge
    agg = df.groupby(["station_id", "hour", "day_of_week"], as_index=False)[value_cols + ["n_obs"]].sum()
    for c in value_cols:
        agg[c] = agg[c] / agg["n_obs"].replace(0, np.nan)
    return agg


def _lookup_seasonal(seasonal_df, station_ids, hour, day_of_week):
    key = seasonal_df[(seasonal_df["hour"] == hour) & (seasonal_df["day_of_week"] == day_of_week)]
    return key.set_index("station_id").reindex(station_ids)


@st.cache_data(show_spinner=False)
def _best_raw_station_id(bike_type):
    """
    canonical_station_id -> whichever RAW station_id (see module docstring)
    the trained model has the most historical rows for, for this bike type.
    Stations with only one raw variant just map to themselves.

    Reads the raw (pre-canonicalization) seasonal baseline parquet -- same
    file load_seasonal_baseline() reads, but *before* its canonicalizing
    merge collapses phantom pairs together, since it's exactly that
    collapsed-away distinction (which raw id has real history) this needs.
    """
    raw = pd.read_parquet(f"{DATA}/seasonal_baseline_{bike_type}.parquet", columns=["station_id", "n_obs"])
    raw["station_id"] = raw["station_id"].astype(str)
    obs_by_raw = raw.groupby("station_id")["n_obs"].sum()

    id_map = load_station_id_map()  # raw -> canonical
    ranked = pd.DataFrame({"raw_id": obs_by_raw.index, "n_obs": obs_by_raw.to_numpy()})
    ranked["canonical_id"] = ranked["raw_id"].map(id_map).fillna(ranked["raw_id"])
    best = ranked.sort_values("n_obs", ascending=False).drop_duplicates("canonical_id", keep="first")
    return best.set_index("canonical_id")["raw_id"]


# ------------------------------------------------------------- snapshot log
def _ensure_log_dir():
    os.makedirs(LIVE_HISTORY_DIR, exist_ok=True)


def log_snapshot_if_new_hour(reference_hour, current_bikes_df):
    """
    current_bikes_df: [station_id, bikes_classic, bikes_electric,
    bikes_total] for whatever stations were live-matched this poll.

    Only appends once per reference_hour (checked against the log's own
    last entry, not wall-clock, so this is safe to call every 120s from an
    auto-refreshing fragment) -- one row per station per hour is exactly
    the granularity net_flow needs; logging every poll would grow the file
    ~60x faster for zero benefit.
    """
    _ensure_log_dir()
    already_logged = False
    if os.path.exists(SNAPSHOT_LOG_PATH):
        try:
            last = pd.read_csv(SNAPSHOT_LOG_PATH, usecols=["logged_at"]).tail(1)
            if not last.empty and pd.Timestamp(last["logged_at"].iloc[0]) >= reference_hour:
                already_logged = True
        except Exception:
            pass  # corrupt/empty log -- just proceed to (re)write
    if already_logged:
        return False

    out = current_bikes_df.copy()
    out.insert(0, "logged_at", reference_hour.isoformat())
    write_header = not os.path.exists(SNAPSHOT_LOG_PATH)
    out.to_csv(SNAPSHOT_LOG_PATH, mode="a", header=write_header, index=False)
    return True


def load_snapshot_log():
    if not os.path.exists(SNAPSHOT_LOG_PATH):
        return pd.DataFrame(columns=["logged_at", "station_id", "bikes_classic", "bikes_electric", "bikes_total"])
    df = pd.read_csv(SNAPSHOT_LOG_PATH)
    df["logged_at"] = pd.to_datetime(df["logged_at"])
    df["station_id"] = df["station_id"].astype(str)
    return df


def history_depth_hours(log_df):
    """Hours between the oldest and newest logged snapshot -- drives the
    'X/168h of live history collected' progress the UI shows."""
    if log_df.empty:
        return 0.0
    span = log_df["logged_at"].max() - log_df["logged_at"].min()
    return span.total_seconds() / 3600.0


# ------------------------------------------------------------- hybrid lag features
def build_live_lag_features(log_df, reference_hour, count_col):
    """
    log_df: output of load_snapshot_log(). count_col: 'bikes_classic',
    'bikes_electric', or 'bikes_total'.

    Returns a DataFrame indexed by station_id with net_flow,
    net_flow_lag_1h/24h/168h, net_flow_rollavg_24h,
    net_flow_avg_last4_sameslot -- NaN wherever the required snapshot(s)
    don't exist yet (caller fills those from the seasonal baseline).
    net_flow itself needs one snapshot pair too (reference_hour and
    reference_hour-1h), same as any other hourly flow value.
    """
    if log_df.empty:
        return pd.DataFrame(columns=["net_flow", "net_flow_lag_1h", "net_flow_lag_24h",
                                      "net_flow_lag_168h", "net_flow_rollavg_24h",
                                      "net_flow_avg_last4_sameslot"])

    df = log_df.sort_values(["station_id", "logged_at"]).copy()
    df["net_flow"] = df.groupby("station_id")[count_col].diff()
    wide = df.pivot(index="station_id", columns="logged_at", values="net_flow")

    def col_at(ts):
        return wide[ts] if ts in wide.columns else pd.Series(np.nan, index=wide.index)

    out = pd.DataFrame(index=wide.index)
    out["net_flow"] = col_at(reference_hour)
    out["net_flow_lag_1h"] = col_at(reference_hour - pd.Timedelta(hours=1))
    out["net_flow_lag_24h"] = col_at(reference_hour - pd.Timedelta(hours=24))
    out["net_flow_lag_168h"] = col_at(reference_hour - pd.Timedelta(hours=168))

    roll_cols = [c for c in wide.columns if reference_hour - pd.Timedelta(hours=24) <= c < reference_hour]
    out["net_flow_rollavg_24h"] = wide[roll_cols].mean(axis=1) if roll_cols else np.nan

    sameslot_cols = [reference_hour - pd.Timedelta(hours=168 * k) for k in range(1, 5)]
    sameslot_cols = [c for c in sameslot_cols if c in wide.columns]
    out["net_flow_avg_last4_sameslot"] = wide[sameslot_cols].mean(axis=1) if sameslot_cols else np.nan
    return out


# ------------------------------------------------------------- live current state
def fetch_live_current_state():
    """
    Orchestrates the live GBFS calls. Returns (state_df, meta):
    - state_df: [station_id (canonical), bikes_classic, bikes_electric,
      bikes_total, live_capacity, last_reported] for matched stations only.
    - meta: {"ok": bool, "reason": str|None, "crosswalk_stats": {...}}
    """
    stations = load_stations()
    live_info = gbfs.fetch_live_station_information()
    if live_info is None:
        return None, {"ok": False, "reason": "Could not reach the live GBFS feed."}

    crosswalk, cw_stats = gbfs.build_crosswalk(stations, live_info)
    if crosswalk is None or crosswalk.empty:
        return None, {"ok": False, "reason": "GBFS reachable but no stations matched our station list."}

    status = gbfs.fetch_live_station_status()
    if status is None:
        return None, {"ok": False, "reason": "Could not reach the live GBFS station-status feed.",
                       "crosswalk_stats": cw_stats}

    merged = crosswalk.merge(status, on="live_station_id", how="inner")
    merged = merged.merge(live_info[["live_station_id", "capacity"]], on="live_station_id", how="left")
    merged = merged.rename(columns={
        "num_classic_available": "bikes_classic",
        "num_ebikes_available": "bikes_electric",
        "num_bikes_available": "bikes_total",
        "capacity": "live_capacity",
    })
    # a station can match multiple live rows only if two feeds in
    # STATION_STATUS_URLS both cover it (not expected today, but be safe)
    merged = merged.groupby("station_id", as_index=False).agg(
        bikes_classic=("bikes_classic", "sum"), bikes_electric=("bikes_electric", "sum"),
        bikes_total=("bikes_total", "sum"), live_capacity=("live_capacity", "first"),
        last_reported=("last_reported", "max"),
    )
    return merged, {"ok": True, "reason": None, "crosswalk_stats": cw_stats}


# ------------------------------------------------------------- full assembly
def assemble_live_snapshot():
    """
    Top-level entry point for Live mode -- mirrors data_utils.get_snapshot()'s
    output shape (station_id, name, lat, lng, borough, capacity,
    current_bikes_total, pred_net_flow_classic, pred_net_flow_electric,
    pred_net_flow_total) plus a meta dict for the UI (freshness, coverage,
    how much of each feature came from real live history vs the seasonal
    proxy). Returns (None, meta) if live data couldn't be fetched at all.
    """
    from inference import predict_net_flow  # local import: avoid a cycle at module load

    state_df, meta = fetch_live_current_state()
    if state_df is None:
        return None, meta

    reference_hour = get_reference_hour()
    log_snapshot_if_new_hour(reference_hour, state_df[["station_id", "bikes_classic", "bikes_electric",
                                                         "bikes_total"]])
    log_df = load_snapshot_log()
    hist_hours = history_depth_hours(log_df)

    dow = _duckdb_dow(reference_hour.dayofweek)
    hour = reference_hour.hour
    month = reference_hour.month
    is_weekend = 1 if dow in (0, 6) else 0
    is_holiday = 1 if reference_hour.normalize() in _load_holidays() else 0

    weather = gbfs.fetch_current_weather()
    now_iso = pd.Timestamp.now(tz=NY_TZ).tz_localize(None).isoformat(timespec="seconds")
    events = gbfs.fetch_live_events(now_iso)

    stations = load_stations().set_index("station_id")
    all_ids = stations.index
    matched_ids = state_df["station_id"]  # coverage below is scoped to these, not all_ids -- see note there

    preds = {}
    coverage = {}
    for bike_type in ["classic_bike", "electric_bike"]:
        count_col = "bikes_classic" if bike_type == "classic_bike" else "bikes_electric"
        seasonal = load_seasonal_baseline(bike_type)
        seasonal_row = _lookup_seasonal(seasonal, all_ids, hour, dow)
        live_lags = build_live_lag_features(log_df, reference_hour, count_col).reindex(all_ids)

        feat = pd.DataFrame(index=all_ids)
        feat["hour"] = hour
        feat["day_of_week"] = dow
        feat["month"] = month
        feat["is_weekend"] = is_weekend
        feat["is_holiday"] = is_holiday
        feat["lat"] = stations["lat"]
        feat["lng"] = stations["lng"]
        feat["borough"] = stations["borough"]

        for wcol, wdefault in [("temperature_2m", 15.0), ("relative_humidity_2m", 60.0), ("precipitation", 0.0),
                                ("rain", 0.0), ("snowfall", 0.0), ("wind_speed_10m", 5.0), ("cloud_cover", 50.0),
                                ("weather_code", 0.0)]:
            feat[wcol] = (weather or {}).get(wcol, wdefault)

        ev = events if events is not None else pd.DataFrame(columns=["borough", "event_count", "major_event_count"])
        ev_lookup = ev.set_index("borough") if not ev.empty else ev
        feat["event_count"] = feat["borough"].map(ev_lookup["event_count"]) if not ev.empty else 0
        feat["major_event_count"] = feat["borough"].map(ev_lookup["major_event_count"]) if not ev.empty else 0
        feat["event_count"] = feat["event_count"].fillna(0)
        feat["major_event_count"] = feat["major_event_count"].fillna(0)

        # hybrid: real self-logged value where available, else the seasonal proxy
        lag_cols = ["net_flow", "net_flow_lag_1h", "net_flow_lag_24h", "net_flow_lag_168h",
                    "net_flow_rollavg_24h", "net_flow_avg_last4_sameslot"]
        seasonal_rename = {
            "net_flow": "avg_net_flow", "net_flow_lag_1h": "avg_net_flow_lag_1h",
            "net_flow_lag_24h": "avg_net_flow_lag_24h", "net_flow_lag_168h": "avg_net_flow_lag_168h",
            "net_flow_rollavg_24h": "avg_net_flow_rollavg_24h",
            "net_flow_avg_last4_sameslot": "avg_net_flow_avg_last4_sameslot",
        }
        is_real = {}
        for c in lag_cols:
            real_vals = live_lags[c] if c in live_lags.columns else pd.Series(np.nan, index=all_ids)
            proxy_vals = seasonal_row[seasonal_rename[c]] if seasonal_rename[c] in seasonal_row.columns \
                else pd.Series(0.0, index=all_ids)
            is_real[c] = real_vals.notna()
            feat[c] = real_vals.fillna(proxy_vals).fillna(0.0)
        # scoped to matched_ids, not all ~2,546 canonical stations: this is meant to answer
        # "of the stations we have live data for, how many have real history for this lag"
        # -- averaging over every canonical station (most of which have no live match at all,
        # e.g. every Hoboken/JC dock) would dilute this toward ~0 regardless of how much real
        # history the app has actually accumulated, and double-counts the separate
        # crosswalk-coverage question that n_matched/crosswalk_stats already answers
        coverage[bike_type] = {c: float(is_real[c].reindex(matched_ids).mean()) for c in lag_cols}

        feat["departures"] = seasonal_row["avg_departures"].reindex(all_ids).fillna(0.0)
        feat["arrivals"] = seasonal_row["avg_arrivals"].reindex(all_ids).fillna(0.0)

        feat = feat.reset_index().rename(columns={"index": "station_id"})
        # predict using the best-represented raw id (see module docstring),
        # then reattach canonical ids positionally -- feat's row order is
        # still exactly all_ids' order at this point, so this is safe and
        # keeps predict_net_flow's own (raw-id) index out of the picture
        canonical_order = feat["station_id"].to_numpy()
        feat["station_id"] = feat["station_id"].map(_best_raw_station_id(bike_type)).fillna(feat["station_id"])
        raw_pred = predict_net_flow(bike_type, feat)
        preds[bike_type] = pd.Series(raw_pred.to_numpy(), index=canonical_order)

    out = stations.copy()
    out["current_bikes_total"] = state_df.set_index("station_id")["bikes_total"].reindex(all_ids)
    out["capacity"] = state_df.set_index("station_id")["live_capacity"].reindex(all_ids).fillna(out["capacity"])
    out["pred_net_flow_classic"] = preds["classic_bike"].reindex(all_ids).fillna(0.0)
    out["pred_net_flow_electric"] = preds["electric_bike"].reindex(all_ids).fillna(0.0)
    out["pred_net_flow_total"] = out["pred_net_flow_classic"] + out["pred_net_flow_electric"]
    out = out.dropna(subset=["current_bikes_total"])  # unmatched-to-live stations excluded, not guessed at
    out = out.reset_index()

    meta.update({
        "reference_hour": reference_hour,
        "as_of": pd.Timestamp.now(tz=NY_TZ),
        "history_depth_hours": hist_hours,
        "coverage": coverage,
        "weather_ok": weather is not None,
        "events_ok": events is not None,
        "n_matched": len(out),
    })
    return out, meta
