"""
Live-data layer for Live mode. Everything here degrades gracefully: any
network failure (offline, feed down, timeout) returns None / an empty
result rather than raising, so callers always have a defined fallback
path. This was written against the *real* live schemas (confirmed via a
manual fetch during development, Aug 2026) but exercised end-to-end only
via mocked responses in this sandbox, which has no outbound access to
these hosts -- see live_data.py's tests.

Station ID crosswalk: live GBFS station_information.json uses a different,
opaque station_id scheme (e.g. "2124037125711300644") than the short-form
IDs (e.g. "7482.15", "HB607") used throughout the historical trip-data
archive this project's models were trained on -- Lyft migrated ID formats
at some point after the historical exports were generated, and there's no
published crosswalk. build_crosswalk() matches the two by nearest lat/lng
instead, within a tight tolerance (a false match would silently attach the
wrong dock to a station).

Known coverage gap: the live feed checked during development
(gbfs.citibikenyc.com) returned only stations under Lyft's "bkn" system
code, which appears to be all five NYC boroughs but NOT Hoboken/Jersey
City -- those ~200 stations in the training data (a real part of the
system Citi Bike operates) may have no live counterpart and will simply
show as unmatched. If Hoboken/JC turns out to publish its own separate
GBFS feed, add its station_information/station_status URLs to
STATION_INFO_URLS / STATION_STATUS_URLS below and build_crosswalk() will
pick up matches from either automatically.
"""
import numpy as np
import pandas as pd
import requests

STATION_INFO_URLS = ["https://gbfs.citibikenyc.com/gbfs/en/station_information.json"]
STATION_STATUS_URLS = ["https://gbfs.citibikenyc.com/gbfs/en/station_status.json"]
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
EVENTS_URL = "https://data.cityofnewyork.us/resource/tvpp-9vvx.json"
CENTRAL_PARK_LAT, CENTRAL_PARK_LNG = 40.7829, -73.9654
MATCH_TOLERANCE_KM = 0.05  # ~50m -- must be essentially the same physical dock to accept a match

# same borough-moving-event definition used at training time (01_densify_and_join_duckdb.py)
MAJOR_EVENT_TYPES = {
    "Special Event", "Street Event", "Block Party", "Production Event",
    "Parade", "Plaza Event", "Plaza Partner Event", "Street Festival",
    "Single Block Festival", "Open Street Partner Event", "Religious Event",
}


def _haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = np.radians(lat1), np.radians(lng1), np.radians(lat2), np.radians(lng2)
    dlat, dlng = lat2 - lat1, lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def fetch_live_station_information(timeout=6):
    """Returns a DataFrame [live_station_id, lat, lng, capacity], concatenated
    across every URL in STATION_INFO_URLS, or None if every feed failed."""
    frames = []
    for url in STATION_INFO_URLS:
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            stations = resp.json()["data"]["stations"]
            df = pd.DataFrame(stations)[["station_id", "lat", "lon", "capacity"]].rename(
                columns={"station_id": "live_station_id", "lon": "lng"}
            )
            frames.append(df.dropna(subset=["lat", "lng"]))
        except Exception:
            continue
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def fetch_live_station_status(timeout=6):
    """
    Returns a DataFrame [live_station_id, num_bikes_available,
    num_ebikes_available, num_classic_available, num_docks_available,
    last_reported (unix ts)], filtered to is_installed & is_renting
    stations only, concatenated across STATION_STATUS_URLS. None if every
    feed failed.

    num_bikes_available is the GBFS total (classic + electric);
    num_classic_available is derived as the remainder after subtracting
    num_ebikes_available, clipped at 0 defensively.
    """
    frames = []
    for url in STATION_STATUS_URLS:
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            stations = resp.json()["data"]["stations"]
            df = pd.DataFrame(stations)
            df = df[(df.get("is_installed", 1) == 1) & (df.get("is_renting", 1) == 1)]
            df = df.rename(columns={"station_id": "live_station_id"})
            df["num_ebikes_available"] = df.get("num_ebikes_available", 0).fillna(0)
            df["num_classic_available"] = (df["num_bikes_available"] - df["num_ebikes_available"]).clip(lower=0)
            cols = ["live_station_id", "num_bikes_available", "num_ebikes_available",
                    "num_classic_available", "num_docks_available", "last_reported"]
            frames.append(df[[c for c in cols if c in df.columns]])
        except Exception:
            continue
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def build_crosswalk(canonical_stations_df, live_info_df=None):
    """
    Nearest-neighbor lat/lng match (within MATCH_TOLERANCE_KM) between our
    canonical station list (station_id, lat, lng) and a live GBFS
    station_information fetch. Returns (crosswalk_df, stats):

    - crosswalk_df: [station_id (canonical), live_station_id, distance_km]
      -- one row per canonical station that found a close-enough live match.
    - stats: {"n_canonical", "n_live", "n_matched"} for surfacing coverage
      in the UI (e.g. "2,340 / 2,546 stations have live data").

    Returns (None, None) if the live feed couldn't be reached at all.
    """
    if live_info_df is None:
        live_info_df = fetch_live_station_information()
    if live_info_df is None or live_info_df.empty:
        return None, None

    live_lat = live_info_df["lat"].to_numpy()
    live_lng = live_info_df["lng"].to_numpy()
    live_ids = live_info_df["live_station_id"].to_numpy()

    rows = []
    for sid, lat, lng in zip(canonical_stations_df["station_id"], canonical_stations_df["lat"],
                              canonical_stations_df["lng"]):
        dists = _haversine_km(lat, lng, live_lat, live_lng)
        j = np.argmin(dists)
        if dists[j] <= MATCH_TOLERANCE_KM:
            rows.append({"station_id": sid, "live_station_id": live_ids[j], "distance_km": dists[j]})

    crosswalk_df = pd.DataFrame(rows)
    stats = {
        "n_canonical": int(len(canonical_stations_df)),
        "n_live": int(len(live_info_df)),
        "n_matched": int(len(crosswalk_df)),
    }
    return crosswalk_df, stats


def fetch_current_weather(lat=CENTRAL_PARK_LAT, lng=CENTRAL_PARK_LNG, timeout=6):
    """
    Returns a dict of current weather fields matching the training feature
    names, or None on failure. Uses Open-Meteo's live forecast endpoint --
    a different endpoint from the historical archive API used to build the
    training data, but the same provider/units.
    """
    try:
        params = {
            "latitude": lat, "longitude": lng,
            "current": "temperature_2m,relative_humidity_2m,precipitation,rain,snowfall,"
                       "wind_speed_10m,cloud_cover,weather_code",
            "timezone": "America/New_York",
        }
        resp = requests.get(WEATHER_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("current")
    except Exception:
        return None


def fetch_live_events(now_iso, timeout=8):
    """
    now_iso: current local time as 'YYYY-MM-DDTHH:MM:SS' (no timezone --
    matches the Socrata floating-timestamp format the historical ETL used).

    Returns a DataFrame [borough, event_count, major_event_count] for
    events active right now (start_date_time <= now <= end_date_time),
    grouped by borough -- same shape/semantics as the historical
    event_agg table joined at training time, just for "right now" instead
    of a specific past date. None on failure (caller should fall back to
    zero events rather than block on it).
    """
    try:
        params = {
            "$where": f"start_date_time <= '{now_iso}' AND end_date_time >= '{now_iso}'",
            "$limit": 5000,
            "$select": "event_borough,event_type",
        }
        resp = requests.get(EVENTS_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json())
        if df.empty:
            return pd.DataFrame(columns=["borough", "event_count", "major_event_count"])
        df["is_major"] = df["event_type"].isin(MAJOR_EVENT_TYPES)
        agg = (
            df.groupby("event_borough")
            .agg(event_count=("is_major", "size"), major_event_count=("is_major", "sum"))
            .reset_index()
            .rename(columns={"event_borough": "borough"})
        )
        return agg
    except Exception:
        return None
