"""
Rebalancing optimizer.

Given predicted bike counts per station at the forecast horizon, finds the
minimum-cost set of truck moves (source station -> destination station,
bike count) that relieves predicted overfills (docks full, arriving riders
can't return a bike) and predicted empties (no bikes available to rent) --
proactively, not just reactively: a station doesn't have to be literally
over capacity to act as a source. Any station predicted to stay
comfortably stocked -- above its own min-buffer plus a safety margin it
keeps for itself -- even after giving bikes away is fair game as a donor
for a nearby station that's predicted to run empty. See
`compute_surplus_deficit()`'s docstring for exactly how "comfortably
stocked" is defined and why hard-overfill stations and merely-comfortable
stations can share one source pool.

This is a transportation-problem / min-cost-flow formulation -- it answers
"how many bikes should move from each source station to each deficit
station to fix the system for the least total distance", not literal
turn-by-turn truck routing. Routing+scheduling several trucks over many
stops each is a vehicle-routing problem (VRP), which is a materially
harder combinatorial problem and out of scope for this project; a driver
using this app takes the ranked list of moves and works down it.

Travel cost between stations is a haversine-distance-based time estimate --
a proxy for real road-network travel time. A production deployment would
swap `travel_time_minutes()` for a real routing engine/API (OSRM, Mapbox
Directions, etc.) without touching the optimizer itself.
"""

import math

import networkx as nx
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0
AVG_URBAN_SPEED_KMH = 16.0  # rough NYC surface-street average incl. lights/traffic
DEFAULT_MIN_BUFFER_BIKES = 2  # a station below this is "at risk of running empty"
DEFAULT_DONOR_BUFFER_BIKES = 3  # extra bikes a donor keeps for itself, on top of min_buffer, before it'll give any away
COST_SCALE = 100  # scales minutes to an integer cost unit for network_simplex


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    dlat, dlng = lat2 - lat1, lng2 - lng1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def compute_surplus_deficit(
    stations_df,
    capacity_col="capacity",
    current_col="current_bikes_total",
    pred_net_flow_col="pred_net_flow_total",
    min_buffer=DEFAULT_MIN_BUFFER_BIKES,
    donor_buffer=DEFAULT_DONOR_BUFFER_BIKES,
):
    """
    stations_df needs: station_id, lat, lng, capacity_col, current_col, pred_net_flow_col.
    Adds predicted_bikes / surplus / deficit / available_to_donate columns.

    surplus = docks that will be short (station predicted over capacity --
    a hard, physical overfill risk), deficit = bikes that will be short
    (predicted below min_buffer). Both >= 0 and mutually exclusive as long
    as capacity > min_buffer, true for every real station. These two drive
    the map's red/orange/healthy status -- they only describe genuine risk,
    so they're deliberately NOT redefined below.

    available_to_donate is a separate, broader number: how many bikes a
    station could hand off to a truck and *still* be predicted to keep at
    least min_buffer + donor_buffer bikes for itself afterward. donor_buffer
    is extra headroom on top of min_buffer -- a station sitting exactly at
    "not currently at-risk" shouldn't be tapped down to the at-risk line the
    moment a truck visits it; the buffer keeps a real margin. Any station
    predicted comfortably stocked qualifies, not just ones over capacity --
    an overfull station's available_to_donate is naturally very large (it
    has far more than it needs), so it still sorts as the most attractive
    source; a merely-comfortable station is included too, just with less to
    give. This is what lets the optimizer proactively pull bikes from a
    nearby well-stocked station into a predicted-empty one even when
    nothing in view is technically overfull -- previously the optimizer
    could only relieve literal overfills, so a system with plenty of empty
    risk but zero overfull docks had nothing to recommend at all.
    """
    df = stations_df.copy()
    df["predicted_bikes"] = df[current_col] + df[pred_net_flow_col]
    df["surplus"] = (
        (df["predicted_bikes"] - df[capacity_col]).clip(lower=0).round().astype(int)
    )
    df["deficit"] = (
        (min_buffer - df["predicted_bikes"]).clip(lower=0).round().astype(int)
    )
    df["available_to_donate"] = (
        (df["predicted_bikes"] - min_buffer - donor_buffer)
        .clip(lower=0)
        .round()
        .astype(int)
    )
    return df


def solve_rebalancing(stations_df, top_k_neighbors=8):
    """
    stations_df: output of compute_surplus_deficit() -- needs station_id,
    lat, lng, available_to_donate, deficit. Returns a DataFrame of
    recommended moves: from_station_id, to_station_id, bikes, distance_km,
    minutes -- sorted by bikes descending (biggest moves first).

    Sources are every station with available_to_donate > 0 -- both stations
    at hard overfill risk (surplus > 0, which always implies a large
    available_to_donate too) and stations that are simply comfortably
    stocked for right now. Nothing here ranks "relieve an overfill" above
    "proactively top off a healthy station" -- both are just source nodes
    in the same min-cost-flow graph, so the solver picks whichever
    combination of moves minimizes total truck travel, same as always.

    Candidate edges are restricted to each source's K nearest deficit
    stations (both for realism -- trucks prefer nearby moves -- and to keep
    the bipartite graph small); this can never make the problem infeasible
    since leftover supply/demand that can't be matched within the k-nearest
    graph is absorbed by a zero-cost dummy node (standard unbalanced-
    transportation-problem technique), it just won't show up as a move.

    Distance-to-neighbors is computed with one vectorized numpy pairwise
    matrix rather than a per-source Python loop over sinks.iterrows() --
    proactive donors mean "sources" can now be most of the system (a few
    thousand stations) instead of just the handful that are literally
    overfull, and the old per-source/per-edge pandas .loc[] lookups scaled
    badly enough at that size to make the map visibly slow to load.
    """
    sources = stations_df[stations_df["available_to_donate"] > 0].reset_index(drop=True)
    sinks = stations_df[stations_df["deficit"] > 0].reset_index(drop=True)
    empty = pd.DataFrame(
        columns=["from_station_id", "to_station_id", "bikes", "distance_km", "minutes"]
    )
    if sources.empty or sinks.empty:
        return empty

    src_ids = sources["station_id"].to_numpy()
    src_donate = sources["available_to_donate"].to_numpy(dtype=int)
    src_lat = np.radians(sources["lat"].to_numpy(dtype=float))
    src_lng = np.radians(sources["lng"].to_numpy(dtype=float))
    snk_ids = sinks["station_id"].to_numpy()
    snk_deficit = sinks["deficit"].to_numpy(dtype=int)
    snk_lat = np.radians(sinks["lat"].to_numpy(dtype=float))
    snk_lng = np.radians(sinks["lng"].to_numpy(dtype=float))

    # haversine, vectorized: every source x every sink in one shot (broadcasting
    # (n_sources, 1) against (1, n_sinks) gives an (n_sources, n_sinks) matrix)
    dlat = snk_lat[None, :] - src_lat[:, None]
    dlng = snk_lng[None, :] - src_lng[:, None]
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(src_lat[:, None]) * np.cos(snk_lat[None, :]) * np.sin(dlng / 2) ** 2
    )
    dist_km = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    minutes_mat = (dist_km / AVG_URBAN_SPEED_KMH) * 60.0

    k = min(top_k_neighbors, len(sinks))
    # argpartition finds the k nearest per source in O(n) instead of a full O(n log n)
    # sort -- order within those k doesn't matter, the flow solver picks among them anyway
    nearest_idx = np.argpartition(dist_km, kth=k - 1, axis=1)[:, :k]

    G = nx.DiGraph()
    for sid, donate in zip(src_ids, src_donate):
        G.add_node(("src", sid), demand=-int(donate))
    for did, deficit in zip(snk_ids, snk_deficit):
        G.add_node(("snk", did), demand=int(deficit))

    total_supply = int(src_donate.sum())
    total_deficit = int(snk_deficit.sum())
    imbalance = total_supply - total_deficit  # >0: excess supply, <0: excess demand
    if imbalance != 0:
        G.add_node("dummy", demand=imbalance)
        if imbalance > 0:
            for sid, donate in zip(src_ids, src_donate):
                G.add_edge(("src", sid), "dummy", capacity=int(donate), weight=0)
        else:
            for did, deficit in zip(snk_ids, snk_deficit):
                G.add_edge("dummy", ("snk", did), capacity=int(deficit), weight=0)

    edge_meta = {}
    for i in range(len(sources)):
        sid, donate = src_ids[i], src_donate[i]
        for j in nearest_idx[i]:
            did, deficit = snk_ids[j], snk_deficit[j]
            cap = min(int(donate), int(deficit))
            if cap <= 0:
                continue
            km, minutes = float(dist_km[i, j]), float(minutes_mat[i, j])
            cost = max(1, round(minutes * COST_SCALE))
            u, v = ("src", sid), ("snk", did)
            G.add_edge(u, v, capacity=cap, weight=cost)
            edge_meta[(u, v)] = (km, minutes)

    try:
        flow_dict = nx.min_cost_flow(G)
    except nx.NetworkXUnfeasible:
        # can happen if a source's k-nearest sinks are all saturated by
        # closer competing sources -- widen the search once before giving up
        return solve_rebalancing(
            stations_df, top_k_neighbors=min(len(sinks), top_k_neighbors * 4)
        )

    rows = []
    for (u, v), (km, minutes) in edge_meta.items():
        bikes = flow_dict.get(u, {}).get(v, 0)
        if bikes > 0:
            rows.append(
                {
                    "from_station_id": u[1],
                    "to_station_id": v[1],
                    "bikes": int(bikes),
                    "distance_km": round(km, 2),
                    "minutes": round(minutes, 1),
                }
            )
    if not rows:
        return empty
    return (
        pd.DataFrame(rows).sort_values("bikes", ascending=False).reset_index(drop=True)
    )


def summarize_system_state(stations_df):
    """KPI tile numbers for the app header."""
    n_overfill = int((stations_df["surplus"] > 0).sum())
    n_empty_risk = int((stations_df["deficit"] > 0).sum())
    bikes_to_remove = int(stations_df["surplus"].sum())
    bikes_to_add = int(stations_df["deficit"].sum())
    return {
        "n_stations": len(stations_df),
        "n_overfill_risk": n_overfill,
        "n_empty_risk": n_empty_risk,
        "n_healthy": int(len(stations_df) - n_overfill - n_empty_risk),
        "bikes_to_remove": bikes_to_remove,
        "bikes_to_add": bikes_to_add,
    }
