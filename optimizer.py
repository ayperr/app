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
import pandas as pd

EARTH_RADIUS_KM = 6371.0
AVG_URBAN_SPEED_KMH = 16.0  # rough NYC surface-street average incl. lights/traffic
DEFAULT_MIN_BUFFER_BIKES = 2  # a station below this is "at risk of running empty"
DEFAULT_DONOR_BUFFER_BIKES = 3  # extra bikes a donor keeps for itself, on top of min_buffer, before it'll give any away
COST_SCALE = 100              # scales minutes to an integer cost unit for network_simplex


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    dlat, dlng = lat2 - lat1, lng2 - lng1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def travel_time_minutes(lat1, lng1, lat2, lng2):
    return (haversine_km(lat1, lng1, lat2, lng2) / AVG_URBAN_SPEED_KMH) * 60.0


def compute_surplus_deficit(stations_df, capacity_col="capacity", current_col="current_bikes_total",
                             pred_net_flow_col="pred_net_flow_total", min_buffer=DEFAULT_MIN_BUFFER_BIKES,
                             donor_buffer=DEFAULT_DONOR_BUFFER_BIKES):
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
    df["surplus"] = (df["predicted_bikes"] - df[capacity_col]).clip(lower=0).round().astype(int)
    df["deficit"] = (min_buffer - df["predicted_bikes"]).clip(lower=0).round().astype(int)
    df["available_to_donate"] = (
        (df["predicted_bikes"] - min_buffer - donor_buffer).clip(lower=0).round().astype(int)
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
    """
    sources = stations_df[stations_df["available_to_donate"] > 0].reset_index(drop=True)
    sinks = stations_df[stations_df["deficit"] > 0].reset_index(drop=True)
    empty = pd.DataFrame(columns=["from_station_id", "to_station_id", "bikes", "distance_km", "minutes"])
    if sources.empty or sinks.empty:
        return empty

    G = nx.DiGraph()
    for _, s in sources.iterrows():
        G.add_node(("src", s.station_id), demand=-int(s.available_to_donate))
    for _, d in sinks.iterrows():
        G.add_node(("snk", d.station_id), demand=int(d.deficit))

    total_supply = int(sources["available_to_donate"].sum())
    total_deficit = int(sinks["deficit"].sum())
    imbalance = total_supply - total_deficit  # >0: excess supply, <0: excess demand
    if imbalance != 0:
        G.add_node("dummy", demand=imbalance)
        if imbalance > 0:
            for _, s in sources.iterrows():
                G.add_edge(("src", s.station_id), "dummy", capacity=int(s.available_to_donate), weight=0)
        else:
            for _, d in sinks.iterrows():
                G.add_edge("dummy", ("snk", d.station_id), capacity=int(d.deficit), weight=0)

    edge_meta = {}
    for _, s in sources.iterrows():
        dists = sorted(
            ((d.station_id, haversine_km(s.lat, s.lng, d.lat, d.lng)) for _, d in sinks.iterrows()),
            key=lambda x: x[1],
        )
        for dest_id, km in dists[:top_k_neighbors]:
            d = sinks.loc[sinks.station_id == dest_id].iloc[0]
            minutes = (km / AVG_URBAN_SPEED_KMH) * 60.0
            cap = min(int(s.available_to_donate), int(d.deficit))
            if cap <= 0:
                continue
            cost = max(1, round(minutes * COST_SCALE))
            u, v = ("src", s.station_id), ("snk", dest_id)
            G.add_edge(u, v, capacity=cap, weight=cost)
            edge_meta[(u, v)] = (km, minutes)

    try:
        flow_dict = nx.min_cost_flow(G)
    except nx.NetworkXUnfeasible:
        # can happen if a source's k-nearest sinks are all saturated by
        # closer competing sources -- widen the search once before giving up
        return solve_rebalancing(stations_df, top_k_neighbors=min(len(sinks), top_k_neighbors * 4))

    rows = []
    for (u, v), (km, minutes) in edge_meta.items():
        bikes = flow_dict.get(u, {}).get(v, 0)
        if bikes > 0:
            rows.append({
                "from_station_id": u[1], "to_station_id": v[1],
                "bikes": int(bikes), "distance_km": round(km, 2), "minutes": round(minutes, 1),
            })
    if not rows:
        return empty
    return pd.DataFrame(rows).sort_values("bikes", ascending=False).reset_index(drop=True)


def summarize_system_state(stations_df):
    """KPI tile numbers for the app header."""
    n_overfill = int((stations_df["surplus"] > 0).sum())
    n_empty_risk = int((stations_df["deficit"] > 0).sum())
    bikes_to_remove = int(stations_df["surplus"].sum())
    bikes_to_add = int(stations_df["deficit"].sum())
    return {
        "n_stations": int(len(stations_df)),
        "n_overfill_risk": n_overfill,
        "n_empty_risk": n_empty_risk,
        "n_healthy": int(len(stations_df) - n_overfill - n_empty_risk),
        "bikes_to_remove": bikes_to_remove,
        "bikes_to_add": bikes_to_add,
    }
