"""
Citi Bike NYC rebalancing dashboard -- built for a Head of Operations /
truck-dispatch audience.

Given a point in time, predicts each station's net bike flow HORIZON hours
ahead (separately for classic and electric bikes), simulates current dock
occupancy, flags stations at risk of overfilling or running empty, and
solves a minimum-cost rebalancing plan (which trucks should move how many
bikes from where to where). See optimizer.py for the min-cost-flow
formulation.

Two modes, picked in the sidebar:
- Historical Playback (data_utils.py): replays a real 5-week historical
  window exactly as the models saw it at training/eval time. Fully
  reproducible, no network needed.
- Live (live_data.py): trains offline on ~3 years of history, then scores
  real-time GBFS + weather + events through those same trained models,
  auto-refreshing every 120s (or on demand). See the app's own "Live data"
  tab, live_data.py's module docstring, and gbfs.py for the full breakdown
  of what's real-live vs a historical stand-in and why.

Both modes produce the exact same snapshot shape (station_id, name, lat,
lng, borough, capacity, current_bikes_total, pred_net_flow_classic/electric/
total), so render_dashboard() below -- KPIs, map, moves/types/performance
tabs -- is fully shared between them; only the sidebar controls and the
data-fetch step differ.
"""

import gc
import json

import live_data
import pandas as pd
import pydeck as pdk
import streamlit as st
from data_utils import (
    BIKE_TYPES,
    HORIZON_HOURS,
    WARMUP_DAYS,
    available_timestamps,
    get_snapshot,
    load_demo_features,
    load_stations,
    simulate_occupancy_total,
)
from optimizer import compute_surplus_deficit, solve_rebalancing, summarize_system_state

st.set_page_config(
    page_title="Citi Bike NYC Rebalancing", page_icon="🚲", layout="wide"
)

STATUS_COLORS = {
    "overfill": [220, 60, 60],
    "empty": [235, 160, 40],
    "healthy": [90, 160, 110],
}
LIVE_REFRESH_SECONDS = 120

FEATURE_SOURCE_TABLE = pd.DataFrame(
    [
        {
            "Feature(s)": "hour, day_of_week, month, is_weekend, is_holiday",
            "Live source": "computed from the last fully-completed hour",
            "Until then / on failure": "none needed -- always exact",
        },
        {
            "Feature(s)": "lat, lng, borough, capacity",
            "Live source": "your station list; capacity is overwritten with real GBFS dock counts for matched stations",
            "Until then / on failure": "station list's estimated capacity, for stations GBFS doesn't cover",
        },
        {
            "Feature(s)": "current bikes (classic / electric split)",
            "Live source": "live GBFS station_status, this instant",
            "Until then / on failure": "none -- a station GBFS doesn't cover or match is left out of the view entirely",
        },
        {
            "Feature(s)": "temperature, humidity, precipitation, rain, snow, wind, cloud cover, weather code",
            "Live source": "Open-Meteo current conditions",
            "Until then / on failure": "mid-range defaults (e.g. 15°C, no precipitation) if the feed is unreachable",
        },
        {
            "Feature(s)": "event_count, major_event_count",
            "Live source": "NYC Open Data's live permitted-events feed, by borough",
            "Until then / on failure": "zero events if the feed is unreachable",
        },
        {
            "Feature(s)": "net_flow + its 1h/24h/168h lags, 24h rolling avg, same-slot-last-4-weeks avg",
            "Live source": "this app's own hourly snapshot log, per feature, once enough history has accumulated",
            "Until then / on failure": "a historical seasonal average for that station's hour-of-day/day-of-week",
        },
        {
            "Feature(s)": "departures, arrivals",
            "Live source": "none -- GBFS reports current dock counts, never flow",
            "Until then / on failure": "always the seasonal average",
        },
    ]
)

COMMON_ABOUT_MARKDOWN = f"""
**Data (~3 years, Aug 2023 - Jul 2026, full NYC system incl. Hoboken/JC) used to train both models:**
Citi Bike trip-history archive (S3), Open-Meteo historical hourly weather (Central Park),
and NYC Open Data's permitted-events history (borough-level).

**Leak safety:** the target is net flow **{HORIZON_HOURS} hours ahead** (SQL `LEAD`); every lag/rolling
feature (1h/24h/168h lag, 24h rolling average, same-weekday-hour rolling average) is computed
with SQL `LAG`/window functions using only *past* rows; the train/val/test split is **chronological**
(train through 2025-07-31, val through 2026-01-31, test after) rather than random k-fold, so no
future information leaks backward into training. This holds regardless of which mode below is
serving predictions -- it's a property of how the models were trained, not how they're served.

**Rebalancing travel cost/time:** a haversine-distance proxy at an assumed average city speed, not
real road-network routing (swap-in point documented in `optimizer.py`). The optimizer solves a
**transportation / min-cost-flow problem** (how many bikes should move from where to where) -- not
full multi-stop truck routing/scheduling (a harder VRP, out of scope).

**Known data issue, found and mitigated during pipeline development:** Citi Bike's own trip-history
exports label 123 stations (~4.6% of the system) inconsistently across months -- the same physical
dock shows up under two different `station_id` values in different months (e.g. `"6098.1"` vs.
`"6098.10"`, identical name/lat/lng). Left alone, this makes the two halves look like separate
stations -- including to the optimizer, which would recommend moving bikes from a station to
itself. Both ids are merged back into one station for every number and map marker shown here
(`station_id_map.csv`). What's *not* fully fixed: the two trained models still learned from each
half's fragmented, gappier history, so predictions for these ~123 stations are inherently noisier
than the system average -- the real fix is re-running the ETL with canonicalized ids from the start.
"""

HISTORICAL_ABOUT_MARKDOWN = f"""
**What's simulated vs real, in Historical Playback:**
- Weather, events, calendar, and all lag/rolling features for the selected snapshot: **real
  historical values**, exactly as the model saw at training/evaluation time.
- Current dock occupancy: **simulated** -- cumulative real historical net flow from a capacity/2
  seed at the start of the window (first {WARMUP_DAYS} days are a warm-up and not selectable).
- Station capacity: **estimated** from typical hourly activity (Citi Bike doesn't publish
  historical dock counts); Live mode overwrites this with real GBFS dock counts where it can.

**Why this mode exists at all, alongside Live:** GBFS only ever exposes *current* station state,
never history, so on a fresh deploy there's no live source yet for the lag/rolling features the
models need. Playback sidesteps that entirely by replaying hours where the real values are already
known -- a fully-reproducible, network-free demo and fallback, useful even once Live mode works.
"""

LIVE_ABOUT_MARKDOWN = f"""
**What Live mode does:** the same two trained XGBoost models above, unchanged -- Live mode only
changes how the *input row* is built each cycle: real-time GBFS station status, live weather, and
live permitted events, reshaped into the exact feature layout the models were trained on. See the
**Live data** tab (next to this one) for exactly which features are real-time vs. a stand-in right
now, with live coverage numbers.

**"Right now" actually means the last fully-completed hour:** a training row at hour H holds flow
that happened *during* [H, H+1) and predicts [H+3, H+4). At, say, 14:37, the hour [14:00, 14:37) is
still in progress and has no real net-flow value yet -- so Live mode uses 13:00 as "the row's hour"
and predicts conditions ~{HORIZON_HOURS}h from there, mirroring training exactly. Weather/events are
still fetched for right now (fine, since both change slowly relative to an hour).

**Refresh:** auto-refreshes every {LIVE_REFRESH_SECONDS}s, or hit **Refresh now** at the top of the
page for an immediate poll. Every refresh also logs one snapshot per station to this app's own
history log (deduplicated to one row per station per hour, regardless of poll frequency) -- that
log is what lets the flow-lag features graduate from a seasonal proxy to real self-observed values
over the following days/week, see the Live data tab.

**Known live-only gap:** the live feed checked during development covers what looks like all five
NYC boroughs but not Hoboken/Jersey City -- roughly 200 stations real to the system (and in the
historical training data) with no live counterpart. Those stations simply don't appear in Live mode
(Historical Playback still covers them). Matching live stations to this app's station list is done
by nearest lat/lng (Citi Bike's live feed uses a different id scheme than the historical archive,
with no published crosswalk), accepted only within ~50m -- see `gbfs.py`.

**A subtler fix worth calling out:** for the ~123 stations mentioned above with fragmented
historical ids, live mode has to pick *one* id to represent the physical station to the model (live
data only has one combined signal per dock, unlike training). Checked rather than assumed: the
"canonical" id `station_id_map.csv` designates is picked by location, not by which id the model
actually has training history for -- for a majority of these stations the "canonical" pick turns out
to be the *less*-represented half, and for a few it's an id the model never saw a single training
row under. Live mode instead feeds whichever raw id the model has the most real history for, per
bike type. Checked empirically, not just in theory: across the affected stations, that choice
changes the 3-hour forecast by 0.6 bikes on average, and by several bikes for a few of them --
worth getting right rather than picking arbitrarily.
"""


# ============================================================== shared dashboard
def render_moves_tab(sd, moves):
    if moves.empty:
        n_surplus = int((sd["surplus"] > 0).sum())
        n_deficit = int((sd["deficit"] > 0).sum())
        if n_surplus == 0 and n_deficit == 0:
            st.success(
                "No rebalancing moves needed -- every station in view is predicted healthy."
            )
        elif n_deficit > 0 and n_surplus == 0:
            st.warning(
                f"{n_deficit} station(s) are predicted to run low on bikes, but no station in view has "
                f"a surplus to pull from right now -- this needs bikes brought in from a depot/warehouse, "
                f"not redistributed between stations. The optimizer only ever moves bikes *between* "
                f"stations, so it has nothing to recommend here even though the shortage is real."
            )
        elif n_surplus > 0 and n_deficit == 0:
            st.warning(
                f"{n_surplus} station(s) are predicted to overfill, but no station in view has a deficit "
                f"to send bikes to -- this needs bikes taken *out* of the system, not redistributed. The "
                f"optimizer only ever moves bikes *between* stations, so it has nothing to recommend here "
                f"even though the overfill risk is real."
            )
        else:
            st.warning(
                f"{n_surplus} surplus and {n_deficit} deficit station(s) exist, but none matched within "
                f"the routing search radius (try raising 'Candidate destinations per pickup' in Advanced "
                f"settings)."
            )
        return
    name_lookup_s = sd.set_index("station_id")["name"]
    display = moves.copy()
    display["From"] = display["from_station_id"].map(name_lookup_s)
    display["To"] = display["to_station_id"].map(name_lookup_s)
    display = display.rename(
        columns={
            "bikes": "Bikes",
            "distance_km": "Distance (km)",
            "minutes": "Est. minutes",
        }
    )
    display = display[["From", "To", "Bikes", "Distance (km)", "Est. minutes"]]
    st.dataframe(display, width="stretch", hide_index=True)

    st.markdown("##### Driver view -- next move")
    options = [
        f"#{i + 1}: {r.From} -> {r.To} ({r.Bikes} bikes)" for i, r in display.iterrows()
    ]
    pick = st.selectbox("Select a move", options)
    row = display.iloc[options.index(pick)]
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Pick up", row["From"])
    mc2.metric("Drop off", row["To"])
    mc3.metric("Bikes / ETA", f"{row['Bikes']} bikes, ~{row['Est. minutes']:.0f} min")


def render_types_tab(sd):
    st.markdown("##### System-wide predicted net flow, next " + f"{HORIZON_HOURS}h")
    total_classic = sd["pred_net_flow_classic"].sum()
    total_electric = sd["pred_net_flow_electric"].sum()
    tc1, tc2 = st.columns(2)
    tc1.metric("Classic bikes -- net flow", f"{total_classic:+.0f}")
    tc2.metric("Electric bikes -- net flow", f"{total_electric:+.0f}")

    by_borough = (
        sd.groupby("borough")[["pred_net_flow_classic", "pred_net_flow_electric"]]
        .sum()
        .rename(
            columns={
                "pred_net_flow_classic": "Classic",
                "pred_net_flow_electric": "Electric",
            }
        )
    )
    by_borough.index.name = "Region"
    st.bar_chart(by_borough)
    st.caption(
        "Positive = net arrivals (station filling up), negative = net departures (station draining). "
        "Classic and electric bikes are modeled and predicted completely separately -- two independent "
        "XGBoost models -- since usage patterns differ (e-bikes skew toward longer/commute trips)."
    )
    sd_scatter = sd.rename(columns={"borough": "Region"})
    st.scatter_chart(
        sd_scatter,
        x="pred_net_flow_classic",
        y="pred_net_flow_electric",
        color="Region",
    )


def render_perf_tab():
    st.markdown(
        "##### Held-out test-set accuracy (most recent ~6 months, never seen in training)"
    )
    rows = []
    for bt in BIKE_TYPES:
        with open(f"models/metrics_{bt}.json") as f:
            m = json.load(f)
        rows.append({"Bike type": bt, "Model": "XGBoost", **m["model"]})
        rows.append(
            {"Bike type": bt, "Model": "Baseline: persistence", **m["baseline_persist"]}
        )
        rows.append(
            {
                "Bike type": bt,
                "Model": "Baseline: same slot last 4wk",
                **m["baseline_seasonal"],
            }
        )
    perf_df = pd.DataFrame(rows).rename(
        columns={"rmse": "RMSE", "mae": "MAE", "r2": "R2"}
    )
    st.dataframe(perf_df, width="stretch", hide_index=True)
    st.caption(
        "XGBoost beats both naive baselines on every metric for both bike types -- persistence and "
        "seasonal-average baselines both score a *negative* R2 (worse than predicting the mean), "
        "which is expected for noisy, low-count, per-station hourly flow."
    )

    st.markdown("##### Feature importance (gain)")
    fc1, fc2 = st.columns(2)
    for col, bt in zip([fc1, fc2], BIKE_TYPES):
        fi = (
            pd.read_csv(f"models/feature_importance_{bt}.csv")
            .head(10)
            .set_index("feature")
        )
        col.markdown(f"**{bt}**")
        col.bar_chart(fi)


def render_dashboard(
    snapshot,
    selected_boroughs,
    min_buffer,
    top_k,
    title_suffix,
    about_markdown,
    extra_tabs=None,
):
    """
    snapshot: unified per-station DataFrame (station_id, name, lat, lng,
    borough, capacity, current_bikes_total, pred_net_flow_classic/electric/
    total) -- same shape whether it came from Historical Playback
    (data_utils.get_snapshot) or Live serving (live_data.assemble_live_snapshot).
    title_suffix: short mode badge appended to the page title.
    about_markdown: mode-specific body appended after the shared About content.
    extra_tabs: optional list of (label, render_fn) for mode-specific tabs
    (e.g. Live's coverage/freshness tab), inserted before "About this demo".
    """
    snapshot = snapshot[snapshot["borough"].isin(selected_boroughs)].reset_index(
        drop=True
    )
    if snapshot.empty:
        st.warning("No stations match the current region filter.")
        return

    sd = compute_surplus_deficit(snapshot, min_buffer=min_buffer)
    sd["status"] = "healthy"
    sd.loc[sd["surplus"] > 0, "status"] = "overfill"
    sd.loc[sd["deficit"] > 0, "status"] = "empty"

    moves = solve_rebalancing(sd, top_k_neighbors=top_k)
    kpi = summarize_system_state(sd)

    # ------------------------------------------------------------ KPIs
    st.title(f"Citi Bike NYC -- Rebalancing Dashboard   {title_suffix}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Stations in view", f"{kpi['n_stations']:,}")
    c2.metric(
        "Overfill risk",
        f"{kpi['n_overfill_risk']:,}",
        help="Predicted to exceed capacity",
    )
    c3.metric(
        "Empty risk",
        f"{kpi['n_empty_risk']:,}",
        help="Predicted to fall below the min-buffer",
    )
    c4.metric("Bikes to remove", f"{kpi['bikes_to_remove']:,}")
    c5.metric("Bikes to add", f"{kpi['bikes_to_add']:,}")

    # ------------------------------------------------------------- map
    st.subheader("Station status & recommended moves")

    sd_map = sd.copy()
    sd_map["color"] = sd_map["status"].map(STATUS_COLORS)
    sd_map["radius"] = 40 + sd_map[["surplus", "deficit"]].max(axis=1) * 15
    # display only -- a station can't physically hold negative bikes or more than its
    # capacity, but the raw regression output can land slightly outside [0, capacity]
    # (e.g. -0.08 predicted for a near-empty station). surplus/deficit above were already
    # computed from the *unclipped* value on purpose (it's a genuine urgency signal for
    # the optimizer -- a station predicted at -5 needs bikes more urgently than one at
    # -0.1, even though both physically bottom out at 0) -- this clip only cleans up the
    # number shown in the map tooltip, it doesn't feed back into any of the routing math.
    sd_map["predicted_bikes"] = (
        sd_map["predicted_bikes"].clip(lower=0).clip(upper=sd_map["capacity"]).round(1)
    )
    # pydeck only takes ONE tooltip template for the whole Deck, shared across every
    # layer -- it does a dumb {field} string-replace on whatever object is under the
    # cursor, and any placeholder that isn't a column on THAT layer's data is left
    # in the tooltip verbatim instead of being blanked out (that's why hovering a
    # move-line used to show literal "{name}" / "Region: {borough}" text -- those
    # are station-dot fields the arc data doesn't have). Fix: give both dataframes
    # the same three generic placeholder columns (title/line1/line2), blank ones as
    # "" rather than omitting them, and keep the actual <b>/<br/> markup in the
    # static template below rather than in the data -- pydeck HTML-escapes
    # substituted field values (so a "<br/>" living inside the data renders as the
    # literal text "<br/>", not a line break; the markup has to stay in the template).
    sd_map["tooltip_title"] = sd_map["name"]
    sd_map["tooltip_line1"] = (
        "Region: " + sd_map["borough"].astype(str) + " | Status: " + sd_map["status"]
    )
    sd_map["tooltip_line2"] = (
        "Predicted bikes: "
        + sd_map["predicted_bikes"].astype(str)
        + " / cap "
        + sd_map["capacity"].astype(str)
    )

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=sd_map,
            get_position="[lng, lat]",
            get_fill_color="color",
            get_radius="radius",
            pickable=True,
            opacity=0.75,
        )
    ]

    if not moves.empty:
        name_lookup = sd.set_index("station_id")[["name", "lat", "lng"]]
        arcs = moves.join(name_lookup.add_prefix("from_"), on="from_station_id")
        arcs = arcs.join(name_lookup.add_prefix("to_"), on="to_station_id")
        arcs["width"] = arcs["bikes"].clip(upper=12)
        arcs["tooltip_title"] = arcs["bikes"].astype(str) + " bikes"
        arcs["tooltip_line1"] = arcs["from_name"] + " -> " + arcs["to_name"]
        arcs["tooltip_line2"] = (
            "net flow: "
            + arcs["distance_km"].astype(str)
            + " km (~"
            + arcs["minutes"].astype(str)
            + " min)"
        )
        layers.append(
            pdk.Layer(
                "ArcLayer",
                data=arcs,
                get_source_position="[from_lng, from_lat]",
                get_target_position="[to_lng, to_lat]",
                get_width="width",
                get_source_color=[220, 60, 60, 160],
                get_target_color=[40, 160, 90, 200],
                pickable=True,
            )
        )

    view_state = pdk.ViewState(latitude=40.745, longitude=-73.97, zoom=10.3, pitch=35)
    tooltip = {
        "html": "<b>{tooltip_title}</b><br/>{tooltip_line1}<br/>{tooltip_line2}",
        "style": {"backgroundColor": "steelblue", "color": "white"},
    }
    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip=tooltip,
            map_provider="carto",
            map_style="light",
        )
    )
    st.caption(
        "🔴 red = overfill risk (docks running out)   🟠 orange = empty risk (bikes running out)   "
        "🟢 green = healthy   |   arcs show the recommended rebalancing plan (red end = pickup, green end = drop-off)"
    )

    # ------------------------------------------------------------ tabs
    tab_specs = [
        ("🚚 Rebalancing moves", lambda: render_moves_tab(sd, moves)),
        ("⚡ Classic vs e-bike", lambda: render_types_tab(sd)),
        ("📊 Model performance", render_perf_tab),
    ]
    if extra_tabs:
        tab_specs += list(extra_tabs)
    tab_specs.append(
        (
            "ℹ️ About this demo",
            lambda: st.markdown(COMMON_ABOUT_MARKDOWN + about_markdown),
        )
    )

    tabs = st.tabs([label for label, _ in tab_specs])
    for tab, (_, render_fn) in zip(tabs, tab_specs):
        with tab:
            render_fn()


# ============================================================== live mode
def render_live_status_tab(out, meta):
    n_canon = meta["crosswalk_stats"]["n_canonical"]
    n_matched = meta["n_matched"]
    st.markdown(
        f"**Live station coverage:** {n_matched:,} / {n_canon:,} stations matched to the live GBFS "
        f"feed by nearest lat/lng (within ~50m). By region, matched vs. total in your station list:"
    )
    all_stations = load_stations()
    total_by_borough = all_stations.groupby("borough").size().rename("Total stations")
    matched_by_borough = out.groupby("borough").size().rename("Matched live")
    cov_by_borough = (
        pd.concat([matched_by_borough, total_by_borough], axis=1).fillna(0).astype(int)
    )
    cov_by_borough.index.name = "Region"
    cov_by_borough["Coverage"] = (
        cov_by_borough["Matched live"] / cov_by_borough["Total stations"]
    ).map(lambda x: f"{x * 100:.0f}%")
    st.dataframe(
        cov_by_borough.sort_values("Total stations", ascending=False), width="stretch"
    )

    st.markdown(
        f"**Live history collected:** {min(meta['history_depth_hours'], 168):.1f}h / 168h (1 week). "
        "Once a lag feature's window is fully covered by this app's own snapshot log, it uses real "
        "self-observed values instead of the seasonal proxy -- coverage below is the fraction of "
        "*matched* stations currently using the real value for each feature, per bike type."
    )
    cov_rows = []
    for bt in BIKE_TYPES:
        cov_rows.append(
            {
                "Bike type": bt,
                **{c: f"{v * 100:.0f}%" for c, v in meta["coverage"][bt].items()},
            }
        )
    st.dataframe(pd.DataFrame(cov_rows), width="stretch", hide_index=True)

    st.markdown("**Where every feature comes from live, and its fallback:**")
    st.dataframe(FEATURE_SOURCE_TABLE, width="stretch", hide_index=True)
    st.caption(
        f"Weather feed reachable: {'✅' if meta['weather_ok'] else '❌ (using defaults)'}   |   "
        f"Events feed reachable: {'✅' if meta['events_ok'] else '❌ (assuming zero events)'}"
    )


@st.fragment(run_every=LIVE_REFRESH_SECONDS)
def render_live_view(selected_boroughs, min_buffer, top_k):
    out, meta = live_data.assemble_live_snapshot()

    header_col, refresh_col = st.columns([5, 1])
    with refresh_col:
        st.button(
            "🔄 Refresh now", width="stretch"
        )  # any click reruns just this fragment, right away

    if not meta.get("ok"):
        with header_col:
            st.caption("🔴 **Live** -- currently unreachable")
        reason = (meta.get("reason") or "unknown error").rstrip(".")
        st.error(
            f"Couldn't load live data: {reason}. Switch to **Historical "
            f"Playback** in the sidebar to keep exploring -- this will keep retrying every "
            f"{LIVE_REFRESH_SECONDS}s, or hit Refresh now."
        )
        cw = meta.get("crosswalk_stats")
        if cw:
            st.caption(
                f"(Reached the station list -- {cw['n_matched']:,}/{cw['n_canonical']:,} stations "
                f"matched to live GBFS ids -- but the live status feed itself failed this cycle.)"
            )
        return

    reference_hour = meta["reference_hour"]
    target_hour = reference_hour + pd.Timedelta(hours=HORIZON_HOURS)
    with header_col:
        st.caption(
            f"🔴 **Live** -- as of {meta['as_of'].strftime('%H:%M:%S')}   |   "
            f"scoring the last completed hour, **{reference_hour.strftime('%a %H:%M')}**, to predict "
            f"**{HORIZON_HOURS}h ahead** -> target **{target_hour.strftime('%a %H:%M')}**   |   "
            f"auto-refreshes every {LIVE_REFRESH_SECONDS}s"
        )

    n_canon = meta["crosswalk_stats"]["n_canonical"]
    cA, cB, cC = st.columns(3)
    cA.metric(
        "Live station coverage",
        f"{meta['n_matched']:,} / {n_canon:,}",
        help="Matched via nearest lat/lng to the live GBFS feed. Full breakdown in the Live data tab.",
    )
    cB.metric(
        "Live history collected",
        f"{min(meta['history_depth_hours'], 168):.0f}h / 168h",
        help="Once this reaches 168h (1 week), every lag feature can be fully real instead of proxied.",
    )
    avg_net_flow_cov = sum(meta["coverage"][bt]["net_flow"] for bt in BIKE_TYPES) / len(
        BIKE_TYPES
    )
    cC.metric(
        "Net-flow realness",
        f"{avg_net_flow_cov * 100:.0f}%",
        help="Share of matched stations where the latest hour's net flow is real, not a seasonal proxy.",
    )

    render_dashboard(
        out,
        selected_boroughs,
        min_buffer,
        top_k,
        title_suffix="🔴 Live",
        about_markdown=LIVE_ABOUT_MARKDOWN,
        extra_tabs=[("🛰️ Live data", lambda: render_live_status_tab(out, meta))],
    )


# --------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🚲 Rebalancing Controls")

    mode = st.radio(
        "Mode", ["📼 Historical Playback", "🔴 Live"], index=1, horizontal=True
    )
    is_live = mode.startswith("🔴")

    # Memory footprint, not correctness: each mode's data layer is cached
    # (@st.cache_data / @st.cache_resource) so it stays resident across
    # reruns for fast switching back -- but on a low-memory host (e.g.
    # Streamlit Community Cloud's free ~1GB tier) holding BOTH Historical
    # Playback's occupancy simulation (dense matrix, ~2,500 stations x
    # weeks of hours) and Live's fetched state in memory at once can push
    # past the limit and get the process OOM-killed mid-switch -- which
    # Streamlit shows as a generic "Oh no." with no Python traceback,
    # since the process dies before it can log one. Dropping the mode
    # you're navigating AWAY from keeps only one mode's heavy data
    # resident at a time, trading "instant" back-and-forth switching for
    # "a few seconds to reload" in exchange for not crashing.
    if "last_mode_is_live" not in st.session_state:
        st.session_state.last_mode_is_live = is_live
    elif st.session_state.last_mode_is_live != is_live:
        if is_live:
            simulate_occupancy_total.clear()
            load_demo_features.clear()
        else:
            live_data.load_seasonal_baseline.clear()
        st.session_state.last_mode_is_live = is_live
        # gc.collect() forces Python to actually reclaim that memory right
        # now rather than whenever the interpreter next feels like it, and
        # st.rerun() restarts the script fresh *after* that reclaim instead
        # of continuing to build the new mode's dashboard in the same pass
        # -- otherwise the moment of switching is exactly when memory usage
        # peaks highest (old mode's data not yet fully released, new
        # mode's data actively being built), which is the most likely spot
        # for a constrained host to OOM-kill the process mid-switch. This
        # trades one extra rerun (a brief blank flash) for never rendering
        # a new mode on top of a not-yet-cleared old one.
        gc.collect()
        st.rerun()

    if is_live:
        st.caption(
            f"Trains offline on ~3 years of history, scores real-time GBFS + weather + events every "
            f"{LIVE_REFRESH_SECONDS}s. See the **Live data** tab for source/coverage detail."
        )
    else:
        timestamps = available_timestamps()
        default_idx = len(timestamps) - 1
        ts_idx = st.select_slider(
            "Snapshot time",
            options=list(range(len(timestamps))),
            value=default_idx,
            format_func=lambda i: timestamps[i].strftime("%a %b %d, %Y  %H:%M"),
        )
        selected_ts = timestamps[ts_idx]
        st.caption(
            f"Predicting **{HORIZON_HOURS}h ahead** -> "
            f"target time **{(selected_ts + pd.Timedelta(hours=HORIZON_HOURS)).strftime('%a %b %d, %H:%M')}**"
        )

    stations_all = load_stations()
    boroughs = sorted(stations_all["borough"].unique())
    selected_boroughs = st.multiselect("Regions", boroughs, default=boroughs)

    with st.expander("Advanced settings"):
        min_buffer = st.slider("Min-buffer bikes (empty-risk threshold)", 0, 6, 2)
        top_k = st.slider("Candidate destinations per pickup (routing)", 3, 15, 8)

    st.divider()
    if is_live:
        st.caption(
            "🔴 Live mode needs outbound internet access to GBFS/Open-Meteo/NYC Open Data. If this "
            "environment doesn't have it, switch back to **Historical Playback**, which runs entirely "
            "offline."
        )
    else:
        st.caption(
            "Demo runs on a real historical window (with real weather/events/lags) by default -- "
            "switch to **Live** above to score real-time conditions instead. See the **About** tab "
            "for why playback exists at all."
        )

# --------------------------------------------------------------- render
# Wrapped so a transient failure (e.g. a live data source timing out, or a
# memory spike right at the moment of a mode switch on a constrained host)
# shows a recoverable in-app message instead of Streamlit's hard "Oh no."
# crash page, which loses all sidebar state and forces a full page reload.
try:
    if is_live:
        render_live_view(selected_boroughs, min_buffer, top_k)
    else:
        snapshot = get_snapshot(selected_ts)
        render_dashboard(
            snapshot,
            selected_boroughs,
            min_buffer,
            top_k,
            title_suffix="📼 Playback",
            about_markdown=HISTORICAL_ABOUT_MARKDOWN,
        )
except Exception as e:
    st.error(
        "This mode hit an error while loading -- often a transient hiccup right at a mode "
        "switch (a live data source timing out, or a memory spike on a constrained host). "
        "Your sidebar settings are preserved."
    )
    if not is_live:
        st.info(
            "Historical Playback is fully self-contained (no network needed) -- try **Refresh** below."
        )
    else:
        st.info(
            "Live mode needs outbound internet access to GBFS/Open-Meteo/NYC Open Data. If this keeps "
            "happening, switch back to **Historical Playback** in the sidebar, which always works offline."
        )
    if st.button("🔄 Refresh"):
        st.rerun()
    with st.expander("Technical details"):
        st.exception(e)
