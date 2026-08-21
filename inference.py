"""
Loads the two trained XGBoost models (one per bike type) and predicts
HORIZON-hours-ahead net flow (arrivals - departures) from a feature-table
snapshot.

The saved metadata (feature_cols_*.json, categories_*.json) pins down the
exact feature order and categorical value sets used at training time, so
predictions here are reproducible regardless of what order columns arrive
in or which stations happen to be present in a given snapshot.
"""
import json

import pandas as pd
import streamlit as st
import xgboost as xgb

MODELS = "models"


@st.cache_resource(show_spinner="Loading trained models...")
def load_model_bundle(bike_type):
    model = xgb.XGBRegressor()
    model.load_model(f"{MODELS}/xgb_{bike_type}.json")
    with open(f"{MODELS}/feature_cols_{bike_type}.json") as f:
        feature_cols = json.load(f)
    with open(f"{MODELS}/categories_{bike_type}.json") as f:
        categories = json.load(f)
    with open(f"{MODELS}/metrics_{bike_type}.json") as f:
        metrics = json.load(f)
    return {"model": model, "feature_cols": feature_cols, "categories": categories, "metrics": metrics}


def predict_net_flow(bike_type, snapshot_df):
    """
    snapshot_df: rows from that bike type's feature table for a single
    (date, hour) -- i.e. one row per station. Returns a pandas Series of
    predicted net_flow (HORIZON hours ahead) indexed by station_id (str).
    """
    if snapshot_df.empty:
        return pd.Series(dtype="float32")

    bundle = load_model_bundle(bike_type)
    feature_cols = bundle["feature_cols"]["all"]
    numeric_cols = bundle["feature_cols"]["numeric"]
    cats = bundle["categories"]

    df = snapshot_df.copy()
    station_ids = df["station_id"].astype(str)
    df["station_id"] = station_ids.astype(pd.CategoricalDtype(categories=sorted(cats["station_id"])))
    df["borough"] = df["borough"].astype(str).astype(pd.CategoricalDtype(categories=sorted(cats["borough"])))
    for c in numeric_cols:
        df[c] = df[c].astype("float32")

    X = df[feature_cols]
    pred = bundle["model"].predict(X)
    return pd.Series(pred, index=station_ids.to_numpy(), name="pred_net_flow")
