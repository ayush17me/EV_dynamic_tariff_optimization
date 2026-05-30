"""
Data Preprocessing Module
Handles cleaning, alignment, and merging of ACN and UrbanEV datasets.

Responsibilities:
- Load raw data from both sources
- Clean missing values, duplicates, type issues
- Align timestamps to common granularity
- Merge into unified schema
- Document all assumptions
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    ACN_RAW_JSON, ACN_RAW_CSV, URBANEV_RAW_DIR,
    MERGED_SESSIONS_CSV, TIME_GRANULARITY, DATA_RAW,
)
from src.utils import get_logger, load_csv, save_csv, data_quality_report

logger = get_logger("preprocessing")


# ══════════════════════════════════════════════
#  ACN-Data Processing
# ══════════════════════════════════════════════

def load_acn_data(filepath: Path = None) -> pd.DataFrame:
    """
    Load ACN dataset from CSV (preferred) or JSON.

    Assumptions documented:
    - Sessions with missing kWhDelivered are dropped (cannot infer energy)
    - Sessions with negative duration are dropped (data errors)
    """
    # Prefer CSV (already converted)
    csv_path = filepath or ACN_RAW_CSV
    json_path = ACN_RAW_JSON

    if csv_path.exists():
        logger.info(f"Loading ACN data from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
    elif json_path.exists():
        logger.info(f"Loading ACN data from JSON: {json_path}")
        df = load_acn_json(json_path)
    else:
        logger.error("No ACN data file found!")
        return pd.DataFrame()

    logger.info(f"  → Loaded {len(df)} raw ACN sessions, {df.shape[1]} columns")
    return df


def load_acn_json(filepath: Path = ACN_RAW_JSON) -> pd.DataFrame:
    """Load ACN dataset from JSON and convert to DataFrame."""
    logger.info(f"Loading ACN JSON from {filepath}")

    with open(filepath, "r") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict) and "data" in raw_data:
        records = raw_data["data"]
    elif isinstance(raw_data, list):
        records = raw_data
    else:
        records = list(raw_data.values()) if isinstance(raw_data, dict) else raw_data

    df = pd.DataFrame(records)
    logger.info(f"  → Loaded {len(df)} raw ACN sessions")
    return df


def clean_acn_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize ACN session data.

    Steps:
    1. Parse timestamps
    2. Compute session duration
    3. Handle missing values
    4. Remove invalid sessions
    5. Standardize column names
    """
    df = df.copy()
    logger.info("Cleaning ACN data...")

    # ── Step 1: Parse timestamps ──
    time_cols = ["connectionTime", "disconnectTime", "doneChargingTime"]
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # ── Step 2: Compute session duration (minutes) ──
    if "connectionTime" in df.columns and "disconnectTime" in df.columns:
        df["session_duration_min"] = (
            (df["disconnectTime"] - df["connectionTime"]).dt.total_seconds() / 60
        )
        if "doneChargingTime" in df.columns:
            df["charging_duration_min"] = (
                (df["doneChargingTime"] - df["connectionTime"]).dt.total_seconds() / 60
            )

    # ── Step 3: Handle missing values ──
    initial_count = len(df)

    if "kWhDelivered" in df.columns:
        df = df.dropna(subset=["kWhDelivered"])
        logger.info(f"  Dropped {initial_count - len(df)} rows with missing kWhDelivered")

    if "charging_duration_min" in df.columns:
        df["charging_duration_min"] = df["charging_duration_min"].fillna(
            df.get("session_duration_min", 0)
        )

    # ── Step 4: Remove invalid sessions ──
    if "session_duration_min" in df.columns:
        invalid_mask = df["session_duration_min"] <= 0
        df = df[~invalid_mask]
        logger.info(f"  Removed {invalid_mask.sum()} sessions with non-positive duration")

    if "kWhDelivered" in df.columns:
        invalid_mask = df["kWhDelivered"] <= 0
        df = df[~invalid_mask]
        logger.info(f"  Removed {invalid_mask.sum()} sessions with non-positive energy")

    # ── Step 5: Standardize column names ──
    rename_map = {
        "connectionTime": "session_start",
        "disconnectTime": "session_end",
        "doneChargingTime": "charging_end",
        "kWhDelivered": "energy_kwh",
        "stationID": "station_id",
        "siteID": "site_id",
        "sessionID": "session_id",
        "userID": "user_id",
        "clusterID": "cluster_id",
        "spaceID": "space_id",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Add source label
    df["data_source"] = "ACN"

    logger.info(f"  → Clean ACN data: {df.shape}")
    return df


# ══════════════════════════════════════════════
#  UrbanEV (ST-EVCDP) Processing
# ══════════════════════════════════════════════

def load_urbanev_data(data_dir: Path = URBANEV_RAW_DIR) -> dict:
    """
    Load UrbanEV dataset components.

    UrbanEV data structure:
    - time.csv: 8,640 timestamps (30 days × 288 intervals = 5-min resolution)
    - volume.csv: Charging volume (kWh) per grid per 5-min interval
    - occupancy.csv: Active chargers per grid per 5-min interval
    - price.csv: Charging price (¥/kWh) per grid per 5-min interval
    - information.csv: Grid metadata (charger counts, location, CBD flag)
    - stations.csv: Station-level metadata
    - duration.csv: Charging duration per grid per 5-min interval
    - adj.csv: Grid adjacency matrix
    - distance.csv: Distance matrix between grids

    Returns dict of DataFrames for flexible access.
    """
    logger.info(f"Loading UrbanEV data from {data_dir}")

    data = {}
    key_files = {
        "time": "time.csv",
        "volume": "volume.csv",
        "occupancy": "occupancy.csv",
        "price": "price.csv",
        "duration": "duration.csv",
        "information": "information.csv",
        "stations": "stations.csv",
    }

    for key, filename in key_files.items():
        filepath = data_dir / filename
        if filepath.exists():
            data[key] = pd.read_csv(filepath)
            logger.info(f"  Loaded {filename}: {data[key].shape}")
        else:
            logger.warning(f"  Missing {filename}")

    return data


def build_urbanev_timeseries(data: dict) -> pd.DataFrame:
    """
    Build a long-format time-series DataFrame from UrbanEV grid-level data.

    Converts wide-format (columns = grid IDs) to long-format with:
    - timestamp, grid_id, volume_kwh, occupancy, price, duration_min
    """
    logger.info("Building UrbanEV time-series from grid data...")

    # 1. Build proper timestamps from time.csv
    time_df = data["time"]
    timestamps = pd.to_datetime(
        time_df[["year", "month", "day", "hour", "minute", "second"]]
    )
    n_timestamps = len(timestamps)
    logger.info(f"  Time range: {timestamps.min()} to {timestamps.max()} ({n_timestamps} intervals)")

    # 2. Melt volume data (wide → long)
    grid_cols = None
    dfs = []

    for key, col_name in [
        ("volume", "volume_kwh"),
        ("occupancy", "active_chargers"),
        ("price", "price_per_kwh"),
        ("duration", "avg_duration_min"),
    ]:
        if key not in data:
            continue

        wide_df = data[key].copy()

        # Get grid columns (all columns except 'timestamp' index)
        if grid_cols is None:
            grid_cols = [c for c in wide_df.columns if c != "timestamp"]
            logger.info(f"  Found {len(grid_cols)} grids")

        # Assign real timestamps
        wide_df["timestamp"] = timestamps.values[:len(wide_df)]

        # Melt to long format
        long = wide_df.melt(
            id_vars=["timestamp"],
            value_vars=grid_cols,
            var_name="grid_id",
            value_name=col_name,
        )
        long["grid_id"] = long["grid_id"].astype(str)
        dfs.append(long.set_index(["timestamp", "grid_id"]))

    # 3. Join all metrics
    if not dfs:
        logger.warning("No UrbanEV time-series data to build!")
        return pd.DataFrame()

    result = dfs[0]
    for df in dfs[1:]:
        result = result.join(df, how="outer")
    result = result.reset_index()

    # 4. Add station metadata from information.csv
    if "information" in data:
        info = data["information"].copy()
        info["grid_id"] = info["grid"].astype(str) if "grid" in info.columns else info["num"].astype(str)

        # Select useful metadata columns
        meta_cols = ["grid_id"]
        for c in ["count", "fast_count", "slow_count", "area", "lon", "la", "CBD", "dynamic_pricing"]:
            if c in info.columns:
                meta_cols.append(c)

        info_subset = info[meta_cols].drop_duplicates(subset=["grid_id"])

        # Rename for clarity
        info_subset = info_subset.rename(columns={
            "count": "total_chargers",
            "la": "latitude",
            "lon": "longitude",
            "CBD": "is_cbd",
        })

        result = result.merge(info_subset, on="grid_id", how="left")

    # 5. Add source label
    result["data_source"] = "UrbanEV"
    result["station_id"] = "UV_" + result["grid_id"]

    logger.info(f"  → UrbanEV time-series: {result.shape}")
    return result


def clean_urbanev_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean UrbanEV time-series data.

    Steps:
    1. Handle missing values (forward-fill within each grid)
    2. Remove fully-null grids
    3. Compute utilization rate from occupancy & total chargers
    """
    df = df.copy()
    logger.info("Cleaning UrbanEV data...")

    initial_count = len(df)

    # Forward-fill missing values within each grid
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    grid_numeric = [c for c in numeric_cols if c not in ["total_chargers", "fast_count", "slow_count"]]
    if "station_id" in df.columns and grid_numeric:
        df[grid_numeric] = df.groupby("station_id")[grid_numeric].transform(
            lambda x: x.ffill().fillna(0)
        )

    # Compute utilization rate if we have occupancy and total charger data
    if "active_chargers" in df.columns and "total_chargers" in df.columns:
        df["charger_utilization_rate"] = (
            df["active_chargers"] / df["total_chargers"].replace(0, np.nan)
        ).clip(0, 1).fillna(0)
        logger.info(f"  Utilization rate — mean: {df['charger_utilization_rate'].mean():.3f}")

    # Compute energy_kwh from volume if not present
    if "volume_kwh" in df.columns and "energy_kwh" not in df.columns:
        df["energy_kwh"] = df["volume_kwh"]

    logger.info(f"  → Clean UrbanEV data: {df.shape}")
    return df


# ══════════════════════════════════════════════
#  ACN → Hourly Aggregation
# ══════════════════════════════════════════════

def aggregate_acn_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate session-level ACN data to hourly station-level time series.

    This makes ACN data compatible with UrbanEV's time-series format.
    """
    df = df.copy()
    logger.info("Aggregating ACN sessions to hourly granularity...")

    if "session_start" not in df.columns:
        logger.warning("No session_start column — cannot aggregate")
        return df

    df["timestamp"] = pd.to_datetime(df["session_start"]).dt.floor("h")

    # Build aggregation dict dynamically based on available columns
    agg_dict = {
        "energy_kwh": [("session_count", "count"), ("total_energy_kwh", "sum"), ("avg_energy_kwh", "mean")],
    }

    agg_named = {}
    agg_named["session_count"] = ("energy_kwh", "count")
    agg_named["total_energy_kwh"] = ("energy_kwh", "sum")
    agg_named["avg_energy_kwh"] = ("energy_kwh", "mean")

    if "session_duration_min" in df.columns:
        agg_named["avg_duration_min"] = ("session_duration_min", "mean")
    if "charging_duration_min" in df.columns:
        agg_named["avg_charging_min"] = ("charging_duration_min", "mean")

    agg = df.groupby(["station_id", "timestamp"]).agg(**agg_named).reset_index()

    # Compute utilization proxy: charging_min / 60 min per charger
    if "avg_charging_min" in agg.columns and "session_count" in agg.columns:
        # Approximate: total charging minutes / total available minutes (1 charger × 60 min)
        agg["charger_utilization_rate"] = (
            (agg["avg_charging_min"] * agg["session_count"]) / 60
        ).clip(0, 1)

    # Rename for unified schema
    agg["energy_kwh"] = agg["total_energy_kwh"]
    agg["data_source"] = "ACN"

    logger.info(f"  → ACN hourly: {agg.shape}")
    return agg


# ══════════════════════════════════════════════
#  UrbanEV → Hourly Aggregation
# ══════════════════════════════════════════════

def aggregate_urbanev_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 5-min UrbanEV data to hourly granularity.
    """
    df = df.copy()
    logger.info("Aggregating UrbanEV data to hourly granularity...")

    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.floor("h")

    # Define aggregation — sum for energy, mean for rates/prices
    agg_dict = {}
    if "energy_kwh" in df.columns:
        agg_dict["energy_kwh"] = "sum"
    if "volume_kwh" in df.columns:
        agg_dict["volume_kwh"] = "sum"
    if "active_chargers" in df.columns:
        agg_dict["active_chargers"] = "mean"
    if "price_per_kwh" in df.columns:
        agg_dict["price_per_kwh"] = "mean"
    if "avg_duration_min" in df.columns:
        agg_dict["avg_duration_min"] = "mean"
    if "charger_utilization_rate" in df.columns:
        agg_dict["charger_utilization_rate"] = "mean"
    if "total_chargers" in df.columns:
        agg_dict["total_chargers"] = "first"
    if "is_cbd" in df.columns:
        agg_dict["is_cbd"] = "first"

    group_cols = ["station_id", "timestamp"]
    if "data_source" in df.columns:
        agg_dict["data_source"] = "first"

    hourly = df.groupby(group_cols).agg(agg_dict).reset_index()

    logger.info(f"  → UrbanEV hourly: {hourly.shape}")
    return hourly


# ══════════════════════════════════════════════
#  Dataset Merging
# ══════════════════════════════════════════════

def merge_datasets(
    acn_hourly: pd.DataFrame,
    urbanev_hourly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge ACN and UrbanEV datasets into a unified schema.

    Common columns: station_id, timestamp, energy_kwh, charger_utilization_rate, data_source
    """
    logger.info("Merging ACN and UrbanEV datasets...")

    # Identify common columns
    common = set(acn_hourly.columns) & set(urbanev_hourly.columns)
    logger.info(f"  Common columns: {sorted(common)}")

    # Concatenate
    merged = pd.concat([acn_hourly, urbanev_hourly], ignore_index=True)
    merged = merged.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    logger.info(f"  → Merged dataset: {merged.shape}")
    logger.info(f"  → ACN rows: {(merged['data_source']=='ACN').sum()}")
    logger.info(f"  → UrbanEV rows: {(merged['data_source']=='UrbanEV').sum()}")

    return merged


# ══════════════════════════════════════════════
#  Main Preprocessing Pipeline
# ══════════════════════════════════════════════

def run_preprocessing_pipeline(
    acn_path: Optional[Path] = None,
    urbanev_dir: Optional[Path] = None,
    output_path: Path = MERGED_SESSIONS_CSV,
) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline:
    1. Load ACN data → clean → aggregate to hourly
    2. Load UrbanEV data → build time-series → clean → aggregate to hourly
    3. Merge into unified dataset
    4. Save to CSV
    """
    logger.info("=" * 60)
    logger.info("  RUNNING PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    # ── Load & clean ACN ──
    acn_path = acn_path or ACN_RAW_CSV
    if acn_path.exists():
        acn_raw = load_acn_data(acn_path)
        acn_clean = clean_acn_data(acn_raw)
        acn_hourly = aggregate_acn_to_hourly(acn_clean)
        data_quality_report(acn_hourly, "ACN (hourly)")
    else:
        logger.warning(f"ACN data not found at {acn_path} — skipping")
        acn_hourly = pd.DataFrame()

    # ── Load & clean UrbanEV ──
    urbanev_dir = urbanev_dir or URBANEV_RAW_DIR
    if urbanev_dir.exists() and any(urbanev_dir.glob("*.csv")):
        urbanev_data = load_urbanev_data(urbanev_dir)
        urbanev_ts = build_urbanev_timeseries(urbanev_data)
        urbanev_clean = clean_urbanev_data(urbanev_ts)
        urbanev_hourly = aggregate_urbanev_to_hourly(urbanev_clean)
        data_quality_report(urbanev_hourly, "UrbanEV (hourly)")
    else:
        logger.warning(f"UrbanEV data not found at {urbanev_dir} — skipping")
        urbanev_hourly = pd.DataFrame()

    # ── Merge ──
    if not acn_hourly.empty and not urbanev_hourly.empty:
        merged = merge_datasets(acn_hourly, urbanev_hourly)
    elif not acn_hourly.empty:
        merged = acn_hourly
    elif not urbanev_hourly.empty:
        merged = urbanev_hourly
    else:
        logger.error("No data loaded! Please check data paths.")
        return pd.DataFrame()

    # ── Save ──
    save_csv(merged, output_path)
    data_quality_report(merged, "Merged Dataset")

    logger.info("=" * 60)
    logger.info("  PREPROCESSING COMPLETE")
    logger.info("=" * 60)

    return merged


# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = run_preprocessing_pipeline()
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head())
