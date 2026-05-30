"""
Feature Engineering Module
Creates all economically meaningful features required by the project spec.

Features engineered:
- Charger Utilization Rate (Charging Time / Total Available Time)
- Revenue per Session (energy_kWh × tariff_rate)
- Energy Cost per kWh
- Queue Length Proxy (overlapping sessions)
- Occupancy Density (active/total chargers per station per time slot)
- Time features (hour, day, weekend, peak)
- Rolling averages (1h, 6h, 24h)
- Lag features (t-1, t-2, t-6, t-24)
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.config import (
    BASELINE_TARIFF_INR, TIME_GRANULARITY,
    FEATURES_CSV, MERGED_SESSIONS_CSV,
)
from src.utils import (
    get_logger, load_csv, save_csv,
    add_time_features, data_quality_report,
)

logger = get_logger("feature_engineering")


# ══════════════════════════════════════════════
#  Core Feature Functions
# ══════════════════════════════════════════════

def compute_charger_utilization_rate(
    df: pd.DataFrame,
    charging_time_col: str = "charging_duration_min",
    total_time_col: str = "duration_min",
) -> pd.Series:
    """
    Charger Utilization Rate = Charging Time / Total Available Time

    Values range from 0.0 (idle) to 1.0 (fully utilized).
    Capped at 1.0 for data quality.

    Author: Manasvi
    """
    if charging_time_col not in df.columns or total_time_col not in df.columns:
        logger.warning(f"Missing columns for utilization rate. Using duration_min as proxy.")
        # Fallback: use session count per station per hour as utilization proxy
        return pd.Series(np.nan, index=df.index)

    utilization = df[charging_time_col] / df[total_time_col].replace(0, np.nan)
    utilization = utilization.clip(0, 1)  # Cap at 1.0

    logger.info(f"  Utilization Rate — mean: {utilization.mean():.3f}, "
                f"median: {utilization.median():.3f}")
    return utilization


def compute_revenue_per_session(
    df: pd.DataFrame,
    energy_col: str = "energy_kwh",
    tariff: float = BASELINE_TARIFF_INR,
) -> pd.Series:
    """
    Revenue per Session = energy_kWh × tariff_rate

    Uses the fixed ₹15/kWh baseline by default.
    """
    if energy_col not in df.columns:
        logger.warning(f"Column '{energy_col}' not found. Revenue cannot be computed.")
        return pd.Series(np.nan, index=df.index)

    revenue = df[energy_col] * tariff
    logger.info(f"  Revenue/Session — mean: ₹{revenue.mean():.2f}, "
                f"total: ₹{revenue.sum():,.0f}")
    return revenue


def compute_energy_cost_per_kwh(
    df: pd.DataFrame,
    total_cost_col: str = "session_cost",
    energy_col: str = "energy_kwh",
) -> pd.Series:
    """
    Energy Cost per kWh = total_energy_cost / total_kWh_delivered
    """
    if total_cost_col not in df.columns:
        # If no cost column exists, assume baseline tariff
        logger.info("  No cost column found — using baseline tariff as energy cost")
        return pd.Series(BASELINE_TARIFF_INR, index=df.index)

    cost = df[total_cost_col] / df[energy_col].replace(0, np.nan)
    return cost


def compute_queue_length_proxy(
    df: pd.DataFrame,
    station_col: str = "station_id",
    start_col: str = "timestamp",
    duration_col: str = "duration_min",
) -> pd.Series:
    """
    Queue Length Proxy = count of overlapping sessions at the same station.

    For each session, count how many other sessions at the same station
    overlap in time. This approximates queue/wait pressure.

    Note: This is computationally expensive for large datasets.
    Consider using the hourly-aggregated version instead.
    """
    if station_col not in df.columns or start_col not in df.columns:
        logger.warning("Cannot compute queue length — missing station/time columns")
        return pd.Series(0, index=df.index)

    logger.info("  Computing queue length proxy (this may take a while)...")

    # Simplified approach: count sessions per station per hour
    df_temp = df.copy()
    df_temp["_hour"] = pd.to_datetime(df_temp[start_col]).dt.floor("h")

    queue = df_temp.groupby([station_col, "_hour"])[start_col].transform("count")
    queue = queue - 1  # Subtract self
    queue = queue.clip(lower=0)

    logger.info(f"  Queue Length — mean: {queue.mean():.2f}, max: {queue.max()}")
    return queue


def compute_occupancy_density(
    df: pd.DataFrame,
    station_col: str = "station_id",
    timestamp_col: str = "timestamp",
    total_chargers_per_station: int = 10,  # Default assumption
) -> pd.Series:
    """
    Occupancy Density = active_chargers / total_chargers per station per time slot.

    Assumption: Each station has `total_chargers_per_station` chargers.
    Active chargers = number of concurrent sessions.

    Author: Manasvi
    """
    if station_col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    df_temp = df.copy()
    df_temp["_hour"] = pd.to_datetime(df_temp[timestamp_col]).dt.floor("h")

    active = df_temp.groupby([station_col, "_hour"])[timestamp_col].transform("count")
    density = active / total_chargers_per_station
    density = density.clip(0, 1)

    logger.info(f"  Occupancy Density — mean: {density.mean():.3f}")
    return density


# ══════════════════════════════════════════════
#  Lag & Rolling Features
# ══════════════════════════════════════════════

def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "energy_kwh",
    lags: list = [1, 2, 6, 24],
    group_col: Optional[str] = "station_id",
) -> pd.DataFrame:
    """
    Add lag features: demand at t-1, t-2, t-6, t-24.

    If group_col is provided, lags are computed within each group.
    """
    df = df.copy()
    for lag in lags:
        col_name = f"demand_lag_{lag}"
        if group_col and group_col in df.columns:
            df[col_name] = df.groupby(group_col)[target_col].shift(lag)
        else:
            df[col_name] = df[target_col].shift(lag)
        logger.info(f"  Added lag feature: {col_name}")

    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_cols: list = ["energy_kwh"],
    windows: list = [1, 6, 24],
    group_col: Optional[str] = "station_id",
) -> pd.DataFrame:
    """
    Add rolling mean features for specified windows (in hours).

    Computes rolling averages for utilization and demand.
    """
    df = df.copy()
    for col in target_cols:
        if col not in df.columns:
            continue
        for window in windows:
            col_name = f"{col.split('_')[0]}_rolling_{window}h"
            if group_col and group_col in df.columns:
                df[col_name] = (
                    df.groupby(group_col)[col]
                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                )
            else:
                df[col_name] = df[col].rolling(window, min_periods=1).mean()
            logger.info(f"  Added rolling feature: {col_name}")

    return df


# ══════════════════════════════════════════════
#  Station-Level Aggregation
# ══════════════════════════════════════════════

def aggregate_to_hourly(
    df: pd.DataFrame,
    station_col: str = "station_id",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Aggregate session-level data to hourly station-level data.

    Output columns per station per hour:
    - session_count: number of sessions
    - total_energy_kwh: total energy delivered
    - avg_duration_min: average session duration
    - total_revenue: total revenue (baseline tariff)
    - utilization_rate: estimated utilization
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df["hour_bucket"] = df[timestamp_col].dt.floor("h")

    agg_dict = {}
    if "energy_kwh" in df.columns:
        agg_dict["energy_kwh"] = ["count", "sum", "mean"]
    if "duration_min" in df.columns:
        agg_dict["duration_min"] = "mean"

    if not agg_dict:
        logger.warning("No numeric columns to aggregate")
        return df

    grouped = df.groupby([station_col, "hour_bucket"]).agg(agg_dict)
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns]
    grouped = grouped.reset_index()

    # Rename for clarity
    rename_map = {
        "energy_kwh_count": "session_count",
        "energy_kwh_sum": "total_energy_kwh",
        "energy_kwh_mean": "avg_energy_kwh",
        "duration_min_mean": "avg_duration_min",
    }
    grouped = grouped.rename(columns={k: v for k, v in rename_map.items() if k in grouped.columns})

    logger.info(f"  → Hourly aggregated: {grouped.shape}")
    return grouped


# ══════════════════════════════════════════════
#  Main Feature Engineering Pipeline
# ══════════════════════════════════════════════

def run_feature_engineering(
    input_path=MERGED_SESSIONS_CSV,
    output_path=FEATURES_CSV,
) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline:
    1. Load merged data
    2. Add time features
    3. Compute utilization, revenue, queue, occupancy
    4. Add lag and rolling features
    5. Save engineered features
    """
    logger.info("=" * 60)
    logger.info("  RUNNING FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)

    # Load
    df = load_csv(input_path)

    # ── Time features ──
    if "timestamp" in df.columns:
        df = add_time_features(df, "timestamp")
    elif "session_start" in df.columns:
        df = add_time_features(df, "session_start")

    # ── Charger Utilization Rate ──
    # Only compute if not already present from preprocessing
    if "charger_utilization_rate" not in df.columns or df["charger_utilization_rate"].isna().all():
        df["charger_utilization_rate"] = compute_charger_utilization_rate(df)
    else:
        logger.info(f"  Utilization Rate — already exists, mean: {df['charger_utilization_rate'].mean():.3f}")

    # ── Revenue per Session ──
    df["revenue_per_session"] = compute_revenue_per_session(df)

    # ── Energy Cost per kWh ──
    df["energy_cost_per_kwh"] = compute_energy_cost_per_kwh(df)

    # ── Queue Length Proxy ──
    df["queue_length_proxy"] = compute_queue_length_proxy(df)

    # ── Occupancy Density ──
    # Use active_chargers / total_chargers if available (from UrbanEV)
    if "active_chargers" in df.columns and "total_chargers" in df.columns:
        df["occupancy_density"] = (
            df["active_chargers"] / df["total_chargers"].replace(0, np.nan)
        ).clip(0, 1).fillna(0)
        logger.info(f"  Occupancy Density (from active_chargers) — mean: {df['occupancy_density'].mean():.3f}")
    else:
        df["occupancy_density"] = compute_occupancy_density(df)

    # ── Lag Features ──
    target = "energy_kwh" if "energy_kwh" in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    df = add_lag_features(df, target_col=target)

    # ── Rolling Features ──
    rolling_targets = [c for c in ["energy_kwh", "charger_utilization_rate"] if c in df.columns]
    df = add_rolling_features(df, target_cols=rolling_targets)

    # ── Save ──
    save_csv(df, output_path)
    data_quality_report(df, "Engineered Features")

    logger.info("=" * 60)
    logger.info("  FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 60)

    return df


# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = run_feature_engineering()
    print(f"\nEngineered dataset shape: {df.shape}")
    print(f"Feature columns: {list(df.columns)}")
