# # Data Preprocessing and Dataset Alignment
#
# **Focus:** Loading raw datasets, cleaning session records, aligning timestamps, and merging into a unified hourly schema.
#
# **Datasets:**
# - **ACN-Data** (Caltech): 16,304 EV charging sessions with timestamps, energy, station IDs
# - **UrbanEV** (ST-EVCDP): 247 charging grids × 8,640 time intervals (5-min resolution)

import sys
sys.path.insert(0, "..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import *
from src.utils import *
from src.preprocessing import *

setup_plot_style()

# ## Raw Data Ingestion

# ### 1.1 ACN-Data
# Source: https://ev.caltech.edu/dataset.html
# 30,000+ sessions from Caltech & JPL charging stations

acn_raw = load_acn_data()
print(f"ACN Raw shape: {acn_raw.shape}")
print(f"\nColumn types:")
print(acn_raw.dtypes)
acn_raw.head()

# Quick stats on ACN data
print(f"\nACN Data Summary:")
print(f"  Sessions: {len(acn_raw):,}")
print(f"  Stations: {acn_raw['stationID'].nunique() if 'stationID' in acn_raw.columns else 'N/A'}")
print(f"  Date range: {acn_raw['connectionTime'].min() if 'connectionTime' in acn_raw.columns else 'N/A'} to {acn_raw['connectionTime'].max() if 'connectionTime' in acn_raw.columns else 'N/A'}")
if 'kWhDelivered' in acn_raw.columns:
    print(f"  Energy range: {acn_raw['kWhDelivered'].min():.2f} - {acn_raw['kWhDelivered'].max():.2f} kWh")
    print(f"  Missing kWh: {acn_raw['kWhDelivered'].isna().sum()}")

# ### 1.2 UrbanEV Data (ST-EVCDP)
# Source: https://github.com/IntelligentSystemsLab/ST-EVCDP
# 247 charging grids in Shenzhen, China — 30 days at 5-minute intervals

urbanev_data = load_urbanev_data()

print("\nUrbanEV Components:")
for key, df in urbanev_data.items():
    print(f"  {key}: {df.shape}")

# Inspect station/grid information
info = urbanev_data.get("information", pd.DataFrame())
if not info.empty:
    print(f"\nGrid Information:")
    print(f"  Total grids: {len(info)}")
    print(f"  Total chargers: {info['count'].sum() if 'count' in info.columns else 'N/A'}")
    print(f"  Fast chargers: {info['fast_count'].sum() if 'fast_count' in info.columns else 'N/A'}")
    print(f"  CBD grids: {info['CBD'].sum() if 'CBD' in info.columns else 'N/A'}")
    print(f"  Dynamic pricing grids: {info['dynamic_pricing'].sum() if 'dynamic_pricing' in info.columns else 'N/A'}")
    info.head()

# ## Initial Data Quality Diagnostics

data_quality_report(acn_raw, "ACN Raw Data")

# ## Data Cleaning and Filtering

# ### 3.1 Clean ACN Data
# - Parse timestamps (connectionTime, disconnectTime, doneChargingTime)
# - Compute session & charging duration
# - Drop sessions with missing energy or invalid durations
# - Standardize column names

acn_clean = clean_acn_data(acn_raw)
print(f"\nACN Clean shape: {acn_clean.shape}")
print(f"Columns: {list(acn_clean.columns)}")
acn_clean.head()

data_quality_report(acn_clean, "ACN Cleaned")

# ### 3.2 Build UrbanEV Time-Series
# - Convert wide-format grids to long-format time-series
# - Join volume, occupancy, price, and duration data
# - Attach station metadata (charger counts, location)

urbanev_ts = build_urbanev_timeseries(urbanev_data)
print(f"\nUrbanEV Time-Series shape: {urbanev_ts.shape}")
print(f"Columns: {list(urbanev_ts.columns)}")
urbanev_ts.head()

urbanev_clean = clean_urbanev_data(urbanev_ts)
data_quality_report(urbanev_clean, "UrbanEV Cleaned")

# ## Chronological Aggregation (Hourly Buckets)

# ACN: session-level → hourly station-level
# Note: Hourly buckets balanced granularity and sequence density for downstream models
acn_hourly = aggregate_acn_to_hourly(acn_clean)
print(f"ACN Hourly: {acn_hourly.shape}")
acn_hourly.head()

# UrbanEV: 5-min intervals → hourly
urbanev_hourly = aggregate_urbanev_to_hourly(urbanev_clean)
print(f"UrbanEV Hourly: {urbanev_hourly.shape}")
urbanev_hourly.head()

# ## Station-Level Dataset Merging

merged = merge_datasets(acn_hourly, urbanev_hourly)
save_csv(merged, MERGED_SESSIONS_CSV)
print(f"\nMerged Dataset: {merged.shape}")
print(f"  ACN rows: {(merged['data_source'] == 'ACN').sum():,}")
print(f"  UrbanEV rows: {(merged['data_source'] == 'UrbanEV').sum():,}")
print(f"  Stations: {merged['station_id'].nunique()}")
merged.head(10)

data_quality_report(merged, "Merged Dataset")

# ## Preprocessing Validation and Diagnostics

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Energy distribution by source
for src, color in [("ACN", "#3498db"), ("UrbanEV", "#e74c3c")]:
    subset = merged[merged["data_source"] == src]["energy_kwh"].dropna()
    if not subset.empty:
        axes[0, 0].hist(subset.clip(0, subset.quantile(0.99)), bins=50, alpha=0.6, label=src, color=color)
axes[0, 0].set_title("Energy Distribution (kWh)")
axes[0, 0].set_xlabel("Energy (kWh)")
axes[0, 0].legend()

# Utilization distribution
util = merged["charger_utilization_rate"].dropna()
if not util.empty:
    axes[0, 1].hist(util, bins=50, color="#2ecc71", alpha=0.7)
    axes[0, 1].axvline(0.3, color="blue", ls="--", label="Discount threshold (30%)")
    axes[0, 1].axvline(0.8, color="red", ls="--", label="Surge threshold (80%)")
    axes[0, 1].set_title("Charger Utilization Rate")
    axes[0, 1].legend()

# Hourly session count (ACN only)
acn_subset = merged[merged["data_source"] == "ACN"]
if "timestamp" in acn_subset.columns:
    ts = pd.to_datetime(acn_subset["timestamp"])
    hourly_counts = ts.dt.hour.value_counts().sort_index()
    axes[1, 0].bar(hourly_counts.index, hourly_counts.values, color="#9b59b6", alpha=0.7)
    axes[1, 0].set_title("ACN Sessions by Hour of Day")
    axes[1, 0].set_xlabel("Hour")

# Records by source
source_counts = merged["data_source"].value_counts()
axes[1, 1].bar(source_counts.index, source_counts.values, color=["#3498db", "#e74c3c"])
axes[1, 1].set_title("Records by Data Source")

plt.tight_layout()
plt.savefig(str(OUTPUTS_DIR / "preprocessing_validation.png"), dpi=150, bbox_inches="tight")
plt.show()

# ## Engineering Decisions and Assumptions Log
#
# | Decision | Rationale |
# |----------|-----------|
# | Dropped sessions with missing kWhDelivered | Cannot compute revenue without energy data |
# | Dropped sessions with duration ≤ 0 | Invalid data (negative time) |
# | Forward-filled UrbanEV missing intervals | Assume charging state persists until next observation |
# | Aligned to 1-hour granularity | Balances granularity and data density for both datasets |
# | Used grid IDs as station_id for UrbanEV | Each grid represents a cluster of charging piles |
# | Prefixed UrbanEV station IDs with "UV_" | Avoids ID collision with ACN station IDs |
# | UrbanEV utilization = active_chargers / total_chargers | Direct measurement from occupancy data |
# | ACN utilization ≈ charging_min × sessions / 60 | Proxy since we don't have real-time occupancy |
