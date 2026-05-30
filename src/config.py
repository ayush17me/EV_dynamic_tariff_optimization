"""
Central Configuration for EV Charging Tariff Optimization Project
All paths, constants, and hyperparameters in one place.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Create directories if they don't exist
for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Dataset Configuration
# ─────────────────────────────────────────────
# ACN-Data (Adaptive Charging Network)
ACN_RAW_JSON = DATA_RAW / "acn_sessions.json"
ACN_RAW_CSV = DATA_RAW / "acn_sessions.csv"

# UrbanEV (ST-EVCDP)
URBANEV_RAW_DIR = DATA_RAW / "urbanev"

# Merged / processed
MERGED_SESSIONS_CSV = DATA_PROCESSED / "merged_sessions.csv"
FEATURES_CSV = DATA_PROCESSED / "features_engineered.csv"

# ─────────────────────────────────────────────
# Pricing Constants
# ─────────────────────────────────────────────
BASELINE_TARIFF_INR = 15.0          # Fixed ₹15/kWh baseline
SURGE_THRESHOLD = 0.80              # Utilization > 80% → surge pricing
DISCOUNT_THRESHOLD = 0.30           # Utilization < 30% → discount pricing
SURGE_MULTIPLIER_RANGE = (1.3, 2.0) # Surge: 1.3x to 2.0x
DISCOUNT_MULTIPLIER_RANGE = (0.6, 0.8)  # Discount: 0.6x to 0.8x

# ─────────────────────────────────────────────
# Time Granularity
# ─────────────────────────────────────────────
TIME_GRANULARITY = "1H"  # Resample to 1-hour buckets
PEAK_HOURS = list(range(8, 12)) + list(range(17, 21))   # 8-12, 17-21
SHOULDER_HOURS = list(range(6, 8)) + list(range(12, 17)) + list(range(21, 23))
OFF_PEAK_HOURS = list(range(0, 6)) + [23]

# ─────────────────────────────────────────────
# Model Hyperparameters — Demand Prediction Agent
# ─────────────────────────────────────────────
DEMAND_MODEL_CONFIG = {
    "xgboost": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 50,
    },
    "lightgbm": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "lstm": {
        "sequence_length": 24,      # Look back 24 time steps
        "hidden_units": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "epochs": 100,
        "batch_size": 64,
        "learning_rate": 0.001,
    },
    "train_test_split": {
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
    },
}

# ─────────────────────────────────────────────
# Model Hyperparameters — Tariff Pricing Agent
# ─────────────────────────────────────────────
TARIFF_MODEL_CONFIG = {
    "pricing_tiers": {
        "off_peak": {"utilization_range": (0.0, 0.30), "multiplier": 0.7},
        "low":      {"utilization_range": (0.30, 0.50), "multiplier": 0.9},
        "normal":   {"utilization_range": (0.50, 0.70), "multiplier": 1.0},
        "high":     {"utilization_range": (0.70, 0.80), "multiplier": 1.2},
        "surge":    {"utilization_range": (0.80, 1.00), "multiplier": 1.5},
    },
    "elasticity_coefficient": -0.3,  # Demand drops 30% for 100% price increase
}

# ─────────────────────────────────────────────
# Monitoring & Learning Agent
# ─────────────────────────────────────────────
MONITORING_CONFIG = {
    "num_episodes": 10,             # Number of simulate → evaluate → retrain loops
    "retrain_threshold": 0.05,      # Retrain if RMSE degrades by > 5%
    "metrics_window": 24,           # Rolling window for metric tracking (hours)
}

# ─────────────────────────────────────────────
# Feature Lists
# ─────────────────────────────────────────────
TIME_FEATURES = [
    "hour_of_day", "day_of_week", "is_weekend",
    "is_peak_hour", "month", "day_of_month",
]

ENGINEERED_FEATURES = [
    "charger_utilization_rate",
    "revenue_per_session",
    "energy_cost_per_kwh",
    "queue_length_proxy",
    "occupancy_density",
]

LAG_FEATURES = [
    "demand_lag_1", "demand_lag_2",
    "demand_lag_6", "demand_lag_24",
]

ROLLING_FEATURES = [
    "utilization_rolling_1h",
    "utilization_rolling_6h",
    "utilization_rolling_24h",
    "demand_rolling_1h",
    "demand_rolling_6h",
    "demand_rolling_24h",
]

# All features combined for model input
ALL_FEATURES = TIME_FEATURES + ENGINEERED_FEATURES + LAG_FEATURES + ROLLING_FEATURES

# ─────────────────────────────────────────────
# Random Seed (reproducibility)
# ─────────────────────────────────────────────
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
PLOT_STYLE = "seaborn-v0_8-darkgrid"
FIG_SIZE = (14, 7)
DPI = 150
