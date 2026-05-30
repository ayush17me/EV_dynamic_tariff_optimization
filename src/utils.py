"""
Shared Utilities
Common helper functions used across all modules and agents.
"""

import os
import logging
import pickle
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    MODELS_DIR, OUTPUTS_DIR, PLOT_STYLE, FIG_SIZE, DPI,
    PEAK_HOURS, SHOULDER_HOURS, OFF_PEAK_HOURS, RANDOM_SEED,
)

# ─────────────────────────────────────────────
# Logger Setup
# ─────────────────────────────────────────────
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a formatted logger for any module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = get_logger("utils")


# ─────────────────────────────────────────────
# Data I/O
# ─────────────────────────────────────────────
def load_csv(filepath: Path, parse_dates: list = None, **kwargs) -> pd.DataFrame:
    """Load a CSV with logging."""
    logger.info(f"Loading CSV: {filepath}")
    df = pd.read_csv(filepath, parse_dates=parse_dates, **kwargs)
    logger.info(f"  → Shape: {df.shape} | Columns: {list(df.columns)}")
    return df


def save_csv(df: pd.DataFrame, filepath: Path, index: bool = False):
    """Save a DataFrame to CSV with logging."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=index)
    logger.info(f"Saved CSV: {filepath} ({len(df)} rows)")


def save_model(model: Any, name: str):
    """Pickle a trained model to the models directory."""
    filepath = MODELS_DIR / f"{name}.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved model: {filepath}")


def load_model(name: str) -> Any:
    """Load a pickled model from the models directory."""
    filepath = MODELS_DIR / f"{name}.pkl"
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Loaded model: {filepath}")
    return model


# ─────────────────────────────────────────────
# Time Helpers
# ─────────────────────────────────────────────
def classify_time_period(hour: int) -> str:
    """Classify an hour into peak, shoulder, or off-peak."""
    if hour in PEAK_HOURS:
        return "peak"
    elif hour in SHOULDER_HOURS:
        return "shoulder"
    else:
        return "off_peak"


def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Add time-based features from a timestamp column.

    Adds: hour_of_day, day_of_week, is_weekend, is_peak_hour, month, day_of_month
    """
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])

    df["hour_of_day"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek          # 0=Mon, 6=Sun
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    df["is_peak_hour"] = ts.dt.hour.isin(PEAK_HOURS).astype(int)
    df["month"] = ts.dt.month
    df["day_of_month"] = ts.dt.day
    df["time_period"] = ts.dt.hour.apply(classify_time_period)

    return df


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, R² for demand prediction evaluation."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
    logger.info(f"  Regression Metrics → RMSE: {metrics['RMSE']:.4f} | "
                f"MAE: {metrics['MAE']:.4f} | R²: {metrics['R2']:.4f}")
    return metrics


def compute_revenue_gain(old_revenue: float, new_revenue: float) -> float:
    """Revenue Gain % = ((New - Old) / Old) × 100"""
    if old_revenue == 0:
        return float("inf")
    gain = ((new_revenue - old_revenue) / old_revenue) * 100
    logger.info(f"  Revenue Gain: {gain:.2f}% (₹{old_revenue:,.0f} → ₹{new_revenue:,.0f})")
    return gain


# ─────────────────────────────────────────────
# Visualization Helpers
# ─────────────────────────────────────────────
def setup_plot_style():
    """Apply consistent plot styling."""
    try:
        plt.style.use(PLOT_STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")


def save_plot(fig: plt.Figure, name: str, subdir: str = "plots"):
    """Save a matplotlib figure to outputs/plots/."""
    plot_dir = OUTPUTS_DIR / subdir
    plot_dir.mkdir(parents=True, exist_ok=True)
    filepath = plot_dir / f"{name}.png"
    fig.savefig(filepath, dpi=DPI, bbox_inches="tight")
    logger.info(f"Saved plot: {filepath}")
    plt.close(fig)


def plot_time_series(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str = "Time",
    ylabel: str = "Value",
    hue_col: Optional[str] = None,
    save_name: Optional[str] = None,
):
    """Quick time-series line plot."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    if hue_col:
        for group_name, group_df in df.groupby(hue_col):
            ax.plot(group_df[x_col], group_df[y_col], label=group_name, alpha=0.8)
        ax.legend(title=hue_col)
    else:
        ax.plot(df[x_col], df[y_col], alpha=0.8)

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    if save_name:
        save_plot(fig, save_name)
    else:
        plt.show()

    return fig


def plot_heatmap(
    data: pd.DataFrame,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    save_name: Optional[str] = None,
    cmap: str = "YlOrRd",
    fmt: str = ".2f",
):
    """Quick heatmap for correlation or pivot tables."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap, ax=ax, linewidths=0.5)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    plt.tight_layout()

    if save_name:
        save_plot(fig, save_name)
    else:
        plt.show()

    return fig


# ─────────────────────────────────────────────
# Data Validation
# ─────────────────────────────────────────────
def data_quality_report(df: pd.DataFrame, name: str = "Dataset") -> pd.DataFrame:
    """Generate a quick data quality summary."""
    report = pd.DataFrame({
        "dtype": df.dtypes,
        "non_null": df.count(),
        "null_count": df.isnull().sum(),
        "null_pct": (df.isnull().sum() / len(df) * 100).round(2),
        "unique": df.nunique(),
        "sample": df.iloc[0] if len(df) > 0 else None,
    })
    logger.info(f"\n{'='*60}\n  Data Quality Report: {name}\n  Shape: {df.shape}\n{'='*60}")
    logger.info(f"\n{report.to_string()}")
    return report


def set_seed(seed: int = RANDOM_SEED):
    """Set random seed for reproducibility across numpy, random, and frameworks."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    logger.info(f"Random seed set to {seed}")
