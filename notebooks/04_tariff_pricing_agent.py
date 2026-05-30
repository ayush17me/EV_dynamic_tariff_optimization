# # Constrained Dynamic Pricing Optimization
#
# **Focus:** Translating hourly demand predictions into revenue-maximizing dynamic tariffs using constrained optimization.
#
# **Logic:**
# - Constrained mathematical optimization using `scipy.optimize.minimize_scalar`.
# - Maximizes operator utility: expected revenue minus operational price deviations and congestion penalties.
# - Precomputes multi-dimensional optimal schedules across a grid for instant vectorized lookup.
#
# **Metrics:** Revenue Gain %, Utilization Rate Change, Off-Peak Uplift

import sys
sys.path.insert(0, "..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from src.config import *
from src.utils import *
from src.tariff_agent import TariffPricingAgent, run_tariff_optimization

setup_plot_style()

# ## Initializing SciPy Dynamic Pricing Agent

agent = TariffPricingAgent()

print("SciPy Optimizer Configuration:")
print(f"  Baseline fixed tariff:   ₹{agent.baseline_tariff:.2f}/kWh")
print(f"  Dynamic price range:    ₹9.00 — ₹30.00/kWh")
print(f"  Elasticity coefficient:  {agent.config['elasticity_coefficient']}")
print(f"  Precomputed lookup size: {len(agent.optimized_lookup)} grid cells (Hour × Util × Congestion)")

# ## Generating Precomputed Lookup Grid

schedule = agent.generate_pricing_schedule()

# Pivot for heatmap visualization
pivot = schedule.pivot_table(
    values="tariff_inr", index="utilization", columns="hour", aggfunc="mean"
)

fig, ax = plt.subplots(figsize=(18, 6))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn_r", ax=ax,
           linewidths=0.5, cbar_kws={"label": "₹/kWh"})
ax.set_title("Dynamic Tariff Schedule (₹/kWh by Utilization × Hour)",
            fontsize=14, fontweight="bold")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Utilization Rate")
plt.tight_layout()
plt.savefig(str(OUTPUTS_DIR / "tariff_schedule_heatmap.png"), dpi=150, bbox_inches="tight")
plt.show()

# ## Dynamic Surcharge Calculations over Features Dataset

df = load_csv(FEATURES_CSV)

# Use actual utilization as predicted (in production this comes from Agent 1)
if "charger_utilization_rate" in df.columns:
    df["predicted_utilization"] = df["charger_utilization_rate"]
else:
    df["predicted_utilization"] = 0.5

# Compute dynamic tariffs using optimized lookup grid
# Precomputes grid of 2904 combinations to maintain O(1) vectorized lookup speeds
df["dynamic_tariff"] = agent.compute_tariffs_batch(df)

# Show tariff distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].hist(df["dynamic_tariff"], bins=50, color="#e74c3c", alpha=0.7, edgecolor="white")
axes[0].axvline(BASELINE_TARIFF_INR, color="blue", ls="--", lw=2, label=f"Baseline ₹{BASELINE_TARIFF_INR}")
axes[0].set_title("Dynamic Tariff Distribution", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Tariff (₹/kWh)")
axes[0].set_ylabel("Frequency")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

# Tariff by hour
hourly_tariff = df.groupby("hour_of_day")["dynamic_tariff"].mean()
axes[1].bar(hourly_tariff.index, hourly_tariff.values, color="#3498db", alpha=0.7)
axes[1].axhline(BASELINE_TARIFF_INR, color="red", ls="--", lw=2, label=f"Baseline ₹{BASELINE_TARIFF_INR}")
axes[1].set_title("Average Dynamic Tariff by Hour", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Hour of Day")
axes[1].set_ylabel("₹/kWh")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(str(OUTPUTS_DIR / "tariff_distribution.png"), dpi=150, bbox_inches="tight")
plt.show()

# ## Revenue Simulation and Impact Evaluation

energy_col = "energy_kwh"
revenue_results = agent.simulate_revenue(df, energy_col=energy_col)

print("\n" + "=" * 50)
print("  REVENUE SIMULATION RESULTS")
print("=" * 50)
for k, v in revenue_results.items():
    if "revenue" in k.lower() or "tariff" in k.lower():
        print(f"  {k}: ₹{v:,.2f}" if isinstance(v, float) and v > 100 else f"  {k}: {v}")
    else:
        print(f"  {k}: {v}")

# ## Demand Response and Utilization Impact (Elasticity Model)

util_results = agent.compute_utilization_change(df)
print("\nUtilization Change:")
for k, v in util_results.items():
    print(f"  {k}: {v}")

# ## Off-Peak Discount Session Uplift Analysis

uplift_results = agent.compute_off_peak_uplift(df)
print("\nOff-Peak Uplift:")
for k, v in uplift_results.items():
    print(f"  {k}: {v}")

# ## Operational Performance Metrics Summary

all_results = agent.evaluate(df)

# Create summary table
summary = pd.DataFrame([all_results])
print("\n" + "=" * 60)
print("  TARIFF AGENT — COMPLETE EVALUATION")
print("=" * 60)
print(summary.T.to_string(header=False))

# Save
save_csv(summary, OUTPUTS_DIR / "tariff_metrics.csv")

# ## Financial Outcomes Comparison Visualization

fig, ax = plt.subplots(figsize=(10, 6))

categories = ["Baseline\n(Fixed ₹15/kWh)", "Dynamic\nPricing"]
revenues = [
    all_results.get("baseline_revenue_INR", 0),
    all_results.get("dynamic_revenue_INR", 0),
]
colors = ["#95a5a6", "#2ecc71"]

bars = ax.bar(categories, revenues, color=colors, width=0.5, edgecolor="white", linewidth=2)
gain_pct = all_results.get("revenue_gain_pct", 0)
ax.set_title(f"Revenue Comparison — {gain_pct:+.2f}% Gain",
            fontsize=14, fontweight="bold")
ax.set_ylabel("Revenue (₹)")
ax.grid(axis="y", alpha=0.3)

for bar, val in zip(bars, revenues):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(revenues)*0.02,
           f"₹{val:,.0f}", ha="center", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig(str(OUTPUTS_DIR / "tariff_revenue_comparison.png"), dpi=150, bbox_inches="tight")
plt.show()

agent.save()
print("Tariff Pricing Agent saved ✓")
