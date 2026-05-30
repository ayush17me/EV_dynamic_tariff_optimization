# # Closed-Loop Multi-Episode Monitoring and Learning
#
# **Focus:** Close the loop: run multi-episode simulation loops and apply agent learning correction signals.
#
#
# **Metrics:**
# - Average Waiting Time Reduction
# - Customer Response Rate (demand elasticity proxy)
# - Pricing Efficiency Score (₹/kWh over time)

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
from src.demand_agent import DemandPredictionAgent
from src.tariff_agent import TariffPricingAgent
from src.monitoring_agent import MonitoringAgent

setup_plot_style()

# ## Loading Agents and Historical Features

# Load data
df = load_csv(FEATURES_CSV)

# Initialize agents (or load saved ones)
try:
    demand_agent = DemandPredictionAgent.load()
    print("✓ Loaded saved Demand Agent")
except:
    demand_agent = DemandPredictionAgent()
    print("⚠ Using fresh Demand Agent (no saved model found)")

tariff_agent = TariffPricingAgent()
monitor = MonitoringAgent()

print(f"\nDataset: {df.shape}")
print(f"Monitoring config: {monitor.config}")

# ## Executing Multi-Episode Closed-Loop Simulation
#
# Each episode:
# 1. **Demand Agent** predicts utilization
# 2. **Tariff Agent** sets dynamic prices
# 3. **Monitoring Agent** evaluates outcomes
# 4. Feedback signals adjust agents for the next episode

num_episodes = 10
episode_results = monitor.run_multi_episode_simulation(
    df=df.copy(),
    demand_agent=demand_agent,
    tariff_agent=tariff_agent,
    num_episodes=num_episodes,
)

print("\n" + "=" * 70)
print("  EPISODE RESULTS")
print("=" * 70)
print(episode_results.to_string(index=False))

# ## Multi-Episode Learning Curve Visualization

fig = monitor.plot_learning_curve(save_name="learning_curve")

# ## Feedback Correction Signals Analysis

print("\nFeedback Signals History:")
print("=" * 70)

for fb in monitor.feedback_signals:
    ep = fb["episode"]
    d_action = fb["demand_feedback"]["action"]
    d_reason = fb["demand_feedback"]["reason"]
    t_action = fb["tariff_feedback"]["action"]
    t_reason = fb["tariff_feedback"]["reason"]
    print(f"\n  Episode {ep}:")
    print(f"    Demand Agent → {d_action}: {d_reason}")
    print(f"    Tariff Agent → {t_action}: {t_reason}")

# ## Detailed Performance Dashboards Over Time

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Revenue over episodes
axes[0, 0].plot(episode_results["episode"], episode_results["revenue_INR"] / 1e6,
               "o-", color="#2196F3", linewidth=2, markersize=8)
axes[0, 0].set_title("Revenue Over Episodes", fontsize=13, fontweight="bold")
axes[0, 0].set_xlabel("Episode")
axes[0, 0].set_ylabel("Revenue (₹ millions)")
axes[0, 0].grid(alpha=0.3)

# Revenue Gain % over episodes
axes[0, 1].bar(episode_results["episode"], episode_results["revenue_gain_pct"],
              color="#4CAF50", alpha=0.7)
axes[0, 1].axhline(0, color="red", ls="--", alpha=0.5)
axes[0, 1].set_title("Revenue Gain % Over Episodes", fontsize=13, fontweight="bold")
axes[0, 1].set_xlabel("Episode")
axes[0, 1].set_ylabel("Revenue Gain (%)")
axes[0, 1].grid(axis="y", alpha=0.3)

# Pricing Efficiency
axes[1, 0].plot(episode_results["episode"], episode_results["pricing_efficiency"],
               "s-", color="#FF9800", linewidth=2, markersize=8)
axes[1, 0].axhline(BASELINE_TARIFF_INR, color="red", ls="--", lw=1.5,
                   label=f"Baseline ₹{BASELINE_TARIFF_INR}")
axes[1, 0].set_title("Pricing Efficiency (₹/kWh)", fontsize=13, fontweight="bold")
axes[1, 0].set_xlabel("Episode")
axes[1, 0].set_ylabel("₹/kWh")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Customer Response Rate
axes[1, 1].plot(episode_results["episode"], episode_results["customer_response_rate"],
               "^-", color="#9C27B0", linewidth=2, markersize=8)
axes[1, 1].set_title("Customer Response Rate", fontsize=13, fontweight="bold")
axes[1, 1].set_xlabel("Episode")
axes[1, 1].set_ylabel("Response Rate")
axes[1, 1].grid(alpha=0.3)

plt.suptitle("Monitoring Agent — Multi-Episode Learning Dashboard",
            fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(str(OUTPUTS_DIR / "monitoring_dashboard.png"), dpi=150, bbox_inches="tight")
plt.show()

# ## Exporting Simulation Outcomes

monitor.export_results()
print("\nAll monitoring results exported to outputs/ ✓")

# ## Operations and Learning Summary
#
# The Monitoring & Learning Agent demonstrates:
# 1. **Autonomous evaluation** of pricing decisions against outcomes
# 2. **Feedback generation** — actionable signals for Demand & Tariff agents
# 3. **Multi-episode learning** — metrics tracked across simulation rounds
# 4. **Pricing efficiency improvement** — ₹/kWh tracked over time shows learning

# Final summary statistics
print("\n" + "=" * 60)
print("  MONITORING AGENT — FINAL SUMMARY")
print("=" * 60)
print(f"  Episodes completed: {len(monitor.episode_history)}")
print(f"  Avg Revenue Gain: {episode_results['revenue_gain_pct'].mean():.2f}%")
print(f"  Avg Pricing Efficiency: ₹{episode_results['pricing_efficiency'].mean():.2f}/kWh")
print(f"  Avg Customer Response: {episode_results['customer_response_rate'].mean():.4f}")
print(f"\n  Feedback actions taken:")
for fb in monitor.feedback_signals:
    print(f"    Ep {fb['episode']}: Demand→{fb['demand_feedback']['action']}, "
          f"Tariff→{fb['tariff_feedback']['action']}")
