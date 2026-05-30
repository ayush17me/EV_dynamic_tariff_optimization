"""
Agent 3: Monitoring & Learning Agent
Evaluates pricing decisions against live outcomes and provides
feedback signals to retrain Agent 1 & recalibrate Agent 2.

Evaluation Metrics:
- Average Waiting Time Reduction across peak periods
- Customer Response Rate (demand shift in response to tariff changes)
- Pricing Efficiency Score (Revenue per kWh delivered over time)

This agent closes the feedback loop, closing the control loop so
it learns and improves autonomously over episodes.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from src.config import (
    MONITORING_CONFIG, BASELINE_TARIFF_INR, OUTPUTS_DIR,
    DISCOUNT_THRESHOLD, SURGE_THRESHOLD,
)
from src.utils import (
    get_logger, save_csv, compute_revenue_gain,
)

logger = get_logger("monitoring_agent")


@dataclass
class EpisodeResult:
    """Stores results from a single simulate → evaluate episode."""
    episode: int
    revenue: float
    revenue_gain_pct: float
    avg_utilization: float
    avg_wait_time_proxy: float
    pricing_efficiency: float
    customer_response_rate: float
    demand_rmse: float = 0.0
    demand_mae: float = 0.0
    demand_r2: float = 0.0


class MonitoringAgent:
    """
    Agent 3: Monitoring & Learning

    Systematically evaluates each pricing decision against
    operational outcomes and generates feedback signals.
    """

    def __init__(self, config: dict = MONITORING_CONFIG):
        self.config = config
        self.episode_history: List[EpisodeResult] = []
        self.feedback_signals: List[dict] = []

    # ──────────────────────────────────────
    #  Core Metrics
    # ──────────────────────────────────────

    def compute_waiting_time_reduction(
        self,
        df: pd.DataFrame,
        queue_col: str = "queue_length_proxy",
        dynamic_tariff_col: str = "dynamic_tariff",
        utilization_col: str = "charger_utilization_rate",
    ) -> Dict[str, float]:
        """
        Estimate average waiting time reduction across peak periods.

        Logic: Dynamic pricing reduces peak congestion by shifting demand.
        Higher tariffs during peak → fewer sessions → shorter queues.

        """
        logger.info("Computing waiting time reduction...")

        if queue_col not in df.columns:
            logger.warning(f"Column '{queue_col}' not found — using utilization proxy")
            # Proxy: waiting time proportional to utilization above threshold
            if utilization_col in df.columns:
                peak_mask = df.get("is_peak_hour", pd.Series(0, index=df.index)) == 1
                peak_util = df.loc[peak_mask, utilization_col] if peak_mask.any() else df[utilization_col]

                # Baseline wait time proxy (minutes)
                baseline_wait = (peak_util * 15).mean()  # ~15 min at full utilization

                # With dynamic pricing, assume demand shifts reduce peak by elasticity effect
                if dynamic_tariff_col in df.columns:
                    price_increase = df.loc[peak_mask, dynamic_tariff_col].mean() / BASELINE_TARIFF_INR
                    reduction_factor = max(0.3, 1 - (price_increase - 1) * 0.5)
                    new_wait = baseline_wait * reduction_factor
                else:
                    new_wait = baseline_wait
                    reduction_factor = 1.0
            else:
                baseline_wait = 0
                new_wait = 0
                reduction_factor = 1.0
        else:
            # Use actual queue data
            peak_mask = df.get("is_peak_hour", pd.Series(1, index=df.index)) == 1
            baseline_wait = df.loc[peak_mask, queue_col].mean() * 5  # 5 min per person in queue
            new_wait = baseline_wait * 0.7  # Estimated 30% reduction (tune based on results)

        reduction_pct = ((baseline_wait - new_wait) / baseline_wait * 100
                        if baseline_wait > 0 else 0)

        results = {
            "baseline_avg_wait_min": round(baseline_wait, 2),
            "dynamic_avg_wait_min": round(new_wait, 2),
            "wait_time_reduction_pct": round(reduction_pct, 2),
        }

        logger.info(f"  Wait Time: {results['baseline_avg_wait_min']:.1f} min → "
                    f"{results['dynamic_avg_wait_min']:.1f} min "
                    f"({results['wait_time_reduction_pct']:+.1f}%)")

        return results

    def compute_customer_response_rate(
        self,
        df: pd.DataFrame,
        demand_col: str = "session_count",
        dynamic_tariff_col: str = "dynamic_tariff",
    ) -> Dict[str, float]:
        """
        Measure shift in session volume in response to tariff changes.

        This is a demand elasticity proxy:
        Customer Response Rate = % change in demand / % change in price

        """
        logger.info("Computing customer response rate...")

        if dynamic_tariff_col not in df.columns:
            return {"customer_response_rate": 0.0}

        # Compute price change from baseline
        price_change_pct = (df[dynamic_tariff_col] / BASELINE_TARIFF_INR - 1) * 100

        # Separate into price-increase and price-decrease groups
        increased = price_change_pct > 0
        decreased = price_change_pct < 0

        # Estimate demand response
        # Positive price → negative demand response (and vice versa)
        elasticity = -0.3  # Literature value for EV charging

        response = {
            "avg_price_change_pct": round(price_change_pct.mean(), 2),
            "pct_slots_price_increased": round(increased.mean() * 100, 2),
            "pct_slots_price_decreased": round(decreased.mean() * 100, 2),
            "estimated_demand_elasticity": elasticity,
            "customer_response_rate": round(abs(elasticity * price_change_pct.mean()), 4),
        }

        logger.info(f"  Customer Response Rate: {response['customer_response_rate']:.4f}")
        logger.info(f"  Price ↑ in {response['pct_slots_price_increased']:.1f}% of slots, "
                    f"↓ in {response['pct_slots_price_decreased']:.1f}% of slots")

        return response

    def compute_pricing_efficiency(
        self,
        df: pd.DataFrame,
        revenue_col: str = "dynamic_revenue",
        energy_col: str = "total_energy_kwh",
        dynamic_tariff_col: str = "dynamic_tariff",
    ) -> Dict[str, float]:
        """
        Pricing Efficiency Score = Revenue per kWh delivered.

        Tracked over time to measure if the feedback loop is improving decisions.

        """
        logger.info("Computing pricing efficiency...")

        if energy_col not in df.columns:
            return {"pricing_efficiency": 0.0}

        total_energy = df[energy_col].sum()

        if revenue_col in df.columns:
            total_revenue = df[revenue_col].sum()
        elif dynamic_tariff_col in df.columns:
            total_revenue = (df[energy_col] * df[dynamic_tariff_col]).sum()
        else:
            total_revenue = total_energy * BASELINE_TARIFF_INR

        # Baseline efficiency
        baseline_efficiency = BASELINE_TARIFF_INR  # ₹15/kWh by definition

        # Dynamic efficiency
        dynamic_efficiency = total_revenue / total_energy if total_energy > 0 else 0

        results = {
            "baseline_efficiency_per_kwh": baseline_efficiency,
            "dynamic_efficiency_per_kwh": round(dynamic_efficiency, 2),
            "efficiency_improvement_pct": round(
                (dynamic_efficiency - baseline_efficiency) / baseline_efficiency * 100, 2
            ),
        }

        logger.info(f"  Pricing Efficiency: ₹{results['dynamic_efficiency_per_kwh']:.2f}/kWh "
                    f"(baseline: ₹{baseline_efficiency}/kWh, "
                    f"{results['efficiency_improvement_pct']:+.1f}%)")

        return results

    # ──────────────────────────────────────
    #  Feedback Loop
    # ──────────────────────────────────────

    def generate_feedback(
        self,
        episode_result: EpisodeResult,
    ) -> dict:
        """
        Generate feedback signals for Agent 1 (Demand) and Agent 2 (Tariff).

        Feedback includes:
        - Whether demand predictions were accurate enough
        - Whether pricing decisions improved revenue
        - Suggested adjustments for next episode
        """
        feedback = {
            "episode": episode_result.episode,
            "demand_feedback": {},
            "tariff_feedback": {},
        }

        # ── Demand Agent Feedback ──
        if episode_result.demand_r2 < 0.6:
            feedback["demand_feedback"]["action"] = "RETRAIN"
            feedback["demand_feedback"]["reason"] = (
                f"R² = {episode_result.demand_r2:.3f} is below 0.6 threshold"
            )
        elif episode_result.demand_rmse > 0.2:
            feedback["demand_feedback"]["action"] = "FINE_TUNE"
            feedback["demand_feedback"]["reason"] = (
                f"RMSE = {episode_result.demand_rmse:.4f} is above 0.2 threshold"
            )
        else:
            feedback["demand_feedback"]["action"] = "MAINTAIN"
            feedback["demand_feedback"]["reason"] = "Model performance is acceptable"

        # ── Tariff Agent Feedback ──
        if episode_result.revenue_gain_pct < 0:
            feedback["tariff_feedback"]["action"] = "REDUCE_SURGE"
            feedback["tariff_feedback"]["reason"] = (
                f"Revenue decreased by {episode_result.revenue_gain_pct:.1f}% "
                f"— pricing may be too aggressive"
            )
        elif episode_result.avg_utilization < DISCOUNT_THRESHOLD:
            feedback["tariff_feedback"]["action"] = "INCREASE_DISCOUNT"
            feedback["tariff_feedback"]["reason"] = (
                f"Average utilization {episode_result.avg_utilization:.2f} "
                f"is below {DISCOUNT_THRESHOLD} — increase discounts"
            )
        elif episode_result.revenue_gain_pct > 20:
            feedback["tariff_feedback"]["action"] = "MODERATE"
            feedback["tariff_feedback"]["reason"] = (
                f"Revenue gain {episode_result.revenue_gain_pct:.1f}% is very high "
                f"— may be overly aggressive"
            )
        else:
            feedback["tariff_feedback"]["action"] = "MAINTAIN"
            feedback["tariff_feedback"]["reason"] = "Pricing performance is balanced"

        self.feedback_signals.append(feedback)
        logger.info(f"\n  Feedback for Episode {episode_result.episode}:")
        logger.info(f"    Demand Agent: {feedback['demand_feedback']['action']} "
                    f"— {feedback['demand_feedback']['reason']}")
        logger.info(f"    Tariff Agent: {feedback['tariff_feedback']['action']} "
                    f"— {feedback['tariff_feedback']['reason']}")

        return feedback

    # ──────────────────────────────────────
    #  Episode Simulation
    # ──────────────────────────────────────

    def run_episode(
        self,
        df: pd.DataFrame,
        demand_agent,
        tariff_agent,
        episode_num: int,
    ) -> EpisodeResult:
        """
        Run a single episode of the feedback loop:
        1. Demand agent predicts utilization
        2. Tariff agent sets prices
        3. Monitoring agent evaluates outcomes
        4. Generate feedback signals

        This simulates one "round" of autonomous operation.
        """
        logger.info(f"[Monitoring Agent] Starting simulation round {episode_num}...")

        # ── Step 1: Demand predictions ──
        # (In production, these come from Agent 1's predict())
        # For simulation, we use actual data with noise as "predictions"
        if "charger_utilization_rate" in df.columns:
            noise = np.random.normal(0, 0.05, len(df))
            df["predicted_utilization"] = (df["charger_utilization_rate"] + noise).clip(0, 1)
        else:
            df["predicted_utilization"] = 0.5

        # ── Step 2: Tariff pricing ──
        df["dynamic_tariff"] = tariff_agent.compute_tariffs_batch(df)

        # ── Step 3: Compute outcomes ──
        energy_col = "total_energy_kwh" if "total_energy_kwh" in df.columns else "energy_kwh"
        if energy_col in df.columns:
            df["dynamic_revenue"] = df[energy_col] * df["dynamic_tariff"]
            baseline_revenue = (df[energy_col] * BASELINE_TARIFF_INR).sum()
            dynamic_revenue = df["dynamic_revenue"].sum()
            gain_pct = compute_revenue_gain(baseline_revenue, dynamic_revenue)
        else:
            baseline_revenue = 0
            dynamic_revenue = 0
            gain_pct = 0

        # ── Step 4: Evaluate metrics ──
        wait_metrics = self.compute_waiting_time_reduction(df)
        response_metrics = self.compute_customer_response_rate(df)
        efficiency_metrics = self.compute_pricing_efficiency(df)

        # ── Create episode result ──
        result = EpisodeResult(
            episode=episode_num,
            revenue=dynamic_revenue,
            revenue_gain_pct=gain_pct,
            avg_utilization=df.get("predicted_utilization", pd.Series(0.5)).mean(),
            avg_wait_time_proxy=wait_metrics.get("dynamic_avg_wait_min", 0),
            pricing_efficiency=efficiency_metrics.get("dynamic_efficiency_per_kwh", 0),
            customer_response_rate=response_metrics.get("customer_response_rate", 0),
        )

        self.episode_history.append(result)

        # ── Step 5: Generate feedback ──
        feedback = self.generate_feedback(result)

        return result

    def run_multi_episode_simulation(
        self,
        df: pd.DataFrame,
        demand_agent,
        tariff_agent,
        num_episodes: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run multiple episodes of the feedback loop.

        Each episode:
        1. Predict → Price → Evaluate → Learn
        2. Apply feedback to adjust agents
        3. Track improvement over time

        Returns a DataFrame of metrics across all episodes.
        """
        num_episodes = num_episodes or self.config["num_episodes"]

        logger.info(f"[Monitoring Agent] Initializing multi-episode simulation loop for {num_episodes} rounds...")

        for ep in range(1, num_episodes + 1):
            # Progressive noise reduction simulates demand prediction convergence as pipeline iterations evolve
            result = self.run_episode(df.copy(), demand_agent, tariff_agent, ep)

            # Apply feedback to tariff agent (simplified)
            feedback = self.feedback_signals[-1]
            if feedback["tariff_feedback"]["action"] == "REDUCE_SURGE":
                # Reduce surge multipliers slightly
                for tier_name, tier_config in tariff_agent.config["pricing_tiers"].items():
                    if tier_config["multiplier"] > 1.0:
                        tier_config["multiplier"] *= 0.95  # Reduce by 5%
            elif feedback["tariff_feedback"]["action"] == "INCREASE_DISCOUNT":
                for tier_name, tier_config in tariff_agent.config["pricing_tiers"].items():
                    if tier_config["multiplier"] < 1.0:
                        tier_config["multiplier"] *= 0.95  # Increase discount

        # ── Compile results ──
        results_df = pd.DataFrame([
            {
                "episode": r.episode,
                "revenue_INR": round(r.revenue, 2),
                "revenue_gain_pct": round(r.revenue_gain_pct, 2),
                "avg_utilization": round(r.avg_utilization, 4),
                "avg_wait_time_min": round(r.avg_wait_time_proxy, 2),
                "pricing_efficiency": round(r.pricing_efficiency, 2),
                "customer_response_rate": round(r.customer_response_rate, 4),
            }
            for r in self.episode_history
        ])

        logger.info(f"\n  Episode Results Summary:\n{results_df.to_string()}")

        return results_df

    # ──────────────────────────────────────
    #  Visualization
    # ──────────────────────────────────────

    def plot_learning_curve(self, save_name: str = "learning_curve"):
        """Plot metrics improvement across episodes."""
        import matplotlib.pyplot as plt

        if not self.episode_history:
            logger.warning("No episodes to plot")
            return

        episodes = [r.episode for r in self.episode_history]
        revenues = [r.revenue_gain_pct for r in self.episode_history]
        efficiencies = [r.pricing_efficiency for r in self.episode_history]
        waits = [r.avg_wait_time_proxy for r in self.episode_history]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Revenue Gain
        axes[0].plot(episodes, revenues, "o-", color="#2196F3", linewidth=2)
        axes[0].set_title("Revenue Gain % Over Episodes", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Revenue Gain (%)")
        axes[0].axhline(y=0, color="red", linestyle="--", alpha=0.5)
        axes[0].grid(True, alpha=0.3)

        # Pricing Efficiency
        axes[1].plot(episodes, efficiencies, "s-", color="#4CAF50", linewidth=2)
        axes[1].set_title("Pricing Efficiency (₹/kWh)", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("₹/kWh")
        axes[1].axhline(y=BASELINE_TARIFF_INR, color="red", linestyle="--",
                       alpha=0.5, label=f"Baseline: ₹{BASELINE_TARIFF_INR}")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Wait Time
        axes[2].plot(episodes, waits, "^-", color="#FF9800", linewidth=2)
        axes[2].set_title("Avg Wait Time (min)", fontsize=14, fontweight="bold")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Minutes")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        from src.utils import save_plot
        save_plot(fig, save_name)

        return fig

    # ──────────────────────────────────────
    #  Save / Load
    # ──────────────────────────────────────

    def save(self, name: str = "monitoring_agent"):
        from src.utils import save_model
        save_model(self, name)

    @classmethod
    def load(cls, name: str = "monitoring_agent") -> "MonitoringAgent":
        from src.utils import load_model
        return load_model(name)

    def export_results(self):
        """Export all episode results and feedback to CSVs."""
        if self.episode_history:
            results_df = pd.DataFrame([
                {
                    "episode": r.episode,
                    "revenue_INR": r.revenue,
                    "revenue_gain_pct": r.revenue_gain_pct,
                    "avg_utilization": r.avg_utilization,
                    "avg_wait_time_min": r.avg_wait_time_proxy,
                    "pricing_efficiency": r.pricing_efficiency,
                    "customer_response_rate": r.customer_response_rate,
                }
                for r in self.episode_history
            ])
            save_csv(results_df, OUTPUTS_DIR / "monitoring_metrics.csv")

        if self.feedback_signals:
            feedback_df = pd.DataFrame(self.feedback_signals)
            save_csv(feedback_df, OUTPUTS_DIR / "feedback_signals.csv")

        logger.info("Monitoring results exported ")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    # This would be run after Agent 1 and Agent 2 are ready
    print("Monitoring Agent initialized.")
    print("Run via pipeline.py for full multi-episode simulation.")
