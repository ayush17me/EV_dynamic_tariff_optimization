"""
Agent 2: Tariff Pricing Agent
Translates demand forecasts into optimal dynamic tariffs.

Logic:
- Surge pricing when utilization > 80% (1.3x–2.0x multiplier)
- Discount pricing when utilization < 30% (0.6x–0.8x multiplier)
- Gradient-based dynamic pricing in the normal band (30%–80%)

Evaluation Metrics:
- Revenue Gain %: ((New Revenue - Old Revenue) / Old Revenue) × 100
- Charger Utilization Rate: Before vs. After dynamic pricing
- Off-Peak Uplift: Increase in sessions during low-demand periods
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional

from src.config import (
    BASELINE_TARIFF_INR, SURGE_THRESHOLD, DISCOUNT_THRESHOLD,
    SURGE_MULTIPLIER_RANGE, DISCOUNT_MULTIPLIER_RANGE,
    TARIFF_MODEL_CONFIG, OUTPUTS_DIR, FEATURES_CSV,
)
from src.utils import (
    get_logger, load_csv, save_csv, save_model, load_model,
    compute_revenue_gain,
)

logger = get_logger("tariff_agent")


class TariffPricingAgent:
    """
    Agent 2: Tariff Pricing

    Takes demand predictions and current conditions, then recommends
    optimal per-kWh tariff to maximize revenue while reducing congestion.
    """

    def __init__(self, config: dict = TARIFF_MODEL_CONFIG):
        self.config = config
        self.baseline_tariff = BASELINE_TARIFF_INR
        self.pricing_history = []
        self.optimized_lookup = self.precompute_optimized_schedule()

    def _optimize_single_tariff(
        self,
        baseline_util: float,
        hour: int,
        congestion_prob: float,
    ) -> float:
        """
        Numerically optimize the tariff for a single time slot using SciPy.

        Maximizes expected revenue while penalizing network congestion.
        """
        from scipy.optimize import minimize_scalar
        
        elasticity = self.config.get("elasticity_coefficient", -0.3)
        base_price = self.baseline_tariff
        
        # Estimate baseline energy demand (kWh) proportional to utilization
        base_energy = baseline_util * 50.0 
        
        # Min and Max tariff bounds
        min_tariff = base_price * DISCOUNT_MULTIPLIER_RANGE[0]  # ₹9
        max_tariff = base_price * SURGE_MULTIPLIER_RANGE[1]     # ₹30
        
        # Time-of-use base price multiplier
        from src.config import PEAK_HOURS, OFF_PEAK_HOURS
        if hour in PEAK_HOURS:
            tou_mult = 1.1
        elif hour in OFF_PEAK_HOURS:
            tou_mult = 0.9
        else:
            tou_mult = 1.0
            
        target_util = SURGE_THRESHOLD  # 0.8
        
        # Objective function: minimize negative penalized operational utility
        def objective(price: float) -> float:
            ref_price = base_price * tou_mult
            price_change_pct = (price / ref_price - 1.0)
            
            # Demand elasticity effect
            demand_multiplier = 1.0 + elasticity * price_change_pct
            demand_multiplier = max(0.2, min(1.8, demand_multiplier))
            
            est_util = baseline_util * demand_multiplier
            est_energy = base_energy * demand_multiplier
            
            # 1. Expected Revenue (₹)
            revenue = est_energy * price
            
            # 2. Operational Utility: balance utilization and customer acceptance
            # If utilization is low (< 30%), penalize high prices to attract users
            # If utilization is high (> 80%), penalize low prices to manage congestion
            # In the normal band, penalize deviations from baseline reference price
            util_cost = 0.0
            if baseline_util < DISCOUNT_THRESHOLD:
                target_discount_price = ref_price * 0.7  # ₹10.50 default discount target
                if price > target_discount_price:
                    util_cost = 10.0 * (price - target_discount_price) ** 2
            elif baseline_util > SURGE_THRESHOLD:
                target_surge_price = ref_price * 1.5     # ₹22.50 default surge target
                if price < target_surge_price:
                    util_cost = 10.0 * (target_surge_price - price) ** 2
            else:
                # Keep price stable near reference price in the normal band
                util_cost = 2.0 * (price - ref_price) ** 2
            
            # 3. Congestion Penalty
            congestion_penalty = 0.0
            if est_util > target_util:
                excess = est_util - target_util
                penalty_weight = 1000.0 * (1.0 + congestion_prob * 2.0)
                congestion_penalty = penalty_weight * (excess ** 2)
                
            penalized_revenue = revenue - util_cost - congestion_penalty
            return -penalized_revenue
            
        res = minimize_scalar(
            objective,
            bounds=(min_tariff, max_tariff),
            method="bounded",
            options={"xatol": 1e-3}
        )
        
        if res.success:
            return round(res.x, 2)
        else:
            return round(base_price * tou_mult, 2)

    def precompute_optimized_schedule(self) -> Dict[Tuple[int, int, int], float]:
        """
        Precomputes the optimal tariff grid across hours, utilization levels,
        and congestion probabilities using SciPy.
        """
        logger.info("Initializing SciPy Dynamic Pricing Optimizer...")
        logger.info("Precomputing optimal tariff lookup matrix (24 hours × 11 utils × 11 congestions)...")
        
        lookup = {}
        for hour in range(24):
            for u_idx in range(11):
                util = u_idx / 10.0
                for c_idx in range(11):
                    cong = c_idx / 10.0
                    optimal_price = self._optimize_single_tariff(util, hour, cong)
                    lookup[(hour, u_idx, c_idx)] = optimal_price
                    
        logger.info("  SciPy Optimizer precomputation complete ")
        return lookup

    # ──────────────────────────────────────
    #  Pricing Logic
    # ──────────────────────────────────────

    def compute_tariff(
        self,
        utilization: float,
        hour: int = 12,
        congestion_prob: float = 0.0,
    ) -> float:
        """
        Compute dynamic tariff for a single time slot using SciPy optimization.

        Args:
            utilization: Current/predicted charger utilization (0.0 to 1.0)
            hour: Hour of day (0-23) for time-of-use adjustments
            congestion_prob: Predicted congestion probability (0.0 to 1.0)

        Returns:
            Optimal tariff in ₹/kWh
        """
        # Clamp inputs to bounds
        utilization = max(0.0, min(1.0, utilization))
        congestion_prob = max(0.0, min(1.0, congestion_prob))
        hour = max(0, min(23, int(hour)))
        
        # Map utilization and congestion to discrete grid indices (0-10)
        u_idx = int(round(utilization * 10))
        c_idx = int(round(congestion_prob * 10))
        
        # Return precomputed optimized price
        return self.optimized_lookup.get((hour, u_idx, c_idx), self.baseline_tariff)

    def compute_tariffs_batch(
        self,
        df: pd.DataFrame,
        utilization_col: str = "predicted_utilization",
        hour_col: str = "hour_of_day",
        congestion_col: str = "congestion_probability",
    ) -> pd.Series:
        """
        Compute dynamic tariffs for an entire DataFrame using vectorized SciPy optimization lookup.

        Returns a Series of tariff values aligned with the DataFrame index.
        """
        logger.info(f"Computing batch optimized tariffs for {len(df)} records...")

        # Fallback columns
        if utilization_col not in df.columns:
            utilization_col = "charger_utilization_rate"
        if utilization_col not in df.columns:
            util = pd.Series(0.5, index=df.index)
        else:
            util = df[utilization_col].fillna(0.5)

        if hour_col not in df.columns:
            hours = pd.Series(12, index=df.index)
        else:
            hours = df[hour_col].fillna(12).astype(int)

        if congestion_col not in df.columns:
            cong = pd.Series(0.0, index=df.index)
        else:
            cong = df[congestion_col].fillna(0.0)

        # ── Vectorized discrete matching grid indices (0-10) ──
        u_idx = (util * 10).round().astype(int).clip(0, 10)
        c_idx = (cong * 10).round().astype(int).clip(0, 10)
        hours_idx = hours.clip(0, 23).astype(int)

        # Map combined keys via zip-comprehension
        keys = list(zip(hours_idx, u_idx, c_idx))
        tariff = pd.Series([self.optimized_lookup.get(k, self.baseline_tariff) for k in keys], index=df.index)
        tariff.name = "dynamic_tariff"

        logger.info(f"  Optimized tariff range: ₹{tariff.min():.2f} — ₹{tariff.max():.2f}")
        logger.info(f"  Optimized tariff mean:  ₹{tariff.mean():.2f}")
        logger.info(f"  Baseline:               ₹{self.baseline_tariff:.2f}")

        return tariff

    # ──────────────────────────────────────
    #  Revenue Simulation
    # ──────────────────────────────────────

    def simulate_revenue(
        self,
        df: pd.DataFrame,
        energy_col: str = "total_energy_kwh",
        dynamic_tariff_col: str = "dynamic_tariff",
    ) -> Dict[str, float]:
        """
        Compare revenue under fixed pricing vs. dynamic pricing.

        Returns dict with old/new revenue and key metrics.
        """
        logger.info("Simulating revenue comparison...")

        if energy_col not in df.columns:
            logger.error(f"Column '{energy_col}' not found")
            return {}

        # ── Old revenue (fixed baseline) ──
        old_revenue = (df[energy_col] * self.baseline_tariff).sum()

        # ── New revenue (dynamic pricing) ──
        if dynamic_tariff_col in df.columns:
            new_revenue = (df[energy_col] * df[dynamic_tariff_col]).sum()
        else:
            logger.warning("Dynamic tariff not computed yet — computing now")
            tariffs = self.compute_tariffs_batch(df)
            new_revenue = (df[energy_col] * tariffs).sum()

        # ── Revenue Gain % ──
        gain = compute_revenue_gain(old_revenue, new_revenue)

        results = {
            "baseline_revenue_INR": round(old_revenue, 2),
            "dynamic_revenue_INR": round(new_revenue, 2),
            "revenue_gain_pct": round(gain, 2),
            "baseline_tariff": self.baseline_tariff,
            "avg_dynamic_tariff": round(df.get(dynamic_tariff_col, pd.Series([self.baseline_tariff])).mean(), 2),
        }

        self.pricing_history.append(results)
        return results

    def compute_utilization_change(
        self,
        df: pd.DataFrame,
        utilization_col: str = "charger_utilization_rate",
        dynamic_tariff_col: str = "dynamic_tariff",
    ) -> Dict[str, float]:
        """
        Estimate how utilization changes with dynamic pricing.

        Uses demand elasticity: % change in demand per % change in price.
        """
        elasticity = self.config.get("elasticity_coefficient", -0.3)

        if utilization_col not in df.columns or dynamic_tariff_col not in df.columns:
            return {}

        # Price change ratio
        price_ratio = df[dynamic_tariff_col] / self.baseline_tariff

        # Estimated demand change (elasticity model)
        demand_change = 1 + elasticity * (price_ratio - 1)
        demand_change = demand_change.clip(0.3, 1.5)  # Bound changes

        # New estimated utilization
        new_utilization = (df[utilization_col] * demand_change).clip(0, 1)

        results = {
            "avg_util_before": round(df[utilization_col].mean(), 4),
            "avg_util_after": round(new_utilization.mean(), 4),
            "util_change_pct": round(
                (new_utilization.mean() - df[utilization_col].mean()) /
                df[utilization_col].mean() * 100, 2
            ) if df[utilization_col].mean() > 0 else 0,
        }

        logger.info(f"  Utilization: {results['avg_util_before']:.3f} → "
                    f"{results['avg_util_after']:.3f} ({results['util_change_pct']:+.1f}%)")

        return results

    def compute_off_peak_uplift(
        self,
        df: pd.DataFrame,
        utilization_col: str = "charger_utilization_rate",
        dynamic_tariff_col: str = "dynamic_tariff",
    ) -> Dict[str, float]:
        """
        Measure increase in sessions during low-demand periods
        (utilization < 30%) after discount pricing is applied.
        """
        if utilization_col not in df.columns:
            return {}

        # Identify off-peak slots (utilization < 30%)
        off_peak_mask = df[utilization_col] < DISCOUNT_THRESHOLD
        off_peak_count_before = off_peak_mask.sum()

        # With dynamic pricing, discount attracts more sessions
        elasticity = abs(self.config.get("elasticity_coefficient", -0.3))

        if dynamic_tariff_col in df.columns:
            off_peak_data = df[off_peak_mask]
            avg_discount = (off_peak_data[dynamic_tariff_col] / self.baseline_tariff).mean()
            price_drop_pct = (1 - avg_discount) * 100 if not np.isnan(avg_discount) else 0
            uplift = price_drop_pct * elasticity  # Estimated % increase in sessions
        else:
            uplift = 0
            price_drop_pct = 0

        results = {
            "off_peak_slots": int(off_peak_count_before),
            "off_peak_pct": round(off_peak_count_before / len(df) * 100, 2) if len(df) > 0 else 0,
            "avg_discount_pct": round(price_drop_pct, 2),
            "estimated_session_uplift_pct": round(uplift, 2),
        }

        logger.info(f"  Off-Peak Uplift: ~{results['estimated_session_uplift_pct']:.1f}% "
                    f"more sessions (avg {results['avg_discount_pct']:.1f}% discount)")

        return results

    # ──────────────────────────────────────
    #  Full Evaluation
    # ──────────────────────────────────────

    def evaluate(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Run full tariff agent evaluation.

        Returns combined metrics dictionary.
        """
        logger.info("[Tariff Agent] Running full agent evaluation...")

        results = {}
        results.update(self.simulate_revenue(df))
        results.update(self.compute_utilization_change(df))
        results.update(self.compute_off_peak_uplift(df))

        return results

    # ──────────────────────────────────────
    #  Pricing Schedule Export
    # ──────────────────────────────────────

    def generate_pricing_schedule(self) -> pd.DataFrame:
        """
        Generate a human-readable pricing schedule table
        showing tariff by utilization level and time period.
        """
        schedule = []
        for hour in range(24):
            for util in np.arange(0, 1.05, 0.1):
                tariff = self.compute_tariff(util, hour)
                schedule.append({
                    "hour": hour,
                    "utilization": round(util, 1),
                    "tariff_inr": tariff,
                })

        return pd.DataFrame(schedule)

    # ──────────────────────────────────────
    #  Save / Load
    # ──────────────────────────────────────

    def save(self, name: str = "tariff_agent"):
        save_model(self, name)
        logger.info(f"Tariff agent saved as '{name}'")

    @classmethod
    def load(cls, name: str = "tariff_agent") -> "TariffPricingAgent":
        return load_model(name)


# ══════════════════════════════════════════════
#  Convenience Runner
# ══════════════════════════════════════════════

def run_tariff_optimization(
    data_path: Path = FEATURES_CSV,
    demand_predictions: Optional[pd.DataFrame] = None,
) -> Tuple[TariffPricingAgent, Dict]:
    """
    Run the full tariff optimization pipeline.
    """
    logger.info("[Tariff Agent] Initializing Dynamic Tariff Optimization solver...")

    # Load data
    df = load_csv(data_path)

    # Initialize agent
    agent = TariffPricingAgent()

    # If demand predictions provided, merge them
    if demand_predictions is not None:
        df = pd.concat([df, demand_predictions], axis=1)

    # Compute dynamic tariffs using optimized lookup grid
    # Precomputes grid of 2904 combinations to maintain O(1) vectorized lookup speeds
    df["dynamic_tariff"] = agent.compute_tariffs_batch(df)

    # Evaluate
    results = agent.evaluate(df)

    # Export pricing schedule
    schedule = agent.generate_pricing_schedule()
    save_csv(schedule, OUTPUTS_DIR / "pricing_schedule.csv")

    # Save tariff metrics
    results_df = pd.DataFrame([results])
    save_csv(results_df, OUTPUTS_DIR / "tariff_metrics.csv")

    # Save agent
    agent.save()

    logger.info("[Tariff Agent] Dynamic pricing calculation complete.")

    return agent, results


# ─────────────────────────────────────────────
if __name__ == "__main__":
    agent, results = run_tariff_optimization()
    print("\nTariff Optimization Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
