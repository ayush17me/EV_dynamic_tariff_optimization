"""
Pipeline Orchestrator
Orchestrates the full agentic pipeline:
1. Preprocessing → 2. Feature Engineering → 3. Demand Agent →
4. Tariff Agent → 5. Monitoring Agent (multi-episode feedback loop)

Run this to execute the entire pipeline end-to-end.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FEATURES_CSV, OUTPUTS_DIR
from src.utils import get_logger, set_seed
from src.preprocessing import run_preprocessing_pipeline
from src.feature_engineering import run_feature_engineering
from src.demand_agent import DemandPredictionAgent, run_demand_prediction
from src.tariff_agent import TariffPricingAgent, run_tariff_optimization
from src.monitoring_agent import MonitoringAgent

logger = get_logger("pipeline")


def run_full_pipeline(
    skip_preprocessing: bool = False,
    skip_feature_engineering: bool = False,
    train_models: list = ["xgboost", "lightgbm", "random_forest"],
    num_episodes: int = 10,
):
    """
    Run the complete agentic AI pipeline.

    Args:
        skip_preprocessing: Skip if data is already preprocessed
        skip_feature_engineering: Skip if features are already engineered
        train_models: List of demand models to train ["xgboost", "random_forest", "lstm"]
        num_episodes: Number of feedback loop episodes
    """
    logger.info("Initializing end-to-end EV Charging Dynamic Tariff Optimization Pipeline...")

    set_seed()

    # Preprocessing
    if not skip_preprocessing:
        logger.info("[Pipeline] Starting Phase 1: Data Preprocessing...")
        merged_df = run_preprocessing_pipeline()
    else:
        logger.info("[Pipeline] Skipping preprocessing (reusing existing merged sessions data)")

    # Feature Engineering
    if not skip_feature_engineering:
        logger.info("[Pipeline] Starting Phase 2: Feature Engineering...")
        features_df = run_feature_engineering()
    else:
        logger.info("[Pipeline] Skipping feature engineering (reusing existing engineered features)")

    # Demand Agent
    logger.info("[Pipeline] Starting Phase 3: Demand Prediction Agent...")
    demand_agent, demand_results = run_demand_prediction(
        train_models=train_models,
    )
    best_m = demand_results.sort_values('RMSE').iloc[0]['model']
    logger.info(f"[Pipeline] Demand Agent execution complete. Best model: {best_m}")

    # Tariff Agent
    logger.info("[Pipeline] Starting Phase 4: Tariff Pricing Agent...")
    tariff_agent, tariff_results = run_tariff_optimization()
    logger.info(f"[Pipeline] Tariff Agent execution complete. Revenue Gain: {tariff_results.get('revenue_gain_pct', 'N/A')}%")

    # Monitoring Agent
    logger.info("[Pipeline] Starting Phase 5: Closed-Loop Monitoring & Learning Agent...")
    from src.utils import load_csv
    df = load_csv(FEATURES_CSV)

    monitor = MonitoringAgent()
    episode_results = monitor.run_multi_episode_simulation(
        df=df,
        demand_agent=demand_agent,
        tariff_agent=tariff_agent,
        num_episodes=num_episodes,
    )

    # Plot learning curve
    monitor.plot_learning_curve()

    # Export all results
    monitor.export_results()

    logger.info("=========================================")
    logger.info("  End-to-End Pipeline Execution Complete")
    logger.info("=========================================")
    logger.info(f"Results successfully exported to: {OUTPUTS_DIR}")

    return {
        "demand_agent": demand_agent,
        "tariff_agent": tariff_agent,
        "monitoring_agent": monitor,
        "demand_results": demand_results,
        "tariff_results": tariff_results,
        "episode_results": episode_results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EV Charging Tariff Optimization Pipeline")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing step")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature engineering step")
    parser.add_argument("--models", nargs="+", default=["xgboost", "lightgbm", "random_forest"],
                       help="Models to train: xgboost, lightgbm, random_forest, lstm")
    parser.add_argument("--episodes", type=int, default=10, help="Number of feedback episodes")

    args = parser.parse_args()

    results = run_full_pipeline(
        skip_preprocessing=args.skip_preprocess,
        skip_feature_engineering=args.skip_features,
        train_models=args.models,
        num_episodes=args.episodes,
    )
