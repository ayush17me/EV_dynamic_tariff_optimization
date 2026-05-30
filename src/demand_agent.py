"""
Agent 1: Demand Prediction Agent
Predicts future charging demand and station utilization using ML models.

Outputs:
- Predicted utilization rate
- Congestion probability (binary)
- Expected charging load (kWh)

Models:
- XGBoost (tabular baseline)
- LSTM/GRU (time-series sequential)
- Random Forest (ensemble comparison)

Evaluation Metrics:
- RMSE: Penalizes large errors in predicted station utilization
- MAE: Average absolute error in predicted demand
- R² Score: Explained variance in actual charging demand
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from src.config import (
    DEMAND_MODEL_CONFIG, ALL_FEATURES, RANDOM_SEED,
    FEATURES_CSV, MODELS_DIR, OUTPUTS_DIR,
)
from src.utils import (
    get_logger, load_csv, save_csv, save_model, load_model,
    compute_regression_metrics, set_seed, data_quality_report,
)

logger = get_logger("demand_agent")


class DemandPredictionAgent:
    """
    Agent 1: Demand Prediction

    Predicts station utilization, congestion probability, and
    expected charging load for the next time period.
    """

    def __init__(self, config: dict = DEMAND_MODEL_CONFIG):
        self.config = config
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.metrics_history = []
        set_seed(RANDOM_SEED)

    # ──────────────────────────────────────
    #  Data Preparation
    # ──────────────────────────────────────

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = "total_energy_kwh",
        feature_cols: Optional[list] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare train/val/test splits using time-based splitting.

        IMPORTANT: Using standard TimeSeriesSplit chronological validation to prevent future-data leakage (shuffling violates causality)
        This prevents future data leakage.
        """
        logger.info("Preparing data for demand prediction...")

        # Select features
        if feature_cols is None:
            feature_cols = [c for c in ALL_FEATURES if c in df.columns]

        if not feature_cols:
            # Fallback: use all numeric columns except target
            feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                          if c != target_col]

        self.feature_columns = feature_cols
        logger.info(f"  Using {len(feature_cols)} features: {feature_cols[:10]}...")

        # Drop rows with NaN in features or target
        df_clean = df[feature_cols + [target_col]].dropna()
        logger.info(f"  Clean samples: {len(df_clean)} (dropped {len(df) - len(df_clean)} NaN rows)")

        X = df_clean[feature_cols].values
        y = df_clean[target_col].values

        # ── Time-based split ──
        split_config = self.config["train_test_split"]
        n = len(X)
        train_end = int(n * split_config["train_ratio"])
        val_end = train_end + int(n * split_config["val_ratio"])

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        logger.info(f"  Split → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

        # ── Scale features ──
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test

    # ──────────────────────────────────────
    #  Model Training
    # ──────────────────────────────────────

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Any:
        """Train XGBoost regressor for demand prediction."""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            logger.error("XGBoost not installed. Run: pip install xgboost")
            return None

        logger.info("Training XGBoost model...")
        params = self.config["xgboost"]

        model = XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            random_state=RANDOM_SEED,
            verbosity=1,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        self.models["xgboost"] = model
        logger.info("  XGBoost training complete ")
        return model

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> RandomForestRegressor:
        """Train Random Forest as a comparison baseline."""
        logger.info("Training Random Forest model...")

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        self.models["random_forest"] = model
        logger.info("  Random Forest training complete ")
        return model

    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Any:
        """Train LightGBM regressor for demand prediction."""
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            logger.error("LightGBM not installed. Run: pip install lightgbm")
            return None

        logger.info("Training LightGBM model...")
        params = self.config["lightgbm"]

        model = LGBMRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            random_state=RANDOM_SEED,
            verbosity=-1,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )

        self.models["lightgbm"] = model
        logger.info("  LightGBM training complete ")
        return model

    def train_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Any:
        """
        Train LSTM model for sequential demand forecasting.

        Reshapes data to (samples, sequence_length, features) for LSTM input.
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        except ImportError:
            logger.error("TensorFlow not installed. Run: pip install tensorflow")
            return None

        logger.info("Training LSTM model...")
        params = self.config["lstm"]
        seq_len = params["sequence_length"]

        # ── Reshape to sequences ──
        def create_sequences(X, y, seq_length):
            Xs, ys = [], []
            for i in range(len(X) - seq_length):
                Xs.append(X[i:i + seq_length])
                ys.append(y[i + seq_length])
            return np.array(Xs), np.array(ys)

        X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_len)

        logger.info(f"  Sequence shapes → Train: {X_train_seq.shape}, Val: {X_val_seq.shape}")

        # ── Build LSTM ──
        model = Sequential([
            LSTM(params["hidden_units"], return_sequences=True,
                 input_shape=(seq_len, X_train.shape[1])),
            Dropout(params["dropout"]),
            LSTM(params["hidden_units"] // 2, return_sequences=False),
            Dropout(params["dropout"]),
            Dense(64, activation="relu"),
            Dense(1),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
            loss="mse",
            metrics=["mae"],
        )

        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
        ]

        model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=callbacks,
            verbose=1,
        )

        self.models["lstm"] = model
        self._lstm_seq_len = seq_len
        logger.info("  LSTM training complete ")
        return model

    # ──────────────────────────────────────
    #  Prediction
    # ──────────────────────────────────────

    def predict(
        self,
        X: np.ndarray,
        model_name: str = "xgboost",
    ) -> np.ndarray:
        """Generate demand predictions using the specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained yet. "
                           f"Available: {list(self.models.keys())}")

        model = self.models[model_name]

        if model_name == "lstm":
            # LSTM needs sequences
            seq_len = self._lstm_seq_len
            if len(X) < seq_len:
                raise ValueError(f"Need at least {seq_len} samples for LSTM prediction")
            # Create sequences from the last seq_len samples
            X_seq = np.array([X[i:i + seq_len] for i in range(len(X) - seq_len + 1)])
            predictions = model.predict(X_seq).flatten()
        else:
            predictions = model.predict(X)

        return predictions

    def predict_congestion(
        self,
        utilization_predictions: np.ndarray,
        threshold: float = 0.80,
    ) -> np.ndarray:
        """
        Predict congestion probability based on predicted utilization.

        Returns binary: 1 if predicted utilization > threshold, else 0.
        """
        return (utilization_predictions > threshold).astype(int)

    # ──────────────────────────────────────
    #  Evaluation
    # ──────────────────────────────────────

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "xgboost",
    ) -> dict:
        """Evaluate a trained model and return RMSE, MAE, R²."""
        logger.info(f"Evaluating {model_name}...")
        y_pred = self.predict(X_test, model_name)

        # Handle LSTM which produces fewer predictions
        if len(y_pred) < len(y_test):
            y_test = y_test[-len(y_pred):]

        metrics = compute_regression_metrics(y_test, y_pred)
        metrics["model"] = model_name
        self.metrics_history.append(metrics)

        return metrics

    def evaluate_all_models(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        """Evaluate all trained models and return comparison table."""
        results = []
        for model_name in self.models:
            metrics = self.evaluate(X_test, y_test, model_name)
            results.append(metrics)

        results_df = pd.DataFrame(results)
        logger.info(f"\n  Model Comparison:\n{results_df.to_string()}")
        return results_df

    # ──────────────────────────────────────
    #  Feature Importance
    # ──────────────────────────────────────

    def get_feature_importance(self, model_name: str = "xgboost") -> pd.DataFrame:
        """Get feature importance from tree-based models."""
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found")

        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame({
                "feature": self.feature_columns,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
            return importance
        else:
            logger.warning(f"Model '{model_name}' does not have feature_importances_")
            return pd.DataFrame()

    # ──────────────────────────────────────
    #  Save / Load
    # ──────────────────────────────────────

    def save(self, name: str = "demand_agent"):
        """Save the agent state (models + scaler)."""
        save_model(self, name)
        logger.info(f"Demand agent saved as '{name}'")

    @classmethod
    def load(cls, name: str = "demand_agent") -> "DemandPredictionAgent":
        """Load a previously saved agent."""
        return load_model(name)


# ══════════════════════════════════════════════
#  Convenience Runner
# ══════════════════════════════════════════════

def run_demand_prediction(
    data_path: Path = FEATURES_CSV,
    target_col: str = "total_energy_kwh",
    train_models: list = ["xgboost", "lightgbm", "random_forest"],
) -> Tuple[DemandPredictionAgent, pd.DataFrame]:
    """
    Run the full demand prediction pipeline:
    1. Load engineered features
    2. Prepare train/val/test splits
    3. Train models
    4. Evaluate and compare
    5. Save best model
    """
    logger.info("[Demand Prediction Agent] Initializing model training and evaluation...")

    # Load data
    df = load_csv(data_path)

    # Initialize agent
    agent = DemandPredictionAgent()

    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = agent.prepare_data(df, target_col)

    # Train models
    if "xgboost" in train_models:
        agent.train_xgboost(X_train, y_train, X_val, y_val)

    if "lightgbm" in train_models:
        agent.train_lightgbm(X_train, y_train, X_val, y_val)

    if "random_forest" in train_models:
        agent.train_random_forest(X_train, y_train)

    if "lstm" in train_models:
        agent.train_lstm(X_train, y_train, X_val, y_val)

    # Evaluate all
    results = agent.evaluate_all_models(X_test, y_test)

    # Save results
    save_csv(results, OUTPUTS_DIR / "demand_metrics.csv")

    # Save agent
    agent.save()

    logger.info("[Demand Prediction Agent] Model training and analysis complete.")

    return agent, results


# ─────────────────────────────────────────────
if __name__ == "__main__":
    agent, results = run_demand_prediction()
    print("\nModel Comparison:")
    print(results)
