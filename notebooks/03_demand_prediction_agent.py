# # Demand Prediction and Congestion Modeling
#
# **Focus:** Training and evaluating machine learning models to forecast station energy demand and utilization rates.
#
# **Models:** XGBoost (gradient boosting) + LightGBM (gradient boosting) + Random Forest (ensemble baseline)
#
# **Metrics:** RMSE, MAE, R² Score

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
from src.demand_agent import DemandPredictionAgent, run_demand_prediction

setup_plot_style()

# ## Loading Engineered Features

df = load_csv(FEATURES_CSV)
print(f"Dataset shape: {df.shape}")

# Show available features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nNumeric features available: {len(numeric_cols)}")
for c in numeric_cols:
    print(f"  {c}: non-null={df[c].notna().sum()}, mean={df[c].mean():.3f}")

# ## Initializing Agent and Splitting Dataset (Chronological Split)

agent = DemandPredictionAgent()

# Use energy_kwh as target (available for both ACN and UrbanEV)
target_col = "energy_kwh"
X_train, X_val, X_test, y_train, y_val, y_test = agent.prepare_data(
    df, target_col=target_col
)

print(f"\nFeatures used ({len(agent.feature_columns)}):")
for f in agent.feature_columns:
    print(f"  • {f}")

# ## XGBoost Regressor Training

xgb_model = agent.train_xgboost(X_train, y_train, X_val, y_val)

# Evaluate
xgb_metrics = agent.evaluate(X_test, y_test, "xgboost")
print(f"\nXGBoost Results:")
for k, v in xgb_metrics.items():
    if k != "model":
        print(f"  {k}: {v:.4f}")

# ## Random Forest Regressor Baseline

rf_model = agent.train_random_forest(X_train, y_train)

# Evaluate
rf_metrics = agent.evaluate(X_test, y_test, "random_forest")
print(f"\nRandom Forest Results:")
for k, v in rf_metrics.items():
    if k != "model":
        print(f"  {k}: {v:.4f}")

# ## LightGBM Regressor Training

lgb_model = agent.train_lightgbm(X_train, y_train, X_val, y_val)

# Evaluate
lgb_metrics = agent.evaluate(X_test, y_test, "lightgbm")
print(f"\nLightGBM Results:")
for k, v in lgb_metrics.items():
    if k != "model":
        print(f"  {k}: {v:.4f}")

# ## Regression Performance Metrics Comparison

results = agent.evaluate_all_models(X_test, y_test)
print("\n" + "=" * 50)
print("  MODEL COMPARISON")
print("=" * 50)
print(results.to_string(index=False))

# Save metrics
save_csv(results, OUTPUTS_DIR / "demand_metrics.csv")

# Best model
best = results.sort_values("RMSE").iloc[0]
print(f"\n🏆 Best Model: {best['model']} (RMSE: {best['RMSE']:.4f}, R²: {best['R2']:.4f})")

# ## Feature Importance Analysis (Tree-based Models)

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

for i, model_name in enumerate(["xgboost", "lightgbm", "random_forest"]):
    importance = agent.get_feature_importance(model_name)
    if not importance.empty:
        top_n = importance.head(15)
        axes[i].barh(top_n["feature"], top_n["importance"], color="#3498db", alpha=0.8)
        axes[i].set_title(f"{model_name.replace('_', ' ').title()} — Top 15 Features",
                         fontsize=13, fontweight="bold")
        axes[i].set_xlabel("Importance")
        axes[i].invert_yaxis()
        axes[i].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(str(OUTPUTS_DIR / "demand_feature_importance.png"), dpi=150, bbox_inches="tight")
plt.show()

# ## Scatter Diagnostics: Predicted vs. Actual Demand

fig, axes = plt.subplots(1, 3, figsize=(24, 7))

for i, model_name in enumerate(["xgboost", "lightgbm", "random_forest"]):
    y_pred = agent.predict(X_test, model_name)

    # Scatter plot
    axes[i].scatter(y_test, y_pred, alpha=0.1, s=5, color="#3498db")
    max_val = max(y_test.max(), y_pred.max())
    axes[i].plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect prediction")
    axes[i].set_title(f"{model_name.replace('_', ' ').title()}: Predicted vs Actual",
                     fontsize=13, fontweight="bold")
    axes[i].set_xlabel("Actual Energy (kWh)")
    axes[i].set_ylabel("Predicted Energy (kWh)")
    axes[i].legend()
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(str(OUTPUTS_DIR / "demand_prediction_scatter.png"), dpi=150, bbox_inches="tight")
plt.show()

# ## Saving Demand Prediction Model State

agent.save()
print("Demand Prediction Agent saved ✓")
print(f"Models trained: {list(agent.models.keys())}")
